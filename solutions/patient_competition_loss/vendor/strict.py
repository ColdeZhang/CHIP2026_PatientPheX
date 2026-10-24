"""Whole-span HPO lookup; no detector, learned head, substring search or NIL guess.

Only the explicitly supplied training file contributes supervised aliases.
The default identities freeze the first F2+F3 -> F1 experiment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HPO_SHA256 = "a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b"
TRAIN32_SHA256 = "38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a"
ROOT = "HP:0000118"
POLICY = "strict-whole-span-v1"
ID = re.compile(r"HP:[0-9]{7}\Z")
COMPOUND_ID = re.compile(r"\s*HP:[0-9]{7}\s*(?:;\s*HP:[0-9]{7}\s*)*\Z")
SYNONYM = re.compile(r'^synonym: "((?:[^"\\]|\\.)*)"\s+EXACT\b')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text):
    """Retain punctuation, digits and modifiers; normalize case and whitespace only."""
    return " ".join(text.casefold().split())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_span(doc, quote, offset):
    if not isinstance(quote, str) or not quote or not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("Expected nonempty exact quote and integer global offset")
    matches = []
    for index, passage in enumerate(doc["full_text"]):
        start = offset - int(passage["offset"])
        if start >= 0 and start + len(quote) <= len(passage["text"]):
            if passage["text"][start:start + len(quote)] == quote:
                matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"Quote must match exactly one source passage: {len(matches)} matches")
    return matches[0]


class HPO:
    def __init__(self, path, expected_sha256=HPO_SHA256):
        self.path = str(Path(path).resolve())
        self.sha256 = sha256(path)
        if self.sha256 != expected_sha256:
            raise ValueError("Unexpected HPO bytes")
        self.terms = {}
        current = None
        self.version = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("data-version: "):
                self.version = line.split(": ", 1)[1]
            if line == "[Term]":
                current = {"parents": [], "synonyms": [], "alt_ids": [], "obsolete": False}
            elif line.startswith("["):
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                    self.terms[line[4:]] = current
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("is_a: "):
                    current["parents"].append(line[6:].split()[0])
                elif line.startswith("alt_id: "):
                    current["alt_ids"].append(line[8:])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
                else:
                    match = SYNONYM.match(line)
                    if match:
                        # OBO escapes are decoded only for escaped quotes/backslashes.
                        current["synonyms"].append(re.sub(r'\\(["\\])', r'\1', match.group(1)))
        if self.version != "hp/releases/2026-06-23":
            raise ValueError(f"Unexpected HPO version: {self.version}")
        children = defaultdict(list)
        for hid, term in self.terms.items():
            for parent in term["parents"]:
                children[parent].append(hid)
        descendants, pending = set(), [ROOT]
        while pending:
            hid = pending.pop()
            if hid not in descendants:
                descendants.add(hid)
                pending.extend(children[hid])
        self.allowed = {hid for hid in descendants if not self.terms[hid]["obsolete"]}
        self.canonical = {hid: hid for hid in self.allowed}
        for hid in sorted(self.allowed):
            for alt in self.terms[hid]["alt_ids"]:
                if alt in self.canonical and self.canonical[alt] != hid:
                    raise ValueError(f"Ambiguous alt ID: {alt}")
                self.canonical[alt] = hid

    def identifiers(self, value):
        """Atomic compound validation: never silently drop one invalid component."""
        if value == "-1":
            return ("-1",)
        if not isinstance(value, str):
            raise ValueError("Identifier must be a string")
        parts = value.split(";")
        if not parts or any(not ID.fullmatch(part.strip()) for part in parts):
            raise ValueError(f"Malformed or mixed identifier: {value!r}")
        try:
            return tuple(sorted({self.canonical[part.strip()] for part in parts}))
        except KeyError as exc:
            raise ValueError(f"ID outside active official phenotype branch: {value!r}") from exc

    def aliases(self):
        for hid in sorted(self.allowed):
            if hid == ROOT:  # Generic branch root is not a specific patient phenotype.
                continue
            term = self.terms[hid]
            yield {"text": term["name"], "ids": (hid,), "source": "hpo_name"}
            for text in sorted(set(term["synonyms"])):
                yield {"text": text, "ids": (hid,), "source": "hpo_exact_synonym"}

    def entity_keys(self, identifier, text):
        """Typed join keys; official NIL uses the literal entity surface."""
        ids = self.identifiers(identifier)
        if ids == ("-1",):
            if not isinstance(text, str) or not text:
                raise ValueError("NIL entity needs nonempty literal text")
            return (("nil", text),)
        return tuple(("hpo", hid) for hid in ids)

    def relation_keys(self, value):
        """Only a complete syntactic HPO string is a concept list; text stays whole."""
        if not isinstance(value, str) or not value:
            raise ValueError("Relation phenotype must be a nonempty string")
        if COMPOUND_ID.fullmatch(value):
            return tuple(("hpo", hid) for hid in self.identifiers(value))
        if value == "-1":
            raise ValueError("Official NIL relation must contain original text, not -1")
        if value.startswith("HP:") and not any(char.isspace() for char in value):
            raise ValueError(f"Malformed HPO relation identifier: {value!r}")
        return (("nil", value),)


class StrictNormalizer:
    def __init__(self, hpo_path, train_path, *, expected_train_sha256=TRAIN32_SHA256):
        self.hpo = HPO(hpo_path)
        actual = sha256(train_path)
        if actual != expected_train_sha256:
            raise ValueError("Unexpected training bytes; freeze a new explicit hash for another split")
        self.exact = defaultdict(lambda: defaultdict(list))
        self.folded = defaultdict(lambda: defaultdict(list))
        self.training_ids = []
        self.excluded_training_entities = []
        counts = Counter()
        for record in self.hpo.aliases():
            self._add(record)
            counts[record["source"]] += 1
        for doc in read_jsonl(train_path):
            pmc = str(doc["pmc_id"])
            if pmc in self.training_ids:
                raise ValueError(f"Duplicate training document: {pmc}")
            self.training_ids.append(pmc)
            for index, entity in enumerate(doc.get("entities", [])):
                text = entity["text"]
                counts["train_entities_seen"] += 1
                try:
                    if entity["type"] != "Phenotype":
                        raise ValueError("Unexpected entity type")
                    if entity["length"] != len(text):
                        raise ValueError("Invalid training length")
                    verify_span(doc, text, entity["offset"])
                    ids = self.hpo.identifiers(entity["identifier"])
                except ValueError as exc:
                    self.excluded_training_entities.append({"pmc_id": pmc, "entity_index": index,
                        "identifier": entity["identifier"], "reason": str(exc)})
                    continue
                self._add({"text": text, "ids": ids, "source": "train_alias",
                           "pmc_id": pmc, "entity_index": index,
                           "original_identifier": entity["identifier"]})
                counts["train_entities"] += 1
                counts["train_nil_entities" if ids == ("-1",) else "train_hpo_entities"] += 1
        self.manifest = {
            "policy": POLICY, "hpo": {"path": self.hpo.path, "sha256": self.hpo.sha256,
            "version": self.hpo.version, "allowed_concepts": len(self.hpo.allowed)},
            "train": {"path": str(Path(train_path).resolve()), "sha256": actual,
                      "pmc_ids": sorted(self.training_ids), "documents": len(self.training_ids)},
            "counts": dict(counts), "exact_keys": len(self.exact), "folded_keys": len(self.folded),
            "ambiguous_exact_keys": sum(len(options) > 1 for options in self.exact.values()),
            "ambiguous_folded_keys": sum(len(options) > 1 for options in self.folded.values()),
            "fitted_model": False, "input_annotations_used": False,
            "excluded_training_entities": self.excluded_training_entities,
        }

    def _add(self, record):
        if not record["text"]:
            raise ValueError("Empty alias")
        self.exact[record["text"]][record["ids"]].append(record)
        self.folded[normalized(record["text"])][record["ids"]].append(record)

    def assert_disjoint_input(self, path):
        docs = read_jsonl(path)
        if any("entities" in doc or "association" in doc for doc in docs):
            raise ValueError("Expected input-only documents without annotation keys")
        ids = [str(doc["pmc_id"]) for doc in docs]
        if len(ids) != len(set(ids)) or set(ids) & set(self.training_ids):
            raise ValueError("Duplicate or training-overlapping held document IDs")
        return {"path": str(Path(path).resolve()), "sha256": sha256(path),
                "documents": len(ids), "pmc_ids": sorted(ids), "training_overlap": []}

    def normalize(self, quote):
        if not isinstance(quote, str) or not quote:
            raise ValueError("Expected nonempty quote")
        key = normalized(quote)
        # Merge all case/space-equivalent evidence to expose, never hide, conflicts.
        options = self.folded.get(key, {})
        candidates = [{"ids": list(ids), "evidence": evidence}
                      for ids, evidence in sorted(options.items())]
        result = {"quote": quote, "status": "abstain", "ids": [], "identifier": None,
                  "reason": "unmatched", "match": None, "candidates": candidates,
                  "policy": POLICY}
        if not options:
            return result
        if len(options) != 1:
            result["reason"] = "ambiguous_mapping"
            return result
        ids, evidence = next(iter(options.items()))
        exact = quote in self.exact
        if ids == ("-1",):
            # NIL requires literal, unanimous annotated evidence, never vector distance.
            if not exact or not any(row["source"] == "train_alias" and row["text"] == quote for row in evidence):
                result["reason"] = "nil_requires_literal_training_evidence"
                return result
        elif len(re.sub(r"\W", "", quote)) <= 3 and not exact:
            result["reason"] = "short_alias_requires_literal_case"
            return result
        result.update(status="nil" if ids == ("-1",) else "mapped", ids=list(ids),
                      identifier=";".join(ids), reason="unanimous_whole_span_alias",
                      match="literal" if exact else "case_whitespace")
        return result

    def normalize_grounded(self, doc, quote, offset):
        passage_index = verify_span(doc, quote, offset)
        return {**self.normalize(quote), "pmc_id": str(doc["pmc_id"]), "offset": offset,
                "length": len(quote), "passage_index": passage_index}

    def ambiguities(self):
        return [{"key": key, "candidates": [{"ids": list(ids), "evidence": rows}
                 for ids, rows in sorted(options.items())]}
                for key, options in sorted(self.folded.items()) if len(options) > 1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--held-input")
    parser.add_argument("--train-sha256", default=TRAIN32_SHA256)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    normalizer = StrictNormalizer(args.hpo, args.train, expected_train_sha256=args.train_sha256)
    manifest = dict(normalizer.manifest)
    if args.held_input:
        manifest["held_input_isolation"] = normalizer.assert_disjoint_input(args.held_input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "strict-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "ambiguities.jsonl").open("w", encoding="utf-8") as handle:
        for row in normalizer.ambiguities():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
