#!/usr/bin/env python3
"""Validate a PatientPheX JSONL prediction against its exact input documents."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
from typing import Any

HPO_ID = re.compile(r"HP:\d{7}\Z")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}: empty line at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {line_number}: {exc.msg}") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no documents")
    index_documents(rows)
    return rows


def index_documents(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row.get("pmc_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every document needs a nonempty string pmc_id")
        if identifier in indexed:
            raise ValueError(f"Duplicate pmc_id: {identifier}")
        indexed[identifier] = row
    return indexed


def read_hpo(path: str | Path, root: str = "HP:0000118") -> dict[str, Any]:
    """Parse only OBO fields needed for canonical branch membership validation."""
    terms: dict[str, dict[str, Any]] = {}
    version = None
    term: dict[str, Any] | None = None

    def flush() -> None:
        if term and "id" in term:
            terms[term["id"]] = term

    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            if line.startswith("data-version: "):
                version = line.split(": ", 1)[1]
            elif line.startswith("["):
                flush()
                term = {"parents": [], "alt_ids": [], "obsolete": False} if line == "[Term]" else None
            elif term is not None:
                if line.startswith("id: "):
                    term["id"] = line[4:]
                elif line.startswith("name: "):
                    term["name"] = line[6:]
                elif line.startswith("is_a: "):
                    term["parents"].append(line[6:].split()[0])
                elif line.startswith("alt_id: "):
                    term["alt_ids"].append(line[8:])
                elif line == "is_obsolete: true":
                    term["obsolete"] = True
    flush()
    if root not in terms:
        raise ValueError(f"HPO branch root missing: {root}")
    children: dict[str, list[str]] = defaultdict(list)
    for identifier, item in terms.items():
        for parent in item["parents"]:
            children[parent].append(identifier)
    descendants = set()
    pending = deque([root])
    while pending:
        identifier = pending.popleft()
        if identifier not in descendants:
            descendants.add(identifier)
            pending.extend(children[identifier])
    allowed = {identifier for identifier in descendants if not terms[identifier]["obsolete"]}
    canonical = {identifier: identifier for identifier in allowed}
    for identifier in allowed:
        for alias in terms[identifier]["alt_ids"]:
            canonical[alias] = identifier
    return {"version": version, "root": root, "terms": terms, "allowed": allowed, "canonical": canonical}


def passage_for_span(document: dict[str, Any], offset: int, length: int) -> dict[str, Any] | None:
    for passage in document.get("full_text", []):
        start = passage["offset"]
        if start <= offset and offset + length <= start + len(passage["text"]):
            return passage
    return None


def validate_predictions(inputs: list[dict[str, Any]], predictions: list[dict[str, Any]],
                         hpo: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return finite error/warning summaries; never silently zip or drop documents."""
    source = index_documents(inputs)
    predicted = index_documents(predictions)
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()

    def issue(message: str, warning: bool = False) -> None:
        kind = "warnings" if warning else "errors"
        counts[kind] += 1
        collection = warnings if warning else errors
        if len(collection) < 50:
            collection.append(message)

    missing = sorted(set(source) - set(predicted))
    extra = sorted(set(predicted) - set(source))
    if missing:
        issue(f"Missing documents ({len(missing)}): {missing[:8]}")
    if extra:
        issue(f"Unexpected documents ({len(extra)}): {extra[:8]}")

    def check_hpo(identifier: str, prefix: str) -> None:
        if not HPO_ID.fullmatch(identifier):
            issue(f"{prefix}: malformed HPO ID {identifier!r}")
        elif hpo is not None:
            canonical = hpo["canonical"].get(identifier)
            if canonical is None:
                issue(f"{prefix}: HPO ID outside allowed current branch: {identifier}")
            elif canonical != identifier:
                issue(f"{prefix}: use canonical ID {canonical} instead of alt_id {identifier}", warning=True)

    for pmc_id in sorted(set(source) & set(predicted)):
        gold, row = source[pmc_id], predicted[pmc_id]
        counts["documents"] += 1
        if "pmid" not in row or row["pmid"] != gold.get("pmid"):
            issue(f"{pmc_id}: pmid missing or differs from input")
        entities = row.get("entities")
        if not isinstance(entities, list):
            issue(f"{pmc_id}: entities must be a list")
            entities = []
        seen_entities = set()
        for index, entity in enumerate(entities):
            prefix = f"{pmc_id}: entity {index}"
            counts["entities"] += 1
            if not isinstance(entity, dict):
                issue(f"{prefix}: must be an object")
                continue
            if entity.get("type") != "Phenotype":
                issue(f"{prefix}: type must be Phenotype")
            if "note" not in entity or entity["note"] not in (None, "NO"):
                issue(f"{prefix}: note must be null or NO")
            identifier = entity.get("identifier")
            if not isinstance(identifier, str) or not identifier:
                issue(f"{prefix}: identifier must be a nonempty string")
            else:
                identifiers = identifier.split(";")
                if "-1" in identifiers and len(identifiers) != 1:
                    issue(f"{prefix}: -1 must not be mixed with HPO IDs")
                for value in identifiers:
                    if value != "-1":
                        check_hpo(value, prefix)
                if len(identifiers) != len(set(identifiers)):
                    issue(f"{prefix}: repeated ID in compound identifier", warning=True)
            offset, length, text = entity.get("offset"), entity.get("length"), entity.get("text")
            if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
                issue(f"{prefix}: offset/length must be nonnegative/positive integers")
                continue
            if not isinstance(text, str) or not text:
                issue(f"{prefix}: text must be a nonempty string")
                continue
            passage = passage_for_span(gold, offset, length)
            if passage is None:
                issue(f"{prefix}: span lies outside input passages")
            else:
                start = offset - passage["offset"]
                if passage["text"][start:start + length] != text:
                    issue(f"{prefix}: text differs from input slice")
            note = entity.get("note")
            key = (offset, length, identifier if isinstance(identifier, str) else None,
                   note if isinstance(note, str) or note is None else "<invalid>")
            if key in seen_entities:
                issue(f"{prefix}: duplicate entity", warning=True)
            seen_entities.add(key)

        expected = {patient["patient_id"] for patient in gold.get("patient", [])}
        associations = row.get("association")
        if not isinstance(associations, list):
            issue(f"{pmc_id}: association must be a list")
            associations = []
        found = set()
        for index, association in enumerate(associations):
            prefix = f"{pmc_id}: association {index}"
            if not isinstance(association, dict):
                issue(f"{prefix}: must be an object")
                continue
            patient_id = association.get("patient_id")
            if not isinstance(patient_id, str) or patient_id not in expected:
                issue(f"{prefix}: patient_id absent from input")
            elif patient_id in found:
                issue(f"{prefix}: duplicate patient record {patient_id}")
            else:
                found.add(patient_id)
            values = association.get("phenotype")
            if not isinstance(values, list):
                issue(f"{prefix}: phenotype must be a list")
                continue
            seen_values = set()
            for value in values:
                counts["relations"] += 1
                if not isinstance(value, str) or not value:
                    issue(f"{prefix}: phenotype items must be nonempty strings")
                    continue
                if value == "-1":
                    issue(f"{prefix}: association requires original text instead of -1")
                elif value.startswith("HP:"):
                    if ";" in value:
                        issue(f"{prefix}: association HPO IDs must be separate list items")
                    else:
                        check_hpo(value, prefix)
                if value in seen_values:
                    issue(f"{prefix}: duplicate phenotype", warning=True)
                seen_values.add(value)
        counts["patients"] += len(expected)
        absent = sorted(expected - found)
        if absent:
            issue(f"{pmc_id}: missing patient records {absent}")

    return {
        "valid": counts["errors"] == 0,
        "counts": dict(counts), "errors": errors, "warnings": warnings,
        "error_count": counts["errors"], "warning_count": counts["warnings"],
        "hpo_version": hpo["version"] if hpo else None,
        "hpo_root": hpo["root"] if hpo else None,
        "notes": ["Format/input consistency checks are not the platform evaluator.",
                  "Only the first 50 examples of each issue type are included."]
    }


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--hpo")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        hpo = read_hpo(args.hpo) if args.hpo else None
        if hpo and hpo["version"] != "hp/releases/2026-06-23":
            raise ValueError("The HPO file is not the competition-required 2026-06-23 version")
        result = validate_predictions(load_jsonl(args.input), load_jsonl(args.pred), hpo)
        result["input_sha256"] = file_sha256(args.input)
        result["prediction_sha256"] = file_sha256(args.pred)
        if args.hpo:
            result["hpo_sha256"] = file_sha256(args.hpo)
    except ValueError as exc:
        result = {"valid": False, "error_count": 1, "errors": [str(exc)]}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
