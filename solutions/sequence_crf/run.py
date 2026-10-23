#!/usr/bin/env python3
"""Fold-local CRF phenotype NER, HPO linking, and learned patient association.

No project-archive imports or cached dictionaries. All supervised resources are
built only from --train; --input annotations, if present, are never consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn_crfsuite

TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NEGATIVE = re.compile(r"\b(?:no|not|without|denied|denies|absent|absence of|negative for)\b", re.I)
FAMILY = re.compile(r"\b(?:mother|father|sister|brother|parent|parents|sibling)\b", re.I)
COMPOUND_HPO = re.compile(r"HP:[0-9]{7}(?:;HP:[0-9]{7})+")


def norm(text):
    return " ".join(m.group().lower() for m in TOKEN.finditer(text))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    return {"python": platform.python_version(),
            **{package: importlib.metadata.version(package) for package in
               ["numpy", "scipy", "scikit-learn", "sklearn-crfsuite", "python-crfsuite"]}}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_hpo(path):
    terms, item = {}, None
    version = None
    for raw in Path(path).read_text().splitlines():
        if raw.startswith("data-version:"):
            version = raw.split(":", 1)[1].strip()
        if raw == "[Term]":
            if item and "id" in item:
                terms[item["id"]] = item
            item = {"names": [], "parents": [], "alt": [], "obsolete": False}
        elif raw.startswith("["):
            if item and "id" in item:
                terms[item["id"]] = item
            item = None
        elif item is not None:
            if raw.startswith("id: "): item["id"] = raw[4:]
            elif raw.startswith("name: "): item["names"].append(raw[6:])
            elif raw.startswith("synonym: "):
                match = re.match(r'synonym: "((?:[^"\\]|\\.)*)" (\w+)', raw)
                if match and match[2] in {"EXACT", "RELATED"}:
                    item["names"].append(match[1].replace('\\"', '"'))
            elif raw.startswith("is_a: "): item["parents"].append(raw[6:].split()[0])
            elif raw.startswith("alt_id: "): item["alt"].append(raw[8:])
            elif raw == "is_obsolete: true": item["obsolete"] = True
    if item and "id" in item: terms[item["id"]] = item
    if version != "hp/releases/2026-06-23":
        raise ValueError("Expected official HPO hp/releases/2026-06-23")
    descendants = {"HP:0000118"}
    while True:
        new = {identifier for identifier, term in terms.items()
               if not term["obsolete"] and any(parent in descendants for parent in term["parents"])}
        if new <= descendants: break
        descendants |= new
    aliases = {alt: key for key in descendants for alt in terms[key]["alt"]}
    return terms, descendants, aliases, version


class Linker:
    def __init__(self, train, hpo):
        self.terms, self.allowed, self.alt, self.version = parse_hpo(hpo)
        self.aliases = defaultdict(Counter)
        for identifier in sorted(self.allowed):
            for name in self.terms[identifier]["names"]:
                if len(norm(name)) >= 3:
                    self.aliases[norm(name)][identifier] += 0.05
        self.trained_aliases = 0
        for doc in train:
            for entity in doc.get("entities", []):
                if str(entity.get("note", "")).upper() == "NO": continue
                identifier = self.canonical(entity.get("identifier", "-1"))
                if identifier and entity.get("text"):
                    self.aliases[norm(entity["text"])][identifier] += 1
                    self.trained_aliases += 1
        self.names = sorted(self.aliases)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                          lowercase=True, min_df=1, max_features=100000,
                                          sublinear_tf=True, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform(self.names).tocsr()
        self.cache = {}
        self.trie = {}
        for name in self.names:
            pieces = name.split()
            if not 1 <= len(pieces) <= 14: continue
            node = self.trie
            for word in pieces: node = node.setdefault(word, {})
            node[None] = True

    def canonical(self, identifier):
        if str(identifier) == "-1": return "-1"
        parts = [self.alt.get(x.strip(), x.strip()) for x in str(identifier).split(";")]
        parts = sorted(set(x for x in parts if x in self.allowed))
        return ";".join(parts) if parts else None

    def link(self, text):
        text = norm(text)
        if text in self.cache: return self.cache[text]
        if text in self.aliases:
            identifier = self.aliases[text].most_common(1)[0][0]
            result = (identifier, 1.0)
        elif len(text) < 4:
            result = ("-1", 0.0)
        else:
            query = self.vectorizer.transform([text])
            score = (self.matrix @ query.T).toarray().ravel()
            idx = int(score.argmax())
            value = float(score[idx])
            result = ((self.aliases[self.names[idx]].most_common(1)[0][0], value)
                      if value >= 0.60 else ("-1", value))
        self.cache[text] = result
        return result

    def gazetteer(self, words):
        flags = [dict(gaz_b=False, gaz_i=False, gaz_e=False, gaz_len=0) for _ in words]
        for i in range(len(words)):
            node = self.trie
            for j in range(i, min(len(words), i + 14)):
                node = node.get(words[j].lower())
                if node is None: break
                if None in node:
                    for k in range(i, j + 1):
                        flags[k]["gaz_b"] |= k == i
                        flags[k]["gaz_i"] |= k != i
                        flags[k]["gaz_e"] |= k == j
                        flags[k]["gaz_len"] = max(flags[k]["gaz_len"], j - i + 1)
        return flags


def tokenize(passage):
    start = int(passage.get("offset", 0))
    return [(m.group(), start + m.start(), start + m.end()) for m in TOKEN.finditer(passage.get("text", ""))]


def features(tokens, passage, linker):
    words = [t[0] for t in tokens]
    gaz = linker.gazetteer(words)
    result = []
    section = str(passage.get("section_type", ""))[:100]
    kind = str(passage.get("type", ""))[:50]
    for i, word in enumerate(words):
        low = word.lower()
        feat = {"bias": 1.0, "w": low, "shape": re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", word))),
                "upper": word.isupper(), "title": word.istitle(), "digit": word.isdigit(),
                "length": min(len(word), 25), "section": section, "ptype": kind,
                "pref2": low[:2], "pref3": low[:3], "suf2": low[-2:], "suf3": low[-3:], "suf4": low[-4:]}
        feat.update(gaz[i])
        for delta in (-2, -1, 1, 2):
            pos = i + delta
            if 0 <= pos < len(words):
                neighbor = words[pos]
                prefix = str(delta) + ":"
                feat[prefix + "w"] = neighbor.lower()
                feat[prefix + "title"] = neighbor.istitle()
                feat[prefix + "gaz"] = gaz[pos]["gaz_b"] or gaz[pos]["gaz_i"]
            else: feat[str(delta) + ":boundary"] = True
        result.append(feat)
    return result


def chunks(tokens, labels=None):
    start = 0
    for i, token in enumerate(tokens):
        end = i + 1
        if (token[0] in {".", "!", "?", ";"} and end - start >= 5) or end - start >= 180:
            yield start, end
            start = end
    if start < len(tokens): yield start, len(tokens)


def labels_for(tokens, entities, linker):
    labels = ["O"] * len(tokens)
    starts = {token[1]: i for i, token in enumerate(tokens)}
    ends = {token[2]: i + 1 for i, token in enumerate(tokens)}
    stats = Counter()
    for entity in sorted(entities, key=lambda e: -int(e.get("length", 0))):
        if str(entity.get("note", "")).upper() == "NO": continue
        if not linker.canonical(entity.get("identifier", "-1")): continue
        start = int(entity["offset"]); end = start + int(entity["length"])
        if start not in starts or end not in ends: continue
        a, b = starts[start], ends[end]
        if b <= a or any(label != "O" for label in labels[a:b]):
            stats["overlapping_gold_skipped"] += 1
            continue
        labels[a] = "B"
        for i in range(a + 1, b): labels[i] = "I"
        stats["gold_aligned"] += 1
    return labels, stats


def passage_for(doc, offset):
    for index, passage in enumerate(doc.get("full_text", [])):
        start = int(passage.get("offset", 0))
        if start <= offset < start + len(passage.get("text", "")):
            return index, passage
    return -1, {"offset": 0, "text": "", "section_type": ""}


def sentence_bounds(text, offset):
    left = max(text.rfind(".", 0, offset), text.rfind(";", 0, offset), text.rfind("\n", 0, offset)) + 1
    right_positions = [v for v in (text.find(".", offset), text.find(";", offset), text.find("\n", offset)) if v >= 0]
    right = min(right_positions) if right_positions else len(text)
    return left, right


def pair_features(doc, entity, patient):
    offset = int(entity["offset"])
    pidx, passage = passage_for(doc, offset)
    pstart = int(passage.get("offset", 0)); text = passage.get("text", "")
    left, right = sentence_bounds(text, offset - pstart)
    mentions = patient.get("mention", [])
    positions = [int(m["offset"]) for m in mentions if m.get("offset") is not None]
    same = [pos for pos in positions if pstart <= pos < pstart + len(text)]
    sent = [pos for pos in same if pstart + left <= pos <= pstart + right]
    dist = min([abs(pos - offset) for pos in positions] or [100000])
    prev = [pos for pos in positions if pos <= offset]
    preceding = offset - max(prev) if prev else 100000
    others = [int(m["offset"]) for other in doc.get("patient", []) if other["patient_id"] != patient["patient_id"]
              for m in other.get("mention", []) if m.get("offset") is not None]
    other_dist = min([abs(pos - offset) for pos in others] or [100000])
    sentence = text[left:right]
    return {"bias": 1.0, "patients": len(doc.get("patient", [])),
            "single": len(doc.get("patient", [])) == 1, "has_mentions": bool(positions),
            "same_passage": bool(same), "same_sentence": bool(sent),
            "log_dist": math.log1p(dist), "log_prev_dist": math.log1p(preceding),
            "nearest": dist <= other_dist, "closer_margin": max(-5.0, min(5.0, math.log1p(other_dist) - math.log1p(dist))),
            "other_same_passage": any(pstart <= pos < pstart + len(text) for pos in others),
            "section": str(passage.get("section_type", "")), "ptype": str(passage.get("type", "")),
            "early_passage": pidx < 3, "family_sentence": bool(FAMILY.search(sentence)),
            "case_sentence": bool(re.search(r"\b(?:patient|proband|case|presented|examination|born|admitted)\b", sentence, re.I)),
            "general_sentence": bool(re.search(r"\b(?:typically|usually|often|patients with|characterized by)\b", sentence, re.I))}


def association_keys(entity):
    identifier = str(entity["identifier"])
    return [entity["text"]] if identifier == "-1" else identifier.split(";")


def gold_associations(doc, linker=None):
    """Split only syntactically complete HPO compounds, never phenotype text."""
    result = {p["patient_id"]: set() for p in doc.get("patient", [])}
    for row in doc.get("association", []):
        values = result.setdefault(row["patient_id"], set())
        for raw in row.get("phenotype", []):
            for item in raw.split(";") if COMPOUND_HPO.fullmatch(raw) else [raw]:
                values.add(linker.alt.get(item, item) if linker is not None else item)
    return result


def fit_relation(train, linker):
    examples, targets = [], []
    for doc in train:
        gold = gold_associations(doc, linker)
        for entity in doc.get("entities", []):
            if str(entity.get("note", "")).upper() == "NO": continue
            identifier = linker.canonical(entity.get("identifier", "-1"))
            if not identifier: continue
            entity = dict(entity, identifier=identifier)
            for patient in doc.get("patient", []):
                examples.append(pair_features(doc, entity, patient))
                targets.append(any(key in gold.get(patient["patient_id"], set()) for key in association_keys(entity)))
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(examples)
    clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", random_state=20261023, max_iter=300)
    clf.fit(matrix, targets)
    return vectorizer, clf, {"relation_examples": len(targets), "relation_positive": sum(targets)}


def fit(train, hpo, work_dir, iterations):
    linker = Linker(train, hpo)
    x, y, stats = [], [], Counter()
    for doc in train:
        for passage in doc.get("full_text", []):
            tokens = tokenize(passage)
            if not tokens: continue
            all_features = features(tokens, passage, linker)
            labels, counts = labels_for(tokens, doc.get("entities", []), linker)
            stats.update(counts)
            for a, b in chunks(tokens):
                local_labels = labels[a:b]
                if local_labels[0] == "I": local_labels[0] = "B"
                x.append(all_features[a:b]); y.append(local_labels)
    stats["training_sequences"] = len(x)
    stats["training_tokens"] = sum(map(len, y))
    print(json.dumps({"stage": "crf_fit", **stats}), flush=True)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                               max_iterations=iterations, all_possible_transitions=True,
                               model_filename=str(work_dir / "crf.model"), verbose=False)
    crf.fit(x, y)
    del x, y
    relation_vectorizer, relation, relation_stats = fit_relation(train, linker)
    stats.update(relation_stats)
    stats["hpo_concepts"] = len(linker.allowed)
    stats["linking_aliases"] = len(linker.names)
    return crf, linker, relation_vectorizer, relation, dict(stats)


def predict(doc, crf, linker, vectorizer, relation):
    entities, seen = [], set()
    for passage in doc.get("full_text", []):
        tokens = tokenize(passage)
        if not tokens: continue
        feats = features(tokens, passage, linker)
        for a, b in chunks(tokens):
            labels = crf.predict_single(feats[a:b])
            marginal = crf.predict_marginals_single(feats[a:b])
            start = None
            for j in range(len(labels) + 1):
                label = labels[j] if j < len(labels) else "O"
                if start is not None and label != "I":
                    span_tokens = tokens[a + start:a + j]
                    offset, end = span_tokens[0][1], span_tokens[-1][2]
                    text = passage["text"][offset - int(passage.get("offset", 0)):end - int(passage.get("offset", 0))]
                    confidence = float(np.mean([marginal[k].get(labels[k], 0.0) for k in range(start, j)]))
                    identifier, link_score = linker.link(text)
                    if text.strip() and any(ch.isalpha() for ch in text) and confidence >= 0.55:
                        key = (offset, end - offset, identifier)
                        if key not in seen:
                            seen.add(key)
                            entities.append({"identifier": identifier, "type": "Phenotype", "offset": offset,
                                             "length": end - offset, "text": text, "note": None})
                    start = None
                if label == "B" or (label == "I" and start is None): start = j
    entities.sort(key=lambda entity: (entity["offset"], entity["length"], entity["identifier"]))
    associations = {patient["patient_id"]: set() for patient in doc.get("patient", [])}
    rows, pairs = [], []
    for entity in entities:
        for patient in doc.get("patient", []):
            rows.append(pair_features(doc, entity, patient))
            pairs.append((entity, patient))
    if rows:
        probs = relation.predict_proba(vectorizer.transform(rows))[:, 1]
        for prob, (entity, patient) in zip(probs, pairs):
            if prob >= 0.55:
                associations[patient["patient_id"]].update(association_keys(entity))
    return {"pmc_id": doc["pmc_id"], "pmid": doc.get("pmid"), "entities": entities,
            "association": [{"patient_id": patient["patient_id"], "phenotype": sorted(associations[patient["patient_id"]])}
                            for patient in doc.get("patient", [])]}


def load_pipeline(path, train_hash, hpo_hash, fingerprint=None):
    # Only load trusted checkpoints generated by this CLI. Store the custom
    # Linker as plain state so loading does not depend on __main__ module names.
    with Path(path).open("rb") as handle: state = pickle.load(handle)
    assert state["train_sha256"] == train_hash and state["hpo_sha256"] == hpo_hash
    if fingerprint is not None and state.get("fingerprint") != fingerprint:
        raise ValueError("Cached pipeline does not match implementation or training configuration")
    linker = Linker.__new__(Linker)
    linker.__dict__.update(state["linker_state"])
    crf, vectorizer, relation, stats = state["estimators"]
    return crf, linker, vectorizer, relation, stats


def main():
    parser = argparse.ArgumentParser()
    for key in ("train", "input", "hpo", "output", "work-dir"): parser.add_argument("--" + key, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    model_path = work / "pipeline.pkl"
    train_hash, hpo_hash = sha(args.train), sha(args.hpo)
    fingerprint = {"train_sha256": train_hash, "hpo_sha256": hpo_hash,
                   "source_sha256": sha(__file__), "iterations": args.iterations}
    if args.reuse_model and model_path.exists():
        crf, linker, vectorizer, relation, stats = load_pipeline(model_path, train_hash, hpo_hash, fingerprint)
    else:
        training = read_jsonl(args.train)
        pipeline = fit(training, args.hpo, work, args.iterations)
        crf, linker, vectorizer, relation, stats = pipeline
        with model_path.open("wb") as handle:
            pickle.dump({"format_version": 2, "fingerprint": fingerprint, "linker_state": linker.__dict__,
                         "estimators": (crf, vectorizer, relation, stats),
                         "train_sha256": train_hash, "hpo_sha256": hpo_hash}, handle, protocol=5)
    fitted = time.monotonic()
    inputs = read_jsonl(args.input)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    total_entities = total_links = 0
    with out.open("w", encoding="utf-8") as handle:
        for doc in inputs:
            pred = predict(doc, crf, linker, vectorizer, relation)
            total_entities += len(pred["entities"])
            total_links += sum(len(a["phenotype"]) for a in pred["association"])
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
    end = time.monotonic()
    manifest = {"method": "crfsuite-BIO+HPO-char-tfidf-linking+logistic-patient-relation", "seed": 20261023,
                "train": str(Path(args.train).resolve()), "train_sha256": train_hash,
                "input": str(Path(args.input).resolve()), "input_sha256": sha(args.input),
                "hpo_sha256": hpo_hash, "hpo_version": linker.version,
                "output": str(out.resolve()), "output_sha256": sha(out), "output_bytes": out.stat().st_size,
                "documents": len(inputs), "entities": total_entities, "patient_phenotypes": total_links,
                "fit_seconds": fitted - start, "predict_seconds": end - fitted, "total_seconds": end - start,
                "iterations": args.iterations, "ner_confidence": 0.55, "relation_threshold": 0.55,
                "linker_threshold": 0.60, "training_stats": stats,
                "source_sha256": sha(__file__), "runtime_versions": runtime_versions(),
                "input_annotations_used": False}
    (work / (out.stem + "-manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
