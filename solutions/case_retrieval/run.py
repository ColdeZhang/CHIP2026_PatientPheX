#!/usr/bin/env python3
"""Patient-centred sparse case retrieval. No external labelled corpus or old code."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOKEN = re.compile(r"[A-Za-z]+|\d+")
SENTENCE_END = re.compile(r"[.!?;]\s+")
NEGATION = re.compile(r"\b(?:no|without|denied|denies|absence of|negative for|no evidence of)\b", re.I)
TERMINAL = "__concepts__"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [str(row["pmc_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate pmc_id in input")
    return rows


def tokens(text):
    return [(m.group().lower(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def alias_key(text):
    return tuple(t[0] for t in tokens(text))


def load_hpo(path):
    terms = {}
    current = None
    version = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("data-version:"):
            version = line.split(":", 1)[1].strip()
        if line == "[Term]":
            current = {"parents": [], "aliases": [], "alt_ids": [], "obsolete": False}
        elif line.startswith("["):
            current = None
        elif current is not None:
            if line.startswith("id: HP:"):
                current["id"] = line[4:].strip()
                terms[current["id"]] = current
            elif line.startswith("name: "):
                current["name"] = line[6:]
                current["aliases"].append(line[6:])
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: "((?:\\.|[^"\\])*)" (\w+)', line)
                if match and match[2] in {"EXACT", "RELATED"}:
                    current["aliases"].append(match[1].replace('\\"', '"'))
            elif line.startswith("is_a: "):
                current["parents"].append(line.split()[1])
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line.split()[1])
            elif line == "is_obsolete: true":
                current["obsolete"] = True
    if not version or "2026-06-23" not in version:
        raise ValueError("This experiment requires HPO 2026-06-23")
    children = defaultdict(list)
    for identifier, term in terms.items():
        for parent in term["parents"]:
            children[parent].append(identifier)
    valid = set()
    stack = ["HP:0000118"]
    while stack:
        identifier = stack.pop()
        if identifier in valid:
            continue
        valid.add(identifier)
        stack.extend(children[identifier])
    valid = {i for i in valid if i in terms and not terms[i]["obsolete"]}
    canonical = {identifier: identifier for identifier in valid}
    for identifier in valid:
        canonical.update({alt: identifier for alt in terms[identifier]["alt_ids"]})
    return terms, valid, canonical, version


def normalize_id(value, canonical):
    if str(value) == "-1":
        return "-1"
    parts = [canonical.get(part.strip()) for part in str(value).split(";")]
    if not parts or any(part is None for part in parts):
        return None
    return ";".join(sorted(set(parts)))


def association_ids(values):
    """Split only complete legal compound HPO values; preserve free text labels."""
    return {part for value in values for part in
            (str(value).split(";") if re.fullmatch(r"HP:[0-9]{7}(?:;HP:[0-9]{7})+", str(value)) else [str(value)])}


def entity_ids(entity):
    identifier = str(entity["identifier"])
    return [entity["text"]] if identifier == "-1" else identifier.split(";")


class LexicalMemory:
    def __init__(self, terms, valid, canonical, training):
        self.canonical = canonical
        self.aliases = defaultdict(set)
        self.prior = defaultdict(Counter)
        self.contexts = defaultdict(list)
        self.correct = Counter()
        self.observed = Counter()
        self.stats = Counter()
        self.trie = {}
        for identifier in sorted(valid):
            for surface in terms[identifier]["aliases"]:
                key = alias_key(surface)
                if self.useful(key):
                    self.aliases[key].add(identifier)
        for doc in training:
            for entity in doc.get("entities", []):
                if str(entity.get("note", "")).upper() == "NO":
                    continue
                identifier = normalize_id(entity.get("identifier"), canonical)
                key = alias_key(entity.get("text", ""))
                if identifier is None or not self.useful(key):
                    continue
                self.aliases[key].add(identifier)
                self.prior[key][identifier] += 1
                for passage in doc["full_text"]:
                    begin = int(entity["offset"]) - int(passage["offset"])
                    if 0 <= begin and begin + int(entity["length"]) <= len(passage["text"]):
                        text = passage["text"][max(0, begin - 120): begin + int(entity["length"]) + 120]
                        self.contexts[key].append((identifier, set(alias_key(text))))
                        break
        for key, identifiers in self.aliases.items():
            node = self.trie
            for token in key:
                node = node.setdefault(token, {})
            node[TERMINAL] = tuple(sorted(identifiers))
        # Count boundary support in the training fold, never the validation inputs.
        for doc in training:
            positives = set()
            for entity in doc.get("entities", []):
                if str(entity.get("note", "")).upper() != "NO":
                    identifier = normalize_id(entity.get("identifier"), canonical)
                    if identifier is not None:
                        positives.add((int(entity["offset"]), int(entity["length"])))
            for passage in doc["full_text"]:
                for start, end, key, _ in self.match(passage["text"]):
                    self.observed[key] += 1
                    if (int(passage["offset"]) + start, end - start) in positives:
                        self.correct[key] += 1
        self.stats["alias_keys"] = len(self.aliases)
        self.stats["positive_contexts"] = sum(map(len, self.contexts.values()))

    @staticmethod
    def useful(key):
        return bool(key) and len(key) <= 14 and sum(map(len, key)) >= 3 and not all(k.isdigit() for k in key)

    def match(self, text):
        ts = tokens(text)
        for index, (_, begin, _) in enumerate(ts):
            node = self.trie
            for j in range(index, min(index + 14, len(ts))):
                token, start, end = ts[j]
                if j > index:
                    gap = text[ts[j-1][2]: start]
                    if re.search(r"[^\s\-/]", gap):
                        break
                if token not in node:
                    break
                node = node[token]
                if TERMINAL in node:
                    key = tuple(t[0] for t in ts[index:j+1])
                    yield begin, end, key, node[TERMINAL]

    def choose_id(self, key, identifiers, context):
        if len(identifiers) == 1:
            return identifiers[0]
        query = set(alias_key(context)) - set(key)
        scores = {identifier: 0.01 for identifier in identifiers}
        for identifier, words in self.contexts[key]:
            support = words - set(key)
            similarity = len(query & support) / max(1, math.sqrt(len(query) * len(support)))
            scores[identifier] = max(scores.get(identifier, 0), similarity)
        for identifier, count in self.prior[key].items():
            scores[identifier] = scores.get(identifier, 0) + min(count, 10) * 0.005
        return max(sorted(scores), key=lambda identifier: scores[identifier])

    def predict(self, doc):
        entities = []
        for passage in doc["full_text"]:
            text = passage["text"]
            candidates = []
            for start, end, key, identifiers in self.match(text):
                surface = text[start:end]
                # Acronyms are case-sensitive even though the token trie is not.
                if len(key) == 1 and len(key[0]) <= 3 and not surface.isupper():
                    continue
                prior = (self.correct[key] + 1.5) / (self.observed[key] + 2.0)
                if self.observed[key] >= 3 and prior < 0.15:
                    continue
                before = re.split(r"[.!?;]", text[max(0, start - 100):start])[-1]
                if NEGATION.search(before) and not re.search(r"\b(?:but|however|although|not only)\b", before, re.I):
                    continue
                context = text[max(0, start - 120):end + 120]
                identifier = self.choose_id(key, identifiers, context)
                confidence = prior if self.observed[key] else 0.66
                # Prefer supported complete spans to their nested partial matches.
                weight = (end-start) ** 0.4 * (0.65 + confidence)
                candidates.append((start, end, key, identifier, weight))
            selected = []
            for candidate in sorted(candidates, key=lambda x: (-x[4], -(x[1]-x[0]), x[0])):
                if any(candidate[0] < other[1] and other[0] < candidate[1] for other in selected):
                    continue
                selected.append(candidate)
            for start, end, _, identifier, _ in sorted(selected):
                entities.append({"identifier": identifier, "type": "Phenotype", "offset": int(passage["offset"]) + start,
                                 "length": end-start, "text": text[start:end], "note": None})
        unique = {(e["offset"], e["length"], e["identifier"]): e for e in entities}
        return [unique[key] for key in sorted(unique)]


def flatten_entities(doc, canonical):
    result = []
    for original in doc.get("entities", []):
        if str(original.get("note", "")).upper() == "NO":
            continue
        identifier = normalize_id(original.get("identifier"), canonical)
        if identifier is None:
            continue
        entity = dict(original)
        entity["identifier"] = identifier
        entity["offset"] = int(entity["offset"])
        entity["length"] = int(entity["length"])
        result.append(entity)
    return result


def mentions_by_patient(doc):
    result = {}
    for patient in doc.get("patient", []):
        spans = []
        for mention in patient.get("mention", []):
            spans.append((int(mention["offset"]), int(mention["offset"]) + int(mention["length"])))
        result[str(patient["patient_id"])] = sorted(spans)
    return result


def relative_bucket(distance):
    for boundary in [0, 40, 120, 300, 1000, 3000]:
        if distance <= boundary:
            return str(boundary)
    return "far"


def span_distance(a, b):
    return max(a[0] - b[1], b[0] - a[1], 0)


def patient_pair(doc, patient_id, entities, mention_map):
    """Represent one document/patient/concept as a case, not one unpaired sentence."""
    targets = mention_map.get(patient_id, [])
    others = [(pid, span) for pid, spans in mention_map.items() if pid != patient_id for span in spans]
    snippets = []
    features = Counter()
    features["patient_count=" + str(min(5, len(mention_map)))] = 1.0
    features["target_has_mentions=" + str(bool(targets))] = 1.0
    nearest_distance = 10**9
    target_same_passage = False
    target_same_sentence = False
    target_closest = False
    other_closest = False
    case_section = False
    discussion = False
    for entity in entities:
        e_span = (entity["offset"], entity["offset"] + entity["length"])
        target_distance = min((span_distance(e_span, span) for span in targets), default=10**9)
        other_distance = min((span_distance(e_span, span) for _, span in others), default=10**9)
        nearest_distance = min(nearest_distance, target_distance)
        target_closest |= target_distance < other_distance
        other_closest |= other_distance < target_distance
        for passage in doc["full_text"]:
            offset = int(passage["offset"])
            begin = entity["offset"] - offset
            if not (0 <= begin and begin + entity["length"] <= len(passage["text"])):
                continue
            text = passage["text"]
            section = str(passage.get("section_type", "")) + " " + str(passage.get("type", ""))
            case_section |= bool(re.search(r"case|result|patient|clinical", section, re.I))
            discussion |= bool(re.search(r"discuss|intro|background|conclu", section, re.I))
            passage_end = offset + len(text)
            local_targets = [span for span in targets if offset <= span[0] < passage_end]
            target_same_passage |= bool(local_targets)
            sentence_breaks = [0] + [m.end() for m in SENTENCE_END.finditer(text)] + [len(text)]
            sentence_index = max(0, bisect_right(sentence_breaks, begin) - 1)
            sentence_start = sentence_breaks[sentence_index] + offset
            sentence_end = sentence_breaks[min(sentence_index + 1, len(sentence_breaks)-1)] + offset
            target_same_sentence |= any(sentence_start <= span[0] < sentence_end for span in targets)
            left, right = max(0, begin-180), min(len(text), begin+entity["length"]+180)
            replacements = [(begin-left, begin+entity["length"]-left, " phenotype ")]
            for pid, spans in mention_map.items():
                for start, end in spans:
                    if offset+left <= start < end <= offset+right:
                        replacements.append((start-offset-left, end-offset-left, " targetpatient " if pid == patient_id else " otherpatient "))
            snippet = text[left:right]
            for start, end, replacement in sorted(replacements, reverse=True):
                snippet = snippet[:start] + replacement + snippet[end:]
            snippets.append((target_distance, snippet))
            break
    features["target_distance=" + relative_bucket(nearest_distance)] = 1.0
    features["target_same_passage=" + str(target_same_passage)] = 1.0
    features["target_same_sentence=" + str(target_same_sentence)] = 1.0
    features["target_closest=" + str(target_closest)] = 1.0
    features["other_closest=" + str(other_closest)] = 1.0
    features["case_section=" + str(case_section)] = 1.0
    features["discussion=" + str(discussion)] = 1.0
    snippets = [snippet for _, snippet in sorted(snippets, key=lambda item: item[0])[:3]]
    return " ".join(snippets) or "missing_evidence", dict(features)


def grouped_concepts(entities):
    groups = defaultdict(list)
    for entity in entities:
        for identifier in entity_ids(entity):
            groups[identifier].append(entity)
    return groups


class CaseRetriever:
    def __init__(self, training, canonical, neighbors=11, threshold=0.48):
        self.neighbors, self.threshold = neighbors, threshold
        texts, structures, labels, doc_ids = [], [], [], []
        for doc in training:
            targets = mentions_by_patient(doc)
            gold = {str(row["patient_id"]): association_ids(row.get("phenotype", [])) for row in doc.get("association", [])}
            concepts = grouped_concepts(flatten_entities(doc, canonical))
            for patient_id in targets:
                for concept, entities in sorted(concepts.items()):
                    text, features = patient_pair(doc, patient_id, entities, targets)
                    texts.append(text)
                    structures.append(features)
                    labels.append(float(concept in gold.get(patient_id, set())))
                    doc_ids.append(str(doc["pmc_id"]))
        if not texts:
            raise ValueError("No training cases available")
        self.text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=26000, min_df=2,
                                              sublinear_tf=True, dtype=np.float32, strip_accents="unicode")
        self.structure_vectorizer = DictVectorizer(dtype=np.float32)
        word_features = self.text_vectorizer.fit_transform(texts)
        structure_features = normalize(self.structure_vectorizer.fit_transform(structures))
        self.matrix = normalize(hstack([word_features * 0.65, structure_features * 1.3], format="csr"))
        self.labels = np.asarray(labels, dtype=np.float32)
        self.doc_ids = np.asarray(doc_ids)
        self.stats = {"training_cases": len(labels), "positive_cases": int(sum(labels)), "text_features": word_features.shape[1],
                      "structural_features": structure_features.shape[1], "neighbors": neighbors, "association_threshold": threshold}

    def predict(self, doc, entities):
        mention_map = mentions_by_patient(doc)
        concepts = grouped_concepts(entities)
        keys, texts, structures = [], [], []
        result = {patient_id: [] for patient_id in mention_map}
        for patient_id in mention_map:
            for concept, occurrences in sorted(concepts.items()):
                text, features = patient_pair(doc, patient_id, occurrences, mention_map)
                keys.append((patient_id, concept)); texts.append(text); structures.append(features)
        if not keys:
            return [{"patient_id": patient["patient_id"], "phenotype": []} for patient in doc.get("patient", [])]
        word_features = self.text_vectorizer.transform(texts)
        structure_features = normalize(self.structure_vectorizer.transform(structures))
        query = normalize(hstack([word_features * 0.65, structure_features * 1.3], format="csr"))
        for start in range(0, len(keys), 128):
            similarities = (query[start:start+128] @ self.matrix.T).toarray()
            for j, scores in enumerate(similarities):
                # At most two neighbours per training document: a long report must not dominate voting.
                candidates = np.argsort(scores)[::-1][:max(100, self.neighbors * 10)]
                selected, per_doc = [], Counter()
                for index in candidates:
                    if per_doc[self.doc_ids[index]] >= 2:
                        continue
                    selected.append(index); per_doc[self.doc_ids[index]] += 1
                    if len(selected) >= self.neighbors:
                        break
                selected = np.asarray(selected, dtype=int)
                weights = np.maximum(scores[selected], 0) ** 5
                probability = float(np.dot(weights, self.labels[selected]) / max(1e-8, float(weights.sum())))
                if probability >= self.threshold:
                    patient_id, concept = keys[start+j]
                    result[patient_id].append(concept)
        return [{"patient_id": patient["patient_id"], "phenotype": sorted(set(result[str(patient["patient_id"])]))}
                for patient in doc.get("patient", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=11)
    parser.add_argument("--association-threshold", type=float, default=0.48)
    args = parser.parse_args()
    began = time.time()
    training, inputs = read_jsonl(args.train), read_jsonl(args.input)
    overlap = {str(doc["pmc_id"]) for doc in training} & {str(doc["pmc_id"]) for doc in inputs}
    if overlap:
        raise ValueError("Training and prediction documents overlap; only disjoint runs are permitted")
    if any(doc.get("entities") or doc.get("association") for doc in inputs):
        raise ValueError("Prediction input contains labels; use the stripped shared input")
    terms, valid, canonical, version = load_hpo(args.hpo)
    lexical = LexicalMemory(terms, valid, canonical, training)
    retriever = CaseRetriever(training, canonical, args.neighbors, args.association_threshold)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    with output.open("w") as handle:
        for index, doc in enumerate(inputs):
            entities = lexical.predict(doc)
            associations = retriever.predict(doc, entities)
            prediction = {"pmc_id": doc["pmc_id"], "pmid": doc.get("pmid"), "entities": entities, "association": associations}
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            totals["documents"] += 1; totals["entities"] += len(entities)
            totals["patients"] += len(associations)
            totals["links"] += sum(len(row["phenotype"]) for row in associations)
            if (index+1) % 10 == 0:
                print(json.dumps({"completed": index+1, "elapsed_seconds": round(time.time()-began, 2)}), flush=True)
    manifest = {"method": "sparse patient case kNN", "version": 1, "hpo_version": version,
                "configuration": vars(args), "training_documents": len(training), "prediction_documents": len(inputs),
                "train_input_overlap": len(overlap), "labelled_prediction_input": False,
                "files": {name: {"path": getattr(args, name), "sha256": sha256(getattr(args, name)),
                                  "bytes": Path(getattr(args, name)).stat().st_size} for name in ["train", "input", "hpo", "output"]},
                "code_sha256": sha256(__file__), "lexical_memory": dict(lexical.stats), "case_memory": retriever.stats,
                "totals": dict(totals), "elapsed_seconds": round(time.time()-began, 3),
                "environment": {"python": sys.version.split()[0], "numpy": np.__version__}}
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"totals": dict(totals), "elapsed_seconds": manifest["elapsed_seconds"], "output_sha256": manifest["files"]["output"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
