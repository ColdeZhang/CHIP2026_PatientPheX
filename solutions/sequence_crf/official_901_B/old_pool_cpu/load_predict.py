#!/usr/bin/env python3
"""Inference-only frozen train64 CRF and original CRF+Sap decoder.

No training documents are opened. A SHA-checked pipeline supplies all learned
state. CRFsuite's external model filename is explicitly rebound after loading.
Raw Sap candidates and CRF confidence remain separate from selected entities.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

TRAIN_SHA = "cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2"
HPO_SHA = "a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b"
PIPELINE_SHA = "8753c5f2b719094dea4bdcbb6d34ebb48a4ffb97b61508ac52e80f8592f75b01"
CRF_SHA = "ee0f34ba65ae1fbaf181c01a1600c8afb2768410413582bcae71ed77e7c846e7"
SEQUENCE_SHA = "41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0"
ONTOLOGY_SHA = "56cdcb45dffd821bfa7fce6ac54b28364624ae7b603058487ecb3485c5878f7a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load_module(name, path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen source mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    raise RuntimeError("Training is forbidden in this inference-only entry")


def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_pipeline(sequence, pipeline, crf_model):
    if sha(pipeline) != PIPELINE_SHA or sha(crf_model) != CRF_SHA:
        raise ValueError("Frozen train64 pipeline/model hash mismatch")
    with open(pipeline, "rb") as handle:
        state = pickle.load(handle)
    assert state["train_sha256"] == TRAIN_SHA and state["hpo_sha256"] == HPO_SHA
    assert state["fingerprint"] == {"train_sha256": TRAIN_SHA, "hpo_sha256": HPO_SHA,
                                   "source_sha256": SEQUENCE_SHA, "iterations": 80}
    linker = sequence.Linker.__new__(sequence.Linker)
    linker.__dict__.update(state["linker_state"])
    crf, vectorizer, relation, stats = state["estimators"]
    original_path = crf.model_filename
    crf.model_filename = str(Path(crf_model).resolve())
    crf.modelfile.name = crf.model_filename
    crf.modelfile.auto = False
    crf._tagger = None
    crf._info_cached = None
    crf.fit = relation.fit = forbidden
    sequence.fit = sequence.fit_relation = sequence.Linker.__init__ = forbidden
    assert linker.version == "hp/releases/2026-06-23"
    return crf, linker, vectorizer, relation, stats, original_path


def confidences(sequence, doc, crf, linker):
    """Replay the frozen BIO decoder, preserving every accepted CRF score."""
    result = {}
    for passage in doc.get("full_text", []):
        tokens = sequence.tokenize(passage)
        if not tokens:
            continue
        feats = sequence.features(tokens, passage, linker)
        for a, b in sequence.chunks(tokens):
            labels = crf.predict_single(feats[a:b])
            marginals = crf.predict_marginals_single(feats[a:b])
            start = None
            for j in range(len(labels) + 1):
                label = labels[j] if j < len(labels) else "O"
                if start is not None and label != "I":
                    offset, end = tokens[a + start][1], tokens[a + j - 1][2]
                    poff = int(passage.get("offset", 0))
                    text = passage["text"][offset - poff:end - poff]
                    probability = float(np.mean([marginals[k].get(labels[k], 0.) for k in range(start, j)]))
                    identifier, similarity = linker.link(text)
                    if text.strip() and any(c.isalpha() for c in text) and probability >= .55:
                        key = (offset, end - offset, identifier)
                        result.setdefault(key, {"offset": offset, "length": end - offset,
                            "identifier": identifier, "text": text,
                            "ner_probability": probability, "link_similarity": similarity})
                    start = None
                if label == "B" or (label == "I" and start is None):
                    start = j
    return [result[k] for k in sorted(result)]


def augment(sequence, ontology, doc, pred, raw, vectorizer, relation):
    entities = list(pred["entities"])
    accepted = []
    for c in raw:
        if c["exact"] or c["probability"] < .65 or c["similarity"] < .92:
            continue
        _, passage = sequence.passage_for(doc, c["offset"])
        off = int(passage.get("offset", 0))
        if ontology.is_negated(passage.get("text", ""), c["offset"] - off,
                               c["offset"] - off + c["length"]):
            continue
        accepted.append(c)
    for c in sorted(accepted, key=lambda c: (-(.65*c["probability"]+.3*c["similarity"]+
                        .025*min(4, len(ontology.TOKEN.findall(c["text"])))), -c["length"], c["offset"])):
        if any(c["offset"] < e["offset"]+e["length"] and e["offset"] < c["offset"]+c["length"] for e in entities):
            continue
        entities.append({k: c[k] for k in ["identifier", "offset", "length", "text"]} |
                        {"type": "Phenotype", "note": None})
    entities.sort(key=lambda e: (e["offset"], e["length"], e["identifier"]))
    values = {p["patient_id"]: set() for p in doc.get("patient", [])}
    pairs = [(e, p) for e in entities for p in doc.get("patient", [])]
    if pairs:
        rows = [sequence.pair_features(doc, e, p) for e, p in pairs]
        probs = relation.predict_proba(vectorizer.transform(rows))[:, 1]
        for (entity, patient), probability in zip(pairs, probs):
            if probability >= .55:
                values[patient["patient_id"]].update(sequence.association_keys(entity))
    return {"pmc_id": doc["pmc_id"], "pmid": doc.get("pmid"), "entities": entities,
            "association": [{"patient_id": p["patient_id"], "phenotype": sorted(values[p["patient_id"]])}
                            for p in doc.get("patient", [])]}


def main():
    ap = argparse.ArgumentParser()
    for name in ["input", "input-sha256", "pipeline", "crf-model", "sequence-source", "ontology-source", "output-dir"]:
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--sap-candidates")
    ap.add_argument("--sap-candidates-sha256")
    ap.add_argument("--reference-crf")
    ap.add_argument("--reference-old-pool")
    args = ap.parse_args()
    started = time.monotonic()
    assert sha(args.input) == args.input_sha256
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "manifest.json").exists(), "Output already complete; use a fresh directory"
    sequence = load_module("frozen_sequence", args.sequence_source, SEQUENCE_SHA)
    ontology = load_module("frozen_ontology", args.ontology_source, ONTOLOGY_SHA)
    docs = read_jsonl(args.input)
    assert all("entities" not in d and "association" not in d for d in docs)
    assert len({str(d["pmc_id"]) for d in docs}) == len(docs)
    crf, linker, vec, relation, stats, old_path = load_pipeline(sequence, args.pipeline, args.crf_model)
    predictions = [sequence.predict(d, crf, linker, vec, relation) for d in docs]
    confidence = [{"pmc_id": d["pmc_id"], "candidates": confidences(sequence, d, crf, linker)} for d in docs]
    for p, c in zip(predictions, confidence):
        assert [(e["offset"], e["length"], e["identifier"]) for e in p["entities"]] == [
            (e["offset"], e["length"], e["identifier"]) for e in c["candidates"]]
    write_jsonl(out / "crf.pred.jsonl", predictions)
    write_jsonl(out / "crf.confidence.jsonl", confidence)
    checks = {}
    if args.reference_crf:
        expected = read_jsonl(args.reference_crf)
        checks["crf_entities_exact"] = len(expected) == len(predictions) and all(
            p["pmc_id"] == e["pmc_id"] and p["entities"] == e["entities"] for p, e in zip(predictions, expected))
        assert checks["crf_entities_exact"]
    if args.sap_candidates:
        assert args.sap_candidates_sha256 and sha(args.sap_candidates) == args.sap_candidates_sha256
        raw = read_jsonl(args.sap_candidates)
        candidates = {str(d["pmc_id"]): d["candidates"] for d in raw}
        assert set(candidates) == {str(d["pmc_id"]) for d in docs}
        combined = [augment(sequence, ontology, d, p, candidates[str(d["pmc_id"])], vec, relation)
                    for d, p in zip(docs, predictions)]
        write_jsonl(out / "old-pool.pred.jsonl", combined)
        if args.reference_old_pool:
            checks["old_pool_json_exact"] = combined == read_jsonl(args.reference_old_pool)
            checks["old_pool_bytes_exact"] = sha(out / "old-pool.pred.jsonl") == sha(args.reference_old_pool)
            assert all(checks.values())
    manifest = {"config": vars(args), "source": ref(__file__),
        "dependencies": {"sequence": ref(args.sequence_source), "ontology": ref(args.ontology_source)},
        "inputs": {k: ref(getattr(args, k)) for k in ["input", "pipeline", "crf_model", "sap_candidates",
                   "reference_crf", "reference_old_pool"] if getattr(args, k)},
        "outputs": {p.name: ref(p) for p in out.glob("*.jsonl")}, "checks": checks,
        "training_source_sha256": TRAIN_SHA, "hpo_sha256": HPO_SHA,
        "documents": len(docs), "patients": sum(len(d.get("patient", [])) for d in docs),
        "crf_entities": sum(len(p["entities"]) for p in predictions),
        "saved_training_stats": stats, "original_crf_model_path": old_path,
        "rebound_crf_model_path": crf.model_filename, "training_called": False,
        "gold_or_train_documents_read": False, "gpu_used": False,
        "runtime_versions": sequence.runtime_versions(), "seconds": time.monotonic()-started}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"checks": checks, "documents": len(docs), "outputs": manifest["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
