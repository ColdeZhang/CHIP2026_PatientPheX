#!/usr/bin/env python3
"""BiomedBERT token classification with the same fold-local linker/relation head.

Uses only domain text pretraining plus the competition --train annotations.
The annotated validation set is never passed to this executable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import random
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(key, "false" if key == "TOKENIZERS_PARALLELISM" else "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from run import Linker, association_keys, fit_relation, pair_features, read_jsonl, runtime_versions, sha

LABELS = {0: "O", 1: "B", 2: "I"}


def document_chunks(doc, tokenizer, linker, length, training=False):
    samples, provenance = [], []
    for pi, passage in enumerate(doc.get("full_text", [])):
        text = passage.get("text", "")
        if not text.strip(): continue
        base = int(passage.get("offset", 0))
        encoding = tokenizer(text, truncation=True, max_length=length, stride=64,
                             return_overflowing_tokens=True, return_offsets_mapping=True)
        spans = []
        if training:
            for entity in sorted(doc.get("entities", []), key=lambda e: -int(e.get("length", 0))):
                if str(entity.get("note", "")).upper() == "NO": continue
                if not linker.canonical(entity.get("identifier", "-1")): continue
                start = int(entity["offset"]) - base; end = start + int(entity["length"])
                if 0 <= start < end <= len(text): spans.append((start, end))
        for ci, offsets in enumerate(encoding["offset_mapping"]):
            sample = {key: encoding[key][ci] for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoding}
            if training:
                targets = []
                for start, end in offsets:
                    if end <= start:
                        targets.append(-100); continue
                    label = 0
                    for a, b in spans:
                        if a <= start and end <= b:
                            label = 1 if start == a else 2
                            break
                    targets.append(label)
                sample["labels"] = targets
            samples.append(sample)
            provenance.append((pi, offsets))
    return samples, provenance


class Collator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer

    def __call__(self, rows):
        labels = [row.get("labels") for row in rows]
        batch = self.tokenizer.pad([{key: value for key, value in row.items() if key != "labels"} for row in rows],
                                   padding=True, return_tensors="pt")
        if labels[0] is not None:
            width = batch["input_ids"].shape[1]
            batch["labels"] = torch.tensor([row + [-100] * (width - len(row)) for row in labels])
        return batch


def train_model(train, linker, tokenizer, model, args, device):
    samples = []
    for doc in train:
        rows, _ = document_chunks(doc, tokenizer, linker, args.max_length, training=True)
        samples.extend(rows)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, collate_fn=Collator(tokenizer), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    use_fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    model.to(device)
    log = []
    for epoch in range(args.epochs):
        model.train(); loss_sum = 0.0
        start = time.monotonic()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_sum += float(loss.detach().cpu())
        row = {"epoch": epoch + 1, "loss": loss_sum / len(loader), "seconds": time.monotonic() - start}
        log.append(row); print(json.dumps(row), flush=True)
    return {"training_chunks": len(samples), "optimizer_steps": steps, "epochs": log}


def predict_entities(doc, tokenizer, model, linker, args, device):
    samples, provenance = document_chunks(doc, tokenizer, linker, args.max_length, training=False)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer), num_workers=0)
    # Overlap windows are combined before BIO decoding, avoiding boundary duplicates.
    accumulated = defaultdict(lambda: defaultdict(lambda: [np.zeros(3, dtype=np.float32), 0]))
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probs = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
            for i in range(len(probs)):
                pi, offsets = provenance[cursor]; cursor += 1
                for j, (start, end) in enumerate(offsets):
                    if end <= start: continue
                    entry = accumulated[pi][(start, end)]
                    entry[0] += probs[i, j]; entry[1] += 1
    entities = []
    for pi, token_probs in accumulated.items():
        passage = doc["full_text"][pi]; text = passage["text"]; base = int(passage.get("offset", 0))
        tokens = sorted(token_probs)
        probabilities = [token_probs[key][0] / token_probs[key][1] for key in tokens]
        labels = [int(prob.argmax()) for prob in probabilities]
        begin = None
        for i in range(len(tokens) + 1):
            label = labels[i] if i < len(tokens) else 0
            if begin is not None and label != 2:
                start, end = tokens[begin][0], tokens[i - 1][1]
                mention = text[start:end]
                confidence = float(np.mean([probabilities[k][labels[k]] for k in range(begin, i)]))
                if confidence >= args.ner_threshold and mention.strip() and any(ch.isalpha() for ch in mention):
                    identifier, _ = linker.link(mention)
                    entities.append({"identifier": identifier, "type": "Phenotype", "offset": base + start,
                                     "length": end - start, "text": mention, "note": None})
                begin = None
            if label == 1 or (label == 2 and begin is None): begin = i
    return sorted(entities, key=lambda e: (e["offset"], e["length"], e["identifier"]))


def main():
    parser = argparse.ArgumentParser()
    for key in ("train", "input", "hpo", "output", "work-dir", "model"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--ner-threshold", type=float, default=0.55)
    parser.add_argument("--relation-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20261023)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Assigned CUDA device unavailable")
    start = time.monotonic()
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    auxiliary = work / "auxiliary.pkl"
    train_hash, hpo_hash = sha(args.train), sha(args.hpo)
    if args.reuse_model:
        with auxiliary.open("rb") as handle: state = pickle.load(handle)
        assert state["train_sha256"] == train_hash and state["hpo_sha256"] == hpo_hash
        linker = Linker.__new__(Linker); linker.__dict__.update(state["linker_state"])
        vectorizer, relation, training_stats = state["heads"]
        tokenizer = AutoTokenizer.from_pretrained(work / "ner", use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(work / "ner").to(device)
    else:
        train = read_jsonl(args.train)
        linker = Linker(train, args.hpo)
        vectorizer, relation, relation_stats = fit_relation(train, linker)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=3,
                    id2label=LABELS, label2id={label: key for key, label in LABELS.items()})
        # A cached base model may contain a newly initialized classifier. Reset
        # only that task head with an explicit seed, leaving the encoder intact.
        torch.manual_seed(20261023)
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=model.config.initializer_range)
        torch.nn.init.zeros_(model.classifier.bias)
        training_stats = train_model(train, linker, tokenizer, model, args, device)
        training_stats.update(relation_stats)
        model.save_pretrained(work / "ner"); tokenizer.save_pretrained(work / "ner")
        with auxiliary.open("wb") as handle:
            pickle.dump({"linker_state": linker.__dict__, "heads": (vectorizer, relation, training_stats),
                         "train_sha256": train_hash, "hpo_sha256": hpo_hash}, handle, protocol=5)
    fitted = time.monotonic()
    inputs = read_jsonl(args.input)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    total_entities = total_links = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, doc in enumerate(inputs):
            entities = predict_entities(doc, tokenizer, model, linker, args, device)
            associations = {patient["patient_id"]: set() for patient in doc.get("patient", [])}
            rows, pairs = [], []
            for entity in entities:
                for patient in doc.get("patient", []):
                    rows.append(pair_features(doc, entity, patient)); pairs.append((entity, patient))
            if rows:
                probs = relation.predict_proba(vectorizer.transform(rows))[:, 1]
                for prob, (entity, patient) in zip(probs, pairs):
                    if prob >= args.relation_threshold:
                        associations[patient["patient_id"]].update(association_keys(entity))
            pred = {"pmc_id": doc["pmc_id"], "pmid": doc.get("pmid"), "entities": entities,
                    "association": [{"patient_id": p["patient_id"], "phenotype": sorted(associations[p["patient_id"]])}
                                    for p in doc.get("patient", [])]}
            total_entities += len(entities); total_links += sum(map(len, associations.values()))
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            if (index + 1) % 10 == 0: print(json.dumps({"predicted_documents": index + 1}), flush=True)
    manifest = {"method": "BiomedBERT-BIO+HPO-char-tfidf-linking+logistic-patient-relation",
                "seed": 20261023, "pretrained_model": args.model,
                "train": str(Path(args.train).resolve()), "train_sha256": train_hash,
                "input": str(Path(args.input).resolve()), "input_sha256": sha(args.input),
                "hpo_sha256": hpo_hash, "hpo_version": linker.version,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "output": str(output.resolve()), "output_sha256": sha(output), "output_bytes": output.stat().st_size,
                "documents": len(inputs), "entities": total_entities, "patient_phenotypes": total_links,
                "fit_seconds": fitted - start, "predict_seconds": time.monotonic() - fitted,
                "training_stats": training_stats,
                "config": {key: getattr(args, key) for key in ["epochs", "batch_size", "max_length", "learning_rate", "ner_threshold", "relation_threshold", "device"]},
                "source_sha256": sha(__file__), "common_source_sha256": sha(Path(__file__).with_name("run.py")),
                "runtime_versions": {**runtime_versions(), "torch": torch.__version__,
                                     "transformers": __import__("transformers").__version__},
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                "input_annotations_used": False}
    (work / (output.stem + "-manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
