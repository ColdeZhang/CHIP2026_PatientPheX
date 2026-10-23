#!/usr/bin/env python3
"""Rebuild four document-disjoint folds inside the frozen train64.

Patient count and text length are the only stratification features. Phenotype
labels, the original dev16 and official test data cannot influence the split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

TRAIN_SHA = "cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2"
DEV_INPUT_SHA = "c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e"
SEED = 2026102302


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def text_length(row):
    return sum(len(passage["text"]) for passage in row["full_text"])


def patient_group(row):
    count = len(row["patient"])
    return "1" if count == 1 else "2-3" if count <= 3 else "4+"


def stripped(row):
    return {key: value for key, value in row.items() if key not in ("entities", "association")}


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"path": str(path), "documents": len(rows), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train64", required=True)
    parser.add_argument("--dev-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if sha(args.train64) != TRAIN_SHA:
        raise ValueError("train64 SHA mismatch: original frozen training data required")
    if sha(args.dev_input) != DEV_INPUT_SHA:
        raise ValueError("dev16.input SHA mismatch: original frozen validation required")
    train, dev = read(args.train64), read(args.dev_input)
    ids = {row["pmc_id"] for row in train}
    dev_ids = {row["pmc_id"] for row in dev}
    if len(train) != 64 or len(ids) != 64 or len(dev_ids) != 16 or ids & dev_ids:
        raise ValueError("Unexpected document count, duplicate IDs or train/dev overlap")

    buckets = defaultdict(list)
    for row in train:
        buckets[patient_group(row)].append(row)
    strata = {}
    for bucket, rows in sorted(buckets.items()):
        ordered = sorted(rows, key=lambda row: (text_length(row), row["pmc_id"]))
        cut = (len(ordered) + 1) // 2
        strata[bucket + "-short"], strata[bucket + "-long"] = ordered[:cut], ordered[cut:]
    rng = random.Random(SEED)
    folds = [[] for _ in range(4)]
    stratum_counts = [Counter() for _ in range(4)]
    patient_counts, characters = [0] * 4, [0] * 4
    for label, rows in sorted(strata.items(), key=lambda item: (-len(item[1]), item[0])):
        rows = rows[:]
        rng.shuffle(rows)
        for row in rows:
            tie = list(range(4))
            rng.shuffle(tie)
            rank = {fold: order for order, fold in enumerate(tie)}
            options = [fold for fold in range(4) if len(folds[fold]) < 16]
            fold = min(options, key=lambda k: (
                stratum_counts[k][label], len(folds[k]), patient_counts[k], characters[k], rank[k]))
            folds[fold].append(row)
            stratum_counts[fold][label] += 1
            patient_counts[fold] += len(row["patient"])
            characters[fold] += text_length(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": "round2-train64-inner4", "seed": SEED, "unit": "pmc_id",
        "method": "patient-count strata (1,2-3,4+) crossed with within-stratum median text length; "
                  "fixed seeded shuffle and balanced assignment to four 16-document folds; "
                  "all stratification features are input-only",
        "generator_sha256": sha(__file__),
        "train64_sha256": TRAIN_SHA, "original_dev16_input_sha256": DEV_INPUT_SHA,
        "excluded_original_dev16_ids": sorted(dev_ids), "folds": [],
        "selection_note": "Inner folds support tuning and robustness checks. Original dev16 all/text v2 "
                          "remains the user-requested >=0.8 gate; no official-score claim.",
    }
    seen = set()
    for fold, rows in enumerate(folds):
        validation = sorted(rows, key=lambda row: row["pmc_id"])
        validation_ids = {row["pmc_id"] for row in validation}
        training = sorted([row for row in train if row["pmc_id"] not in validation_ids],
                          key=lambda row: row["pmc_id"])
        if len(validation) != 16 or len(training) != 48 or seen & validation_ids or validation_ids & dev_ids:
            raise AssertionError("Split invariant failed")
        seen |= validation_ids
        dest = out / f"fold-{fold}"
        dest.mkdir(parents=True, exist_ok=True)
        manifest["folds"].append({
            "fold": fold,
            "training_ids": [row["pmc_id"] for row in training],
            "validation_ids": [row["pmc_id"] for row in validation],
            "training_documents": 48, "validation_documents": 16,
            "training_patients": sum(len(row["patient"]) for row in training),
            "validation_patients": patient_counts[fold],
            "validation_input_strata": dict(sorted(stratum_counts[fold].items())),
            "files": [write(dest / "train48.jsonl", training),
                      write(dest / "validation16.input.jsonl", [stripped(row) for row in validation]),
                      write(dest / "validation16.gold.jsonl", validation)],
        })
    if seen != ids:
        raise AssertionError("Inner validation folds do not partition train64")
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "validation_patients": patient_counts}))


if __name__ == "__main__":
    main()
