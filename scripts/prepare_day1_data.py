#!/usr/bin/env python3
"""Freeze one document-disjoint split from freshly downloaded official data."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({row["pmc_id"] for row in rows}) == len(rows), "Duplicate pmc_id"
    return rows


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"path": path.name, "documents": len(rows), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inputs(rows):
    return [{key: value for key, value in row.items() if key not in {"entities", "association"}} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--dev-a", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    train, test, dev_a = read(args.train), read(args.test), read(args.dev_a)
    assert (len(train), len(test), len(dev_a)) == (80, 100, 20), "Unexpected official dataset sizes"
    rows = sorted(train, key=lambda row: row["pmc_id"])
    random.Random(args.seed).shuffle(rows)
    validation, training = rows[:16], rows[16:]
    assert not ({row["pmc_id"] for row in training} & {row["pmc_id"] for row in validation})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = {"train64.jsonl": training, "dev16.gold.jsonl": validation,
                "dev16.input.jsonl": inputs(validation), "train80.jsonl": train,
                "B.input.jsonl": inputs(test), "A.input.jsonl": inputs(dev_a)}
    manifest = {"seed": args.seed, "method": "sort pmc_id, Python random.Random(seed).shuffle, first 16 held out",
                "unit": "pmc_id", "training_documents": [row["pmc_id"] for row in training],
                "validation_documents": [row["pmc_id"] for row in validation],
                "validation_role": "local holdout; not official test score",
                "leakage_rule": "derive resources only from train64 for validation; train80 only for final B refit",
                "sources": [{"name": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                            for path in (args.train, args.test, args.dev_a)],
                "files": [write(out / name, content) for name, content in variants.items()],
                "patients": {"train": sum(len(row["patient"]) for row in training),
                             "validation": sum(len(row["patient"]) for row in validation)}}
    dest = Path(args.manifest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(training), "validation": len(validation), "patients": manifest["patients"]}))


if __name__ == "__main__":
    main()
