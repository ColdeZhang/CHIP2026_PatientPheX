#!/usr/bin/env python3
"""Reproduce two archived-model defects without training or loading a model.

Dependency: pyahocorasick. Python standard library suffices for the NER probe.
Run from any directory:
  python patientphex_model_diagnostics.py --repo /path/to/CHIP2026_PatientPheX \
      --out /tmp/patientphex-model-diagnostics.json

The script reads repository sources/data, executes only the relation builder's
in-memory construction portion, and extracts the NER decoding function by AST.
It never imports the model scripts, writes repository files, or loads torch,
transformers, numpy, model weights, or network resources.
"""

import argparse
import ast
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Source no longer matches audited version: {old!r}")
    return text.replace(old, new, 1)


def relation_diagnostic(root):
    source_path = root / "src/v3_build_rel_data.py"
    source = source_path.read_text(encoding="utf-8")
    marker = "# Split train/dev"
    if source.count(marker) != 1:
        raise ValueError("Cannot locate relation builder split boundary")
    # Exclude every output write, and redirect only the two input paths.
    prefix = source.split(marker, 1)[0]
    prefix = replace_once(prefix, 'os.chdir("/dataset/wanggenyi/patient_phex")', "")
    prefix = replace_once(
        prefix, '"data/hpo_dict_augmented.json"',
        repr(str(root / "dict/hpo_dict_augmented.json")),
    )
    prefix = replace_once(
        prefix, '"data/train.jsonl"',
        repr(str(root / "data/train/PatientPheX-train.jsonl")),
    )
    prefix = replace_once(
        prefix,
        'samples.append({"text": text_inp, "label": label})',
        'samples.append({"text": text_inp, "label": label, '
        '"doc": d["pmc_id"], '
        '"candidate_matches_assoc": bool(hpo_ids & gold_assoc.get(pid, set()))})',
    )
    namespace = {}
    exec(compile(prefix, str(source_path), "exec"), namespace)
    samples = namespace["samples"]
    random = namespace["random"]  # Builder already seeded it with 42.
    # Mirror the archived split and balancing exactly, retaining metadata only
    # to diagnose overlap. No output files from the original builder are used.
    random.shuffle(samples)
    n_dev = int(len(samples) * 0.1)
    dev = samples[:n_dev]
    train = samples[n_dev:]
    positives = [row for row in train if row["label"] == 1]
    negatives = [row for row in train if row["label"] == 0]
    random.shuffle(negatives)
    train = positives + negatives[:len(positives) * 2]
    random.shuffle(train)
    train_texts = {row["text"] for row in train}
    train_docs = {row["doc"] for row in train}
    dev_docs = {row["doc"] for row in dev}
    text_labels = defaultdict(set)
    for row in samples:
        text_labels[row["text"]].add(row["label"])
    ambiguous = {text for text, labels in text_labels.items() if len(labels) > 1}
    return {
        "samples_total": len(samples),
        "train_after_balance": len(train),
        "dev_samples": len(dev),
        "train_docs": len(train_docs),
        "dev_docs": len(dev_docs),
        "overlap_docs": len(train_docs & dev_docs),
        "dev_identical_text_in_train": sum(row["text"] in train_texts for row in dev),
        "all_unique_feature_texts": len(text_labels),
        "ambiguous_feature_texts": len(ambiguous),
        "samples_with_ambiguous_text": sum(row["text"] in ambiguous for row in samples),
        "positive_samples": sum(row["label"] for row in samples),
        "positive_but_candidate_hpo_not_associated": sum(
            row["label"] and not row["candidate_matches_assoc"] for row in samples
        ),
    }


class FakeTensor:
    """Minimal interface used by the archived decoder; no tensor library."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def argmax(self, axis):
        # Input already holds the chosen token classes.
        return self


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        if text != "with fever":
            raise ValueError("Unexpected text in fixed NER regression probe")
        return FakeEncoding(
            input_ids=FakeTensor([[1, 2, 3, 4]]),
            attention_mask=FakeTensor([[1, 1, 1, 1]]),
            offset_mapping=FakeTensor([[(0, 0), (0, 4), (5, 10), (0, 0)]]),
        )


class FakeModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=FakeTensor([[0, 0, 1, 0]]))


def ner_diagnostic(root):
    source_path = root / "src/v5_predict_ner_union.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "predict_spans"
    )
    namespace = {
        "torch": SimpleNamespace(no_grad=contextlib.nullcontext),
        "LABEL2ID": {"O": 0, "B-PHEN": 1, "I-PHEN": 2},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    actual = namespace["predict_spans"]("with fever", FakeModel(), FakeTokenizer())
    return {
        "text": "with fever",
        "tokens": ["[CLS]", "with", "fever", "[SEP]"],
        "offsets": [[0, 0], [0, 4], [5, 10], [0, 0]],
        "predicted_labels": ["O", "O", "B-PHEN", "O"],
        "expected_spans": [[5, 10]],
        "actual_spans": [list(span) for span in actual],
        "matches_expected": actual == [(5, 10)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="Optional JSON report path outside repository")
    args = parser.parse_args()
    root = args.repo.resolve()
    inputs = [
        "src/v3_build_rel_data.py", "src/v5_predict_ner_union.py",
        "dict/hpo_dict_augmented.json", "data/train/PatientPheX-train.jsonl",
    ]
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(root),
        "method": "In-memory relation reconstruction with archived seed/split; AST-extracted NER decoder with fixed stubs. No model training.",
        "input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in inputs
        },
        "relation": relation_diagnostic(root),
        "ner_tail_decode": ner_diagnostic(root),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        target = args.out.resolve()
        if target == root or root in target.parents:
            parser.error("--out must be outside the repository (this is a read-only audit)")
        target.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
