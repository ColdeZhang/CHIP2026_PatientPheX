"""Prepare public-only SapBERT rows, or prove and slice a historical text bank.

No command fits a head or reads supervised examples. The default audit only reads
metadata; passing --bank explicitly permits local array verification and slicing.
Encoding is a separate explicit command and was not run for initial preparation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

try:
    from .strict import HPO, sha256
except ImportError:
    from strict import HPO, sha256

PUBLIC_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
MODEL_FILES = {
    "config.json": "c0fa35def52bf7d81865d23dba0417fdbb67d416fdde70cae5c7beab476b584e",
    "tokenizer_config.json": "76eb84b4b5b0d1bdba02906d7fae249be541c41d72e2024094aea513f98f2be7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "vocab.txt": "79489a52be45e6fa033521e8ce8e4f62aedc0a742ee2aa6fc04667e5b0b1454d",
    "model.safetensors": "dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814",
    "manifest.json": "0f4f8a8c16d0d7f978ab544c80d117b930666de2793276797991175ea52a81d8",
}
SEMANTIC_SOURCE_SHA256 = "b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a"
OLD_LABELS_SHA256 = "a369ec3de1f32764272748f30d0de4244b743fbd28f527adea767bd5fa0b2cd2"
OLD_RECONSTRUCTION_SHA256 = "469e5299cdc9593a53efae907741b4df9bd3ee7e6b15ed2eebfd48062a775a5f"


def legacy_text(text):
    """Exactly the frozen encoder's text preprocessing, not learned normalization."""
    return " ".join(text.lower().split()).strip()


def public_rows(hpo):
    by_text = defaultdict(lambda: {"ids": set(), "provenance": []})
    for alias in hpo.aliases():
        if len(alias["text"]) < 3:
            continue  # Match the public portion of the frozen historical bank.
        key = legacy_text(alias["text"])
        by_text[key]["ids"].update(alias["ids"])
        by_text[key]["provenance"].append({"surface": alias["text"], "source": alias["source"],
                                           "id": alias["ids"][0]})
    return [{"text": text, "identifiers": sorted(item["ids"]),
             "provenance": sorted(item["provenance"], key=lambda row: (row["id"], row["surface"], row["source"]))}
            for text, item in sorted(by_text.items())]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def metadata(hpo, rows):
    return {"hpo_sha256": hpo.sha256, "hpo_version": hpo.version,
            "strict_source_sha256": sha256(Path(__file__).with_name("strict.py")),
            "bank_utility_source_sha256": sha256(__file__),
            "public_rows": len(rows), "rows_content_sha256": digest_json(rows),
            "ambiguous_rows": sum(len(row["identifiers"]) > 1 for row in rows),
            "text_preprocessing": "lower; whitespace split/join; original surface length >= 3",
            "included_sources": ["hpo_name", "hpo_exact_synonym"], "root_excluded": True,
            "training_aliases_used": False, "training_statistics_used": False,
            "old_label_ids_used_for_mapping": False, "fitted_head_used": False,
            "id_contract": "active HP:0000118 descendants; canonical IDs; all public ambiguity retained; no compound/NIL labels",
            "public_model": PUBLIC_MODEL, "revision": REVISION,
            "model_files_sha256": MODEL_FILES, "pooling": "CLS_L2", "max_length": 32}


def audit_old_labels(hpo, rows, labels_path, reconstruction_path, semantic_source):
    if sha256(labels_path) != OLD_LABELS_SHA256:
        raise ValueError("Unexpected frozen ordered label list")
    if sha256(reconstruction_path) != OLD_RECONSTRUCTION_SHA256:
        raise ValueError("Unexpected reconstruction manifest")
    if sha256(semantic_source) != SEMANTIC_SOURCE_SHA256:
        raise ValueError("Unexpected encoder source")
    old = json.loads(Path(labels_path).read_text())
    frozen = json.loads(Path(reconstruction_path).read_text())
    labels = old["labels"]
    if not all(isinstance(text, str) for text in labels) or labels != sorted(set(labels)):
        raise ValueError("Historical row order must be sorted unique strings")
    bank = frozen["bank_cache"]
    ordered_hash = hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest()
    if ordered_hash != bank["ordered_texts_sha256"] or len(labels) != bank["rows"]:
        raise ValueError("Historical labels are not bound to bank rows")
    signature = "sapbert" + MODEL_FILES["config.json"] + MODEL_FILES["manifest.json"]
    cache_key = hashlib.sha256((signature + "\0" + "\0".join(labels)).encode()).hexdigest()
    if signature != frozen["encoder_signature"] or Path(bank["path"]).name != cache_key + ".npy":
        raise ValueError("Historical cache signature differs")
    if frozen["semantic_source_sha256"] != SEMANTIC_SOURCE_SHA256 or frozen["inputs"]["hpo"] != hpo.sha256:
        raise ValueError("Historical HPO/encoder differs")
    for name, expected in MODEL_FILES.items():
        if name != "manifest.json" and frozen["actual_model_files_sha256"][name] != expected:
            raise ValueError(f"Historical encoder weights/config mismatch: {name}")
    positions = {text: index for index, text in enumerate(labels)}
    missing = [row["text"] for row in rows if row["text"] not in positions]
    if missing:
        raise ValueError(f"Historical bank missing {len(missing)} public strings")
    indexes = [positions[row["text"]] for row in rows]
    overwritten = [{"text": row["text"], "old_identifier": old["label_ids"][positions[row["text"]]],
                    "public_identifiers": row["identifiers"]}
                   for row in rows if old["label_ids"][positions[row["text"]]] not in row["identifiers"]]
    report = {**metadata(hpo, rows), "audit": "local_metadata_only",
        "historical_labels": {"path": str(Path(labels_path).resolve()), "sha256": sha256(labels_path),
                              "rows": len(labels), "ordered_texts_sha256": ordered_hash},
        "historical_reconstruction": {"path": str(Path(reconstruction_path).resolve()), "sha256": sha256(reconstruction_path)},
        "historical_encoder_source": {"path": str(Path(semantic_source).resolve()), "sha256": sha256(semantic_source)},
        "historical_bank": bank, "public_rows_matched": len(indexes), "nonpublic_rows_excluded": len(labels)-len(indexes),
        "selection_indices_sha256": digest_json(indexes),
        "old_mapping_conflicts_discarded": overwritten,
        "vector_provenance": "Frozen source Encoder.encode uses only supplied text and public AutoModel.eval; System.fit_detector is downstream of bank construction and does not alter encoder weights.",
        "bank_bytes_verified_here": False, "new_encoding_performed": False}
    return indexes, report


def verify_model(model_dir):
    actual = {name: sha256(Path(model_dir) / name) for name in MODEL_FILES}
    if actual != MODEL_FILES:
        raise ValueError("Public SapBERT files differ from frozen FP16 assets")
    return actual


def encode_public(rows, model_dir, output_path, device, batch_size):
    """Explicit optional encoding, independent of every old fitted detector."""
    verify_model(model_dir)
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    if device.startswith("cuda"):
        model.half()
    output = []
    for start in range(0, len(rows), batch_size):
        texts = [row["text"] for row in rows[start:start + batch_size]]
        batch = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        with torch.inference_mode():
            values = model(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state[:, 0].float()
        values = values.cpu().numpy().astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
        output.append(values)
    vectors = np.concatenate(output)
    if not np.isfinite(vectors).all() or vectors.shape != (len(rows), 768):
        raise ValueError("Invalid public embeddings")
    np.save(output_path, vectors)
    return {"torch": torch.__version__, "device": device, "parameters": sum(p.numel() for p in model.parameters()),
            "new_encoding_performed": True, "bank_sha256": sha256(output_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit", "encode"])
    parser.add_argument("--hpo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-labels")
    parser.add_argument("--reconstruction")
    parser.add_argument("--semantic-source")
    parser.add_argument("--bank")
    parser.add_argument("--model-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    hpo = HPO(args.hpo)
    rows = public_rows(hpo)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = metadata(hpo, rows)
    if args.command == "audit":
        if not all([args.old_labels, args.reconstruction, args.semantic_source]):
            parser.error("audit requires --old-labels --reconstruction --semantic-source")
        indexes, report = audit_old_labels(hpo, rows, args.old_labels, args.reconstruction, args.semantic_source)
        if args.bank:
            import numpy as np
            if sha256(args.bank) != report["historical_bank"]["sha256"]:
                raise ValueError("Historical bank bytes differ")
            old_bank = np.load(args.bank, allow_pickle=False, mmap_mode="r")
            if old_bank.shape != (report["historical_labels"]["rows"], 768) or old_bank.dtype != np.float32:
                raise ValueError("Historical bank shape/dtype differs")
            bank = old_bank[indexes]
            if not np.isfinite(bank).all() or not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-5):
                raise ValueError("Historical selected embeddings invalid")
            np.save(output / "public-bank.npy", bank)
            report.update(bank_bytes_verified_here=True, audit="verified_local_array_public_subset",
                          public_bank_sha256=sha256(output / "public-bank.npy"))
    else:
        if args.command == "encode":
            if not args.model_dir:
                parser.error("encode requires --model-dir")
            report.update(encode_public(rows, args.model_dir, output / "public-bank.npy", args.device, args.batch_size))
        else:
            report["new_encoding_performed"] = False
    # The metadata-only audit stays small; full provenance rows accompany prepared/sliced banks.
    if args.command != "audit" or args.bank:
        labels = output / "public-labels.json"
        labels.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
        report["public_labels_sha256"] = sha256(labels)
    (output / "public-bank-manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ["public_rows", "ambiguous_rows", "audit", "new_encoding_performed"]}))


if __name__ == "__main__":
    main()
