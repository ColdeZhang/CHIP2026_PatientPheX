"""Authorized one-shot public bank slicing and train-only SapBERT query encoding."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from strict import HPO, sha256
from public_bank import audit_old_labels, public_rows
from semantic import PublicConceptBank, PublicSapEncoder, load_policy


def put(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    start = time.monotonic()
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    put(out / "started.json", {"pid": os.getpid(), "plan_sha256": sha256(args.plan), "no_fit": True})
    try:
        for path, expected in plan["files"].items():
            if sha256(path) != expected["sha256"] or Path(path).stat().st_size != expected["bytes"]:
                raise ValueError(f"Frozen input/source differs: {path}")
        policy = load_policy(plan["policy"])
        queries = json.loads(Path(plan["queries"]).read_text())
        if not 0 < len(queries) <= 1200 or len({row["quote"] for row in queries}) != len(queries):
            raise ValueError("Expected unique complete train-only queries within budget")
        hpo = HPO(plan["hpo"])
        rows = public_rows(hpo)
        indexes, manifest = audit_old_labels(hpo, rows, plan["old_labels"], plan["reconstruction"], plan["old_semantic_source"])
        if sha256(plan["old_bank"]) != manifest["historical_bank"]["sha256"]:
            raise ValueError("Old bank absent or bytes changed; re-encoding forbidden")
        old = np.load(plan["old_bank"], mmap_mode="r", allow_pickle=False)
        if old.shape != (42575, 768) or old.dtype != np.float32:
            raise ValueError("Historical bank shape/dtype differs")
        bank = old[indexes]
        np.save(out / "public-bank.npy", bank)
        put(out / "public-labels.json", rows)
        manifest.update(bank_bytes_verified_here=True, public_bank_sha256=sha256(out / "public-bank.npy"),
            public_labels_sha256=sha256(out / "public-labels.json"), audit="verified_public_subset_no_encoding")
        put(out / "public-bank-manifest.json", manifest)
        public = PublicConceptBank(hpo, out / "public-bank.npy", out / "public-labels.json", out / "public-bank-manifest.json")
        if len(public.rows) != policy["public_bank_rows"]:
            raise ValueError("Policy public row count mismatch")
        encoder = PublicSapEncoder(plan["model_dir"], "cuda")
        put(out / "encoder-runtime.json", encoder.manifest)
        quotes = [row["quote"] for row in queries]
        vectors, counts = encoder.encode(quotes)
        np.save(out / "query-vectors.npy", vectors)
        # Persist actual embeddings before ranking or policy decisions.
        put(out / "query-token-counts.json", {"quotes": quotes, "token_counts": counts,
            "query_vectors_sha256": sha256(out / "query-vectors.npy")})
        results = [{**row, "token_count": count, "candidates": public.rank(vector) if count <= 32 else []}
                   for row, vector, count in zip(queries, vectors, counts)]
        put(out / "query-scores.json", results)
        assets = {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in [
            "public-bank.npy", "public-labels.json", "public-bank-manifest.json", "query-vectors.npy",
            "query-token-counts.json", "query-scores.json", "encoder-runtime.json"]}
        put(out / "complete.json", {"status": "COMPLETE_FIXED_PUBLIC_TRAIN_QUERIES", "query_count": len(queries),
            "public_bank_rows": len(rows), "query_overflows_abstain": sum(count > 32 for count in counts),
            "policy_sha256": sha256(plan["policy"]), "plan_sha256": sha256(args.plan),
            "no_training_head_or_statistics_loaded": True, "no_fit": True, "no_held_queries": True,
            "no_bank_reencoding": True, "seconds": time.monotonic() - start, "assets": assets})
    except BaseException:
        put(out / "failure.json", {"traceback": traceback.format_exc(), "seconds": time.monotonic() - start,
                                  "no_automatic_retry": True})
        raise


if __name__ == "__main__":
    main()
