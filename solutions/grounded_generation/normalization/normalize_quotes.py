"""Map exact globally addressed quotes to HPO/NIL/abstain with fixed public SapBERT."""
import argparse
import json
from pathlib import Path

try:
    from .strict import read_jsonl, verify_span, sha256
    from .semantic import GroundedNormalizer, DEFAULT_POLICY
except ImportError:
    from strict import read_jsonl, verify_span, sha256
    from semantic import GroundedNormalizer, DEFAULT_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["hpo", "train", "model-dir", "bank-dir", "input", "quotes", "output-dir"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    documents = read_jsonl(args.input)
    if any("entities" in doc or "association" in doc for doc in documents):
        raise ValueError("Input documents must be annotation-free")
    by_id = {str(doc["pmc_id"]): doc for doc in documents}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate input document")
    quotes = read_jsonl(args.quotes)
    passages = [verify_span(by_id[str(row["pmc_id"])], row["quote"], row["offset"]) for row in quotes]
    normalizer = GroundedNormalizer(args.hpo, args.train, model_dir=args.model_dir, bank_dir=args.bank_dir,
                                   policy_path=args.policy, device=args.device)
    normalizer.strict.assert_disjoint_input(args.input)
    results = normalizer.normalize_many([row["quote"] for row in quotes])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "normalizations.jsonl").open("x") as handle:
        for row, result, passage in zip(quotes, results, passages):
            handle.write(json.dumps({**row, **result, "length": len(row["quote"]), "passage_index": passage}, ensure_ascii=False) + "\n")
    (out / "semantic-evidence.json").write_text(json.dumps(normalizer.cache, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({"policy_sha256": sha256(args.policy),
        "input_sha256": sha256(args.input), "quotes_sha256": sha256(args.quotes), "records": len(results),
        "normalizations_sha256": sha256(out / "normalizations.jsonl"),
        "semantic_evidence_sha256": sha256(out / "semantic-evidence.json"),
        "encoder_loaded": normalizer.encoder is not None,
        "encoder": normalizer.encoder.manifest if normalizer.encoder else None,
        "no_input_annotations": True, "no_fit": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
