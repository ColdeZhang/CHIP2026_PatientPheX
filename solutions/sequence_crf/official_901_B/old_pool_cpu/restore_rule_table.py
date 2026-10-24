#!/usr/bin/env python3
"""Restore the unpersisted frozen Gazetteer counts; never fit a learned model.

Parent/root explicitly authorized deterministic original-train64 count recovery
after the existing inventory established that this table was not serialized.
The original source constructor is executed unchanged, then state is saved.
Dev usage is prediction-cache equality only, with no gold or quality evaluation.
"""
import argparse
import json
from pathlib import Path
import pickle
import time

from load_predict import HPO_SHA, ONTOLOGY_SHA, TRAIN_SHA, load_module, read_jsonl, ref, sha, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    for name in ["train", "hpo", "ontology-source", "dev-input", "dev-reference", "b-input", "output-dir"]:
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert sha(args.train) == TRAIN_SHA and sha(args.hpo) == HPO_SHA
    assert sha(args.dev_input) == "c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e"
    assert sha(args.dev_reference) == "5a994261965f7887f2bfef384401380d49581940435bbe46d19604f6e4868e4b"
    assert sha(args.b_input) == "878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "manifest.json").exists()
    ontology = load_module("frozen_rules_table", args.ontology_source, ONTOLOGY_SHA)
    train, dev, bdocs = [read_jsonl(path) for path in [args.train, args.dev_input, args.b_input]]
    assert len(train) == 64 and len(dev) == 16 and len(bdocs) == 100
    assert not {d["pmc_id"] for d in train} & {d["pmc_id"] for d in dev + bdocs}
    assert all("entities" not in d and "association" not in d for d in dev + bdocs)
    model = ontology.Gazetteer(args.hpo, train)
    with (out / "gazetteer-state.pkl").open("wb") as handle:
        pickle.dump({"ontology_source_sha256": ONTOLOGY_SHA, "train_sha256": TRAIN_SHA,
                     "hpo_sha256": HPO_SHA, "state": model.__dict__}, handle, protocol=5)
    # All inference below uses the reopened saved state.
    loaded = ontology.Gazetteer.__new__(ontology.Gazetteer)
    loaded.__dict__.update(pickle.loads((out / "gazetteer-state.pkl").read_bytes())["state"])
    replay = [ontology.predict(doc, loaded) for doc in dev]
    write_jsonl(out / "dev.base.jsonl", replay)
    checks = {"dev_json_exact": replay == read_jsonl(args.dev_reference),
              "dev_bytes_exact": sha(out / "dev.base.jsonl") == sha(args.dev_reference)}
    assert all(checks.values()), checks
    prediction = [ontology.predict(doc, loaded) for doc in bdocs]
    write_jsonl(out / "B.base.jsonl", prediction)
    manifest = {"config": vars(args), "source": ref(__file__), "ontology_source": ref(args.ontology_source),
        "inputs": {k: ref(getattr(args, k)) for k in ["train", "hpo", "dev_input", "dev_reference", "b_input"]},
        "authorization": "parent/root explicitly allowed exact original train64 Gazetteer count restoration; no learned estimator fitting",
        "initial_gap": "Full Gazetteer was not serialized; existing deployment inventory records actual_full_object_not_serialized=true",
        "outputs": {p.name: ref(p) for p in out.iterdir() if p.is_file()},
        "checks": checks, "stats": model.stats, "train_documents_read": 64,
        "dev_gold_read": False, "learned_model_fit": False, "quality_evaluation": False,
        "b_documents": len(bdocs), "b_patients": sum(len(d.get("patient", [])) for d in bdocs),
        "b_entities": sum(len(d["entities"]) for d in prediction), "seconds": time.monotonic()-started}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"checks": checks, "stats": model.stats, "outputs": manifest["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
