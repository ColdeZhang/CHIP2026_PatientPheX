#!/usr/bin/env python3
"""Gold-aware day1 bottleneck diagnostics; never emit oracle predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluate import associations, evaluate, prf
from validate_predictions import file_sha256, load_jsonl

GOLD_SHA = "c1f0f24a58b62bb898eb672550dc58247ddb9a0e8768c6f899812cfc327cc050"
EVALUATOR_SHA = "ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781"
ROUTES = {
    "ontology-rules": "ontology-rules/dev-v1/predictions.jsonl",
    "case-retrieval": "case-retrieval/dev-cpu/predictions.jsonl",
    "sequence-crf": "sequence-crf/crf-dev-v1/pred.jsonl",
    "fuzzy-ranker": "fuzzy-ranker/dev-v1/predictions.jsonl",
    "semantic-neural": "semantic-neural/dev-sapbert/predictions.jsonl",
}


def entity_concepts(row):
    concepts = set()
    for entity in row["entities"]:
        if str(entity.get("note") or "").upper() == "NO":
            continue
        concepts.update(entity["text"] if identifier == "-1" else identifier
                        for identifier in entity["identifier"].split(";"))
    return concepts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, help="Day1 directory containing five route subdirectories")
    parser.add_argument("--gold", required=True, help="Original dev16.gold.jsonl, diagnostics only")
    parser.add_argument("--output", required=True, help="JSON report; no prediction artifact is emitted")
    args = parser.parse_args()
    base, gold_path = Path(args.runs_dir), Path(args.gold)
    evaluator_path = Path(__file__).parent / "evaluate.py"
    if file_sha256(gold_path) != GOLD_SHA:
        raise ValueError("Frozen dev16 gold SHA mismatch")
    if file_sha256(evaluator_path) != EVALUATOR_SHA:
        raise ValueError("Frozen v2 evaluator SHA mismatch")
    gold = load_jsonl(gold_path)
    truth = {row["pmc_id"]: row for row in gold}
    report = {
        "warning": "Diagnostic oracle metrics use validation gold and are NOT model scores. "
                   "They cannot qualify for the 0.8 gate, train any model, or be submitted.",
        "generator_sha256": file_sha256(__file__),
        "evaluator_sha256": EVALUATOR_SHA, "gold_sha256": GOLD_SHA, "routes": {},
    }
    candidate_maps = {}
    for route, relative_path in ROUTES.items():
        rows = load_jsonl(base / relative_path)
        baseline = evaluate(gold, rows)
        metrics = baseline["metrics"]
        counts, patient_f1, per_document = Counter(), [], []
        candidate_maps[route] = {}
        outside_count = 0
        for row in rows:
            concepts = entity_concepts(row)
            candidate_maps[route][row["pmc_id"]] = concepts
            outside_count += sum(len(values - concepts) for values in associations(row).values())
            doc_counts = Counter()
            gold_associations = associations(truth[row["pmc_id"]])
            for values in gold_associations.values():
                covered, missing = values & concepts, values - concepts
                counts.update(tp=len(covered), fn=len(missing))
                doc_counts.update(tp=len(covered), fn=len(missing))
                patient_f1.append(prf(len(covered), 0, len(missing))["f1"])
            per_document.append({
                "pmc_id": row["pmc_id"], "patients": len(gold_associations),
                "gold_relations": doc_counts["tp"] + doc_counts["fn"],
                "candidate_covered_gold_relations": doc_counts["tp"],
                "missing_from_entity_output": doc_counts["fn"],
            })
        oracle_micro = prf(counts["tp"], 0, counts["fn"])["f1"]
        oracle_macro = sum(patient_f1) / len(patient_f1)
        mention, document = baseline["counts"]["mention"], baseline["counts"]["document"]
        filtered_mention = prf(mention["tp"], 0, mention["fn"])["f1"]
        filtered_document = prf(document["tp"], 0, document["fn"])["f1"]
        report["routes"][route] = {
            "prediction_path": str(base / relative_path),
            "prediction_sha256": file_sha256(base / relative_path),
            "current_relations_outside_entity_output": outside_count,
            "baseline": metrics, "counts": baseline["counts"],
            "required_mean_relation_f1_if_entities_fixed": (3.2 - metrics["menF1"] - metrics["docF1"]) / 2,
            "required_mean_entity_f1_if_relations_fixed": (3.2 - metrics["micF1"] - metrics["macF1"]) / 2,
            "perfect_relation_score_with_current_entities": (metrics["menF1"] + metrics["docF1"] + 2) / 4,
            "perfect_entity_score_with_current_relations": (2 + metrics["micF1"] + metrics["macF1"]) / 4,
            "entity_candidate_coverage_relation_oracle": {
                "tp": counts["tp"], "fn": counts["fn"], "fp": 0,
                "micF1": oracle_micro, "macF1": oracle_macro,
                "Score_with_current_entities": (metrics["menF1"] + metrics["docF1"] + oracle_micro + oracle_macro) / 4,
            },
            "perfect_entity_fp_filter_score_with_current_relations": (
                filtered_mention + filtered_document + metrics["micF1"] + metrics["macF1"]) / 4,
            "per_document_candidate_coverage": per_document,
        }
    union = Counter()
    for document_id, row in truth.items():
        concepts = set.union(*(candidate_maps[route][document_id]
                               for route in ("ontology-rules", "case-retrieval", "sequence-crf")))
        for values in associations(row).values():
            union.update(tp=len(values & concepts), fn=len(values - concepts))
    report["three_route_union_relation_candidate_coverage"] = dict(union)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "baseline_Scores": {
        route: item["baseline"]["Score"] for route, item in report["routes"].items()}}))


if __name__ == "__main__":
    main()
