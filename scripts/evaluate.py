#!/usr/bin/env python3
"""Approximate public PatientPheX metrics; explicitly not the official evaluator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from validate_predictions import file_sha256, index_documents, load_jsonl, passage_for_span, validate_predictions

VERSION = "patientphex-public-approx-v2"
EXCLUDED_SECTIONS = {"TITLE", "ABSTRACT", "REF", "REFERENCES", "AUTH_CONT", "COMP_INT", "ACK_FUND", "SUPPL"}
COMPOUND_HPO = re.compile(r"HP:[0-9]{7}(?:;HP:[0-9]{7})+")


def prf(tp: int, fp: int, fn: int, empty_value: float = 1.0) -> dict[str, float | int]:
    if tp + fp + fn == 0:
        precision = recall = f1 = empty_value
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compare_sets(gold: set, predicted: set) -> tuple[int, int, int]:
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def included(entity: dict[str, Any], document: dict[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    passage = passage_for_span(document, entity["offset"], entity["length"])
    if passage is None:
        return False
    section = str(passage.get("section_type", "")).upper()
    passage_type = str(passage.get("type", "")).lower()
    return section not in EXCLUDED_SECTIONS and passage_type != "front" and not passage_type.startswith("abstract")


def positive_units(document: dict[str, Any], source: dict[str, Any], scope: str) -> dict[tuple[int, int], set[str]]:
    result: dict[tuple[int, int], set[str]] = defaultdict(set)
    for entity in document["entities"]:
        if str(entity.get("note") or "").upper() == "NO" or not included(entity, source, scope):
            continue
        result[(entity["offset"], entity["length"])].update(entity["identifier"].split(";"))
    return result


def mention_counts(gold: dict, predicted: dict, wildcard_no_id: bool = True) -> tuple[int, int, int]:
    """Exact HPO matches first; one unlinked gold mention can consume one remaining span match."""
    tp = fp = fn = 0
    for span in set(gold) | set(predicted):
        truth, guess = set(gold.get(span, set())), set(predicted.get(span, set()))
        matched = truth & guess
        tp += len(matched)
        truth -= matched
        guess -= matched
        if wildcard_no_id and "-1" in truth and guess:
            truth.remove("-1")
            guess.remove(sorted(guess)[0])
            tp += 1
        fp += len(guess)
        fn += len(truth)
    return tp, fp, fn


def document_concepts(document: dict[str, Any], source: dict[str, Any], scope: str, no_id_mode: str) -> set:
    result = set()
    for entity in document["entities"]:
        if str(entity.get("note") or "").upper() == "NO" or not included(entity, source, scope):
            continue
        for identifier in entity["identifier"].split(";"):
            if identifier != "-1":
                result.add(("hpo", identifier))
            elif no_id_mode == "text":
                result.add(("unlinked_text", entity["text"]))
            elif no_id_mode == "shared_id":
                result.add(("unlinked_id", "-1"))
    return result


def associations(document: dict[str, Any]) -> dict[str, set[str]]:
    result = {}
    for item in document["association"]:
        concepts = set()
        for value in item["phenotype"]:
            # Official training annotations also contain compound relation IDs.
            # Preserve unlinked surface text (which may contain semicolons),
            # non-HPO identifiers and malformed/mixed strings verbatim.
            concepts.update(value.split(";") if COMPOUND_HPO.fullmatch(value) else [value])
        result[item["patient_id"]] = concepts
    return result


def evaluate(gold_rows: list[dict[str, Any]], predicted_rows: list[dict[str, Any]],
             body_scope: str = "all", no_id_doc: str = "text", wildcard_no_id: bool = True,
             validate: bool = True) -> dict[str, Any]:
    gold, predictions = index_documents(gold_rows), index_documents(predicted_rows)
    if set(gold) != set(predictions):
        raise ValueError("Gold and prediction pmc_id sets must match exactly; missing documents are not silently ignored")
    if validate:
        report = validate_predictions(gold_rows, predicted_rows)
        if not report["valid"]:
            raise ValueError("Invalid prediction: " + "; ".join(report["errors"][:5]))
    totals = {name: Counter({"tp": 0, "fp": 0, "fn": 0}) for name in ("mention", "document", "relation")}
    patient_f1: list[float] = []
    per_document = []
    for pmc_id in sorted(gold):
        truth, guess = gold[pmc_id], predictions[pmc_id]
        m = mention_counts(positive_units(truth, truth, body_scope), positive_units(guess, truth, body_scope), wildcard_no_id)
        d = compare_sets(document_concepts(truth, truth, body_scope, no_id_doc),
                         document_concepts(guess, truth, body_scope, no_id_doc))
        g_assoc, p_assoc = associations(truth), associations(guess)
        expected_patients = {item["patient_id"] for item in truth["patient"]}
        if set(g_assoc) != expected_patients:
            raise ValueError(f"{pmc_id}: gold association coverage differs from its patient list")
        if set(p_assoc) != expected_patients:
            raise ValueError(f"{pmc_id}: prediction association coverage differs from input patient list")
        r = Counter({"tp": 0, "fp": 0, "fn": 0})
        for patient_id in sorted(expected_patients):
            counts = compare_sets(g_assoc[patient_id], p_assoc[patient_id])
            r.update(dict(zip(("tp", "fp", "fn"), counts)))
            patient_f1.append(float(prf(*counts)["f1"]))
        for name, counts in (("mention", m), ("document", d), ("relation", (r["tp"], r["fp"], r["fn"]))):
            totals[name].update(dict(zip(("tp", "fp", "fn"), counts)))
        per_document.append({"pmc_id": pmc_id, "mention": prf(*m), "document": prf(*d),
                             "relation": prf(r["tp"], r["fp"], r["fn"]), "patients": len(expected_patients)})
    details = {name: prf(counts["tp"], counts["fp"], counts["fn"]) for name, counts in totals.items()}
    metrics = {
        "menF1": details["mention"]["f1"], "docF1": details["document"]["f1"],
        "micF1": details["relation"]["f1"], "macF1": sum(patient_f1) / len(patient_f1) if patient_f1 else 1.0,
    }
    metrics["Score"] = sum(float(value) for value in metrics.values()) / 4
    return {"metrics": metrics, "counts": details, "documents": len(gold), "patients": len(patient_f1),
            "per_document": per_document}


def evaluate_report(gold_rows: list[dict[str, Any]], predicted_rows: list[dict[str, Any]],
                    body_scope: str = "all", no_id_doc: str = "text") -> dict[str, Any]:
    result = evaluate(gold_rows, predicted_rows, body_scope, no_id_doc)
    result.update({
        "evaluator_version": VERSION,
        "result_type": "local_public_rule_approximation_not_official",
        "configuration": {"body_scope": body_scope, "unlinked_document_mode": no_id_doc,
                          "unlinked_gold_mention": "span-only one-to-one after exact HPO matches",
                          "duplicate_entities": "deduplicated by span and split HPO ID",
                          "compound_relations": "split only full HP:[0-9]{7}(;HP:[0-9]{7})+ strings; deduplicate per patient",
                          "empty_patient_f1": 1.0},
        "caveats": [
            "Official evaluator source is unavailable; these numbers are not official scores.",
            "Default all includes every input passage for a consistent local comparison; exact official body filtering is unknown.",
            "body mode is a declared heuristic excluding title/abstract/references and selected metadata, not confirmed official behavior.",
            "Official document-level representation of -1 is unspecified; text/shared_id/ignore sensitivity is reported.",
            "NO entities are excluded; unlinked gold mention matching is span-only and one-to-one, while linked gold still requires its HPO ID.",
            "v2 splits syntactically valid HPO compound relation IDs; MESH/mixed/text values stay verbatim. HPO alt IDs are not canonicalized.",
            "Global empty-set precision/recall/F1 are defined as 1; official public text only specifies this convention for individual patients."
        ]})
    alternatives = {}
    for mode in ("text", "ignore", "shared_id"):
        if mode != no_id_doc:
            alternatives["unlinked_doc_" + mode] = evaluate(gold_rows, predicted_rows, body_scope, mode, validate=False)["metrics"]
    other_scope = "body" if body_scope == "all" else "all"
    alternatives["scope_" + other_scope] = evaluate(gold_rows, predicted_rows, other_scope, no_id_doc, validate=False)["metrics"]
    alternatives["strict_unlinked_identifier"] = evaluate(gold_rows, predicted_rows, body_scope, no_id_doc, wildcard_no_id=False, validate=False)["metrics"]
    result["sensitivity"] = alternatives
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output")
    parser.add_argument("--body-scope", choices=("all", "body"), default="all")
    parser.add_argument("--no-id-doc", choices=("text", "ignore", "shared_id"), default="text")
    args = parser.parse_args()
    try:
        result = evaluate_report(load_jsonl(args.gold), load_jsonl(args.pred), args.body_scope, args.no_id_doc)
        result["gold_sha256"] = file_sha256(args.gold)
        result["prediction_sha256"] = file_sha256(args.pred)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(json.dumps({"evaluator_version": VERSION, **result["metrics"], "documents": result["documents"],
                      "patients": result["patients"], "result_type": result["result_type"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
