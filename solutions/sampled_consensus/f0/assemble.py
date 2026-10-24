#!/usr/bin/env python3
"""Freeze the single F0 primary B0 union (S1 intersection S2), without labels.

The SHA-pinned F1 implementation supplies the audited strict parser, row gates,
row_id alignment, and byte-preserving prediction writer. No greedy stream or
single-seed prediction is consumed or emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


F1_HELPERS_SHA = "80caec32d9a3c57ebb86d54a0c31c9b4b2b193bace80dd18aa5f6e0bf20c874d"
F1_HELPERS_PATH = Path(__file__).resolve().parents[1] / "assemble.py"
if hashlib.sha256(F1_HELPERS_PATH.read_bytes()).hexdigest() != F1_HELPERS_SHA:
    raise ValueError("Audited F1 assembly helpers changed")
_spec = importlib.util.spec_from_file_location("_sampled_consensus_f0_audited_helpers", F1_HELPERS_PATH)
f1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = f1
_spec.loader.exec_module(f1)

require, sha, jsha = f1.require, f1.sha, f1.jsha
reference, read_json, read_jsonl = f1.reference, f1.read_json, f1.read_jsonl
indexed, relation_set = f1.indexed, f1.relation_set
save_json, save_bytes = f1.save_json, f1.save_bytes
SEEDS, EOS_IDS, THINK_IDS = f1.SEEDS, f1.EOS_IDS, f1.THINK_IDS
ENDPOINT = "primary-B0-union-S1-intersect-S2"
ENDPOINTS = (ENDPOINT,)
PINNED = {
    "baseline": "94504b974d5203fdde6b68ff10ca12bb67d24b0db235023023920b0ec595e1bc",
    "prompts": "d633225256dbf1cd5b64e94c74d7ce7f220b834cd748158082b306f12191fab2",
    "row_audit": "2fa5c495f8341ba4bf5f6ea4f731f75128f159f28ecb6302abaf7ead67872d02",
    "inputs": "ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37",
    "parser_source": "062cbf90e04cec32e635975a2f903efa5fe7c1774fe568f2066b23dda26d824f",
}
GENERATION = {
    "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "repetition_penalty": 1.0, "use_cache": True, "num_return_sequences": 1, "num_beams": 1,
    "max_new_tokens": 1600,
}
ASSEMBLY_CONTRACT = {
    "schema": "sampled-consensus-F0-primary-v1",
    "endpoint": ENDPOINT,
    "formula": "B0 union (S1 intersection S2)",
    "seeds": list(SEEDS),
    "rows_per_seed": 58, "sampled_fresh_rows": 116, "cached_rows": 0,
    "documents": 16, "patients": 47, "entities": 947,
    "candidate_concepts": 415, "candidate_patient_pairs": 1166,
    "baseline_relations": 315, "baseline_outside_candidate_pool": 39,
    "input_limit": 16384, "generation_limit": 1600,
    "input_tokens_per_seed": 505090, "maximum_input_tokens": 16245,
    "generation_order": "longest_first_then_remaining_ascending_input_tokens_row_id",
    "longest_original_index": 30, "longest_row_id": "3563609:relations:0",
    "generation_kwargs": GENERATION,
    "eos_token_ids": EOS_IDS, "think_token_ids": THINK_IDS,
    "parser_source_sha256": PINNED["parser_source"], "f1_helpers_sha256": F1_HELPERS_SHA,
    "invalid_response": "whole-row abstention; no JSON repair, missing-key fill, or fallback",
    "preservation": "all B0 relations and every non-association UTF-8 byte",
    "prediction_outputs": 1, "greedy_streams": 0, "single_seed_quality_calls": 0,
}


def generation_order(audit):
    ascending = sorted(audit, key=lambda row: (row["expanded"]["tokens"], row["row_id"]))
    longest = ascending[-1]
    return [longest["row_id"]] + [row["row_id"] for row in ascending[:-1]]


def validate_universe(prompts, audit, baseline, inputs):
    by_prompt, by_audit = indexed(prompts), indexed(audit)
    by_document, by_input = indexed(baseline, "pmc_id"), indexed(inputs, "pmc_id")
    require(len(prompts) == len(audit) == 58 and len(baseline) == len(inputs) == 16,
            "Expected complete F0: 58 prompt rows and 16 documents")
    require(set(by_prompt) == set(by_audit), "Prompt/audit coverage differs")
    require(list(by_document) == list(by_input), "B0/input document coverage or order differs")
    patients = {}
    for document in inputs:
        require("entities" not in document and "association" not in document and "answer" not in document,
                "Expected pure input without labels")
        pmc = document["pmc_id"]
        pids = [p["patient_id"] for p in document["patient"]]
        require(pids and all(type(pid) is str for pid in pids) and len(pids) == len(set(pids)),
                "Input has invalid/duplicate patients")
        require(set(pids) == {a["patient_id"] for a in by_document[pmc]["association"]},
                "B0/input patient coverage differs")
        require(document["pmid"] == by_document[pmc]["pmid"], "B0/input PMID identity differs")
        patients[pmc] = pids
    concepts, universe, covered = set(), set(), set()
    for index, row in enumerate(prompts):
        require(set(row) == {"row_id", "pmc_id", "messages", "code_map", "patient_ids"},
                "Unexpected pure-input prompt schema")
        expected = by_audit[row["row_id"]]
        require(expected["index"] == index and expected["pmc_id"] == row["pmc_id"]
                and expected["expanded_row_sha256"] == jsha(row), "Prompt identity/audit mismatch")
        pmc, code_map, pids = row["pmc_id"], row["code_map"], row["patient_ids"]
        require(pmc in patients and type(pids) is list and pids
                and all(type(pid) is str for pid in pids) and len(pids) == len(set(pids))
                and set(pids) == set(patients[pmc]), "Prompt/input PID mismatch")
        require(type(code_map) is dict and code_map
                and list(code_map) == [f"C{i:03d}" for i in range(len(code_map))], "Invalid supplied CID map")
        require(type(expected["expanded"]["tokens"]) is int
                and 0 < expected["expanded"]["tokens"] <= ASSEMBLY_CONTRACT["input_limit"],
                "Expanded prompt exceeds the fixed F0 input budget")
        covered.add(pmc)
        for concept in code_map.values():
            require(type(concept) is str and (pmc, concept) not in concepts,
                    "Duplicate/nonliteral concept across rows")
            concepts.add((pmc, concept))
            universe.update((pmc, pid, concept) for pid in pids)
    base = relation_set(baseline)
    require(covered == set(patients) and sum(map(len, patients.values())) == 47,
            "F0 document/patient universe differs")
    require(len(concepts) == 415 and len(universe) == 1166, "F0 candidate universe differs")
    require(sum(len(d["entities"]) for d in baseline) == 947, "B0 entity count differs")
    require(len(base) == 315 and len(base - universe) == 39, "B0 relation counts differ")
    require(sum(row["expanded"]["tokens"] for row in audit) == 505090,
            "F0 expanded input token total differs")
    order = generation_order(audit)
    require(order[0] == "3563609:relations:0" and by_audit[order[0]]["index"] == 30
            and by_audit[order[0]]["expanded"]["tokens"] == 16245, "F0 longest row differs")
    return by_prompt, by_audit, universe, base


def consensus(base, first, second):
    return base | (first & second)


def verify_plan(plan, *, fixture_only=False):
    require(plan.get("assembly_contract") == ASSEMBLY_CONTRACT, "F0 assembly contract differs from preregistration")
    require(plan.get("seeds") == list(SEEDS) and plan.get("generation_kwargs") == GENERATION
            and type(plan.get("input_limit")) is int and plan["input_limit"] == 16384,
            "Frozen sampling or input parameters differ")
    require(plan.get("fixture_only", False) is fixture_only, "CPU fixture is not a production plan")


def verify_completion(complete, plan_ref, raw_refs, *, fixture_only=False):
    require(complete.get("status") == "COMPLETE_FIXED116_FRESH_NATIVE_NONTHINKING_NO_QUALITY",
            "No complete formal 116-row native inference terminal")
    require(type(complete.get("fresh_rows")) is int and complete["fresh_rows"] == 116
            and type(complete.get("cached_rows")) is int and complete["cached_rows"] == 0,
            "F0 requires all 116 formal rows fresh")
    require(complete.get("plan_sha256") == plan_ref["sha256"], "Actual inference plan SHA differs")
    require(complete.get("raw_ids_saved_before_decode") is True and complete.get("raw_saved_before_parse") is True
            and complete.get("raw_native_decoder_exact") is True, "Actual raw-first/decoder evidence missing")
    require(complete.get("gold_read") is False and complete.get("quality_calls") == 0,
            "Actual inference terminal unexpectedly reports quality/labels")
    require(complete.get("fixture_only", False) is fixture_only, "CPU fixture is not a production terminal")
    rows = complete.get("seeds")
    require(type(rows) is list and len(rows) == 2 and [row.get("seed") for row in rows] == list(SEEDS),
            "Actual seed stream coverage differs")
    for index, row in enumerate(rows, 1):
        require(type(row.get("rows")) is int and type(row.get("fresh_rows")) is int
                and row["rows"] == row["fresh_rows"] == 58, "Actual stream lacks 58 fresh rows")
        require(row.get("responses_sha256") == raw_refs[f"seed{index}_raw"]["sha256"],
                "Actual stream raw SHA differs")


def run(args, *, fixture_only=False):
    """Production CLI cannot enable fixture_only; contracts label their outputs."""
    refs = {}
    for name, expected in PINNED.items():
        refs[name] = reference(getattr(args, name))
        require(refs[name]["sha256"] == expected, "Pinned F0 source differs: " + name)
    refs.update(plan=reference(args.plan), actual_complete=reference(args.actual_complete))
    plan, complete = read_json(args.plan), read_json(args.actual_complete)
    verify_plan(plan, fixture_only=fixture_only)
    for name in ("seed1_raw", "seed2_raw"):
        refs[name] = reference(getattr(args, name))
    verify_completion(complete, refs["plan"], refs, fixture_only=fixture_only)
    baseline_lines, baseline = read_jsonl(args.baseline)
    _, prompts = read_jsonl(args.prompts)
    _, inputs = read_jsonl(args.inputs)
    audit = read_json(args.row_audit)
    by_prompt, by_audit, universe, base = validate_universe(prompts, audit, baseline, inputs)
    parser = f1.load_native_parser(args.parser_source)
    streams, diagnostics, stats, observed_orders = [], [], {}, {}
    for index, seed in enumerate(SEEDS, 1):
        name = f"S{index}"
        _, records = read_jsonl(getattr(args, f"seed{index}_raw"))
        require(all(row.get("_cpu_contract_fixture", False) is fixture_only for row in records),
                "Synthetic contract raw cannot be used for a production freeze")
        pairs, rows = f1.parse_stream(records, by_prompt, by_audit, parser, THINK_IDS, seed=seed)
        require(pairs <= universe, "Sampled relations escaped the fixed F0 candidate/PID universe")
        streams.append(pairs)
        observed_orders[name] = [row["row_id"] for row in records]
        for row in rows:
            row["stream"] = name
        diagnostics.extend(rows)
        stats[name] = {
            "seed": seed, "rows": len(rows), "fresh_rows": 58, "cached_rows": 0,
            "strict_complete_rows": sum(row["strict_complete"] for row in rows),
            "whole_row_abstentions": sum(row["whole_row_abstention"] for row in rows),
            "relations": len(pairs),
            "reason_counts": dict(Counter(reason for row in rows for reason in row["reasons"])),
        }
    primary = consensus(base, *streams)
    require(base <= primary and (base - universe) <= primary, "Primary lost a B0 relation")
    require(primary - base <= universe, "Primary introduced a relation outside the candidate universe")
    payload = f1.assemble_prediction(baseline_lines, baseline, primary)
    # Source identities are stable between reading and final freeze.
    for ref in refs.values():
        require(sha(ref["path"]) == ref["sha256"], "Source changed during assembly: " + ref["path"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / (ENDPOINT + ".pred.jsonl")
    save_bytes(prediction_path, payload)
    save_bytes(output / "row-diagnostics.jsonl",
               b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        for row in diagnostics))
    freeze = {
        "status": "CPU_SYNTHETIC_PRIMARY_FREEZE_NOT_FOR_QUALITY" if fixture_only
                  else "PRIMARY_COMPLETE_PREDICTION_FROZEN_NO_QUALITY",
        "fixture_only": fixture_only, "assembly_contract": ASSEMBLY_CONTRACT,
        "source": reference(__file__), "f1_helpers": reference(F1_HELPERS_PATH),
        "references": refs, "endpoints": {ENDPOINT: reference(prediction_path)},
        "row_diagnostics": reference(output / "row-diagnostics.jsonl"),
        "streams": stats, "model_raw_rows": 116, "sampled_fresh_rows": 116,
        "documents": 16, "patients": 47, "entities_each": 947,
        "candidate_concepts": 415, "candidate_patient_pairs": 1166,
        "baseline_relations": len(base), "baseline_outside_candidate_pool": len(base - universe),
        "baseline_retained_primary": len(base & primary),
        "baseline_outside_candidate_pool_retained_primary": len((base - universe) & primary),
        "primary_relations": len(primary), "primary_added_relations": len(primary - base),
        "generation_order_expected": generation_order(audit), "raw_row_orders_observed": observed_orders,
        "alignment": "row_id; aggregate storage order cannot change votes",
        "nonrelation_bytes_preserved": True, "single_seed_predictions": 0,
        "greedy_streams": 0, "gold_read": False, "quality_calls": 0,
    }
    save_json(output / "prediction-freeze.json", freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PINNED:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--actual-complete", required=True)
    parser.add_argument("--seed1-raw", required=True)
    parser.add_argument("--seed2-raw", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "endpoints": result["endpoints"],
                      "freeze": str(Path(args.output_dir).resolve() / "prediction-freeze.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
