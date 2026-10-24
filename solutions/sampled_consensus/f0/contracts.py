#!/usr/bin/env python3
"""F0 CPU contracts using fully synthetic responses over the pinned input universe."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import assemble as a


def reject(function, name):
    try:
        function()
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("Expected rejection: " + name)


def make_fixture(prompt, audit, seed, relations):
    mapping = {cid: [pid for pid in prompt["patient_ids"]
                     if (prompt["pmc_id"], pid, concept) in relations]
               for cid, concept in prompt["code_map"].items()}
    response = json.dumps(mapping)
    return {
        "row_id": prompt["row_id"], "pmc_id": prompt["pmc_id"], "response": response,
        "raw_with_special_tokens": response + "<|im_end|>",
        "input_tokens": audit["expanded"]["tokens"], "output_tokens": 3,
        "natural_eos": True, "hit_token_limit": False, "removed_demonstrations": 0,
        "seed": seed, "row_seed": a.f1.row_seed(seed, prompt["row_id"]), "fresh": True,
        "thinking": False, "has_think_marker": False, "generation_limit": 1600,
        "input_ids_sha256": audit["expanded"]["input_ids_sha256"],
        "rendered_sha256": audit["expanded"]["rendered_sha256"],
        "generated_ids": [100, 101, a.EOS_IDS[-1]], "configured_eos_ids": a.EOS_IDS,
        "_cpu_contract_fixture": True,
    }


def write_fixture_run(path, static_paths, streams):
    """Exercise the real freeze writer; mark every fixture source and freeze."""
    path.mkdir()
    raw_refs = {}
    for index, rows in enumerate(streams, 1):
        raw_path = path / f"synthetic-seed{index}.jsonl"
        a.save_bytes(raw_path, b"".join((json.dumps(row) + "\n").encode() for row in rows))
        raw_refs[f"seed{index}_raw"] = a.reference(raw_path)
    plan = {"fixture_only": True, "assembly_contract": a.ASSEMBLY_CONTRACT,
            "seeds": list(a.SEEDS), "generation_kwargs": a.GENERATION, "input_limit": 16384,
            "purpose": "CPU_SYNTHETIC_CONTRACT_ONLY_NO_MODEL_NO_QUALITY"}
    plan_path = path / "synthetic-plan.json"
    a.save_json(plan_path, plan)
    complete = {
        "fixture_only": True, "status": "COMPLETE_FIXED116_FRESH_NATIVE_NONTHINKING_NO_QUALITY",
        "fresh_rows": 116, "cached_rows": 0, "plan_sha256": a.sha(plan_path),
        "raw_ids_saved_before_decode": True, "raw_saved_before_parse": True,
        "raw_native_decoder_exact": True, "gold_read": False, "quality_calls": 0,
        "seeds": [{"seed": seed, "rows": 58, "fresh_rows": 58,
                   "responses_sha256": raw_refs[f"seed{index}_raw"]["sha256"]}
                  for index, seed in enumerate(a.SEEDS, 1)],
    }
    complete_path = path / "synthetic-complete.json"
    a.save_json(complete_path, complete)
    args = SimpleNamespace(**{key: str(value) for key, value in static_paths.items()},
                           plan=str(plan_path), actual_complete=str(complete_path),
                           seed1_raw=raw_refs["seed1_raw"]["path"], seed2_raw=raw_refs["seed2_raw"]["path"],
                           output_dir=str(path / "synthetic-freeze"))
    freeze = a.run(args, fixture_only=True)
    output = Path(args.output_dir)
    assert list(freeze["endpoints"]) == [a.ENDPOINT]
    assert len(list(output.glob("*.pred.jsonl"))) == 1
    assert (output / "prediction-freeze.json").is_file()
    assert freeze["fixture_only"] is True and freeze["status"] == "CPU_SYNTHETIC_PRIMARY_FREEZE_NOT_FOR_QUALITY"
    assert freeze["baseline_retained_primary"] == 315
    assert freeze["baseline_outside_candidate_pool_retained_primary"] == 39
    assert freeze["single_seed_predictions"] == freeze["greedy_streams"] == freeze["quality_calls"] == 0
    return freeze, args, plan, complete, raw_refs


def run(args):
    began = time.monotonic()
    paths = {name: Path(getattr(args, name)) for name in a.PINNED}
    for name, path in paths.items():
        a.require(a.sha(path) == a.PINNED[name], "Pinned contract input mismatch: " + name)
    baseline_lines, baseline = a.read_jsonl(paths["baseline"])
    _, prompts = a.read_jsonl(paths["prompts"])
    _, inputs = a.read_jsonl(paths["inputs"])
    audit = a.read_json(paths["row_audit"])
    by_prompt, by_audit, universe, base = a.validate_universe(prompts, audit, baseline, inputs)
    parser = a.f1.load_native_parser(paths["parser_source"])
    ordered_pairs = sorted(universe)
    shared = {pair for index, pair in enumerate(ordered_pairs) if index % 5 == 0}
    first_only = {pair for index, pair in enumerate(ordered_pairs) if index % 5 == 1}
    second_only = {pair for index, pair in enumerate(ordered_pairs) if index % 5 == 2}
    wanted = [shared | first_only, shared | second_only]
    order = a.generation_order(audit)
    streams = [[make_fixture(by_prompt[rid], by_audit[rid], seed, selected) for rid in order]
               for seed, selected in zip(a.SEEDS, wanted)]
    for rows, seed, selected in zip(streams, a.SEEDS, wanted):
        actual, diagnostic = a.f1.parse_stream(rows, by_prompt, by_audit, parser, a.THINK_IDS, seed=seed)
        shuffled, _ = a.f1.parse_stream(list(reversed(rows)), by_prompt, by_audit, parser, a.THINK_IDS, seed=seed)
        assert actual == shuffled == selected
        assert len(diagnostic) == 58 and all(row["strict_complete"] for row in diagnostic)
        assert {row["row_id"]: row["row_seed"] for row in rows} == {
            row["row_id"]: row["row_seed"] for row in reversed(rows)}
    primary = a.consensus(base, *wanted)
    assert primary == base | shared and base <= primary
    assert not ((first_only | second_only) - base) & primary
    assert len(base - universe) == 39 and (base - universe) <= primary
    assert a.consensus(base, set(), set()) == a.consensus(base, wanted[0], set()) == base
    payload = a.f1.assemble_prediction(baseline_lines, baseline, primary)
    parsed_documents = [json.loads(line) for line in payload.splitlines()]
    assert len(parsed_documents) == 16 and sum(len(d["entities"]) for d in parsed_documents) == 947
    assert a.relation_set(parsed_documents) == primary
    assert a.sha(a.F1_HELPERS_PATH) == a.F1_HELPERS_SHA
    good = next(row for row in streams[0] if len(by_prompt[row["row_id"]]["code_map"]) > 1
                and any(parser(row["response"], by_prompt[row["row_id"]]).values()))
    prompt, expected = by_prompt[good["row_id"]], by_audit[good["row_id"]]
    mapping = parser(good["response"], prompt)
    cid, pid = next(iter(mapping)), prompt["patient_ids"][0]
    cases = {}

    def response_case(name, value):
        row = deepcopy(good)
        row["response"] = value if type(value) is str else json.dumps(value)
        row["raw_with_special_tokens"] = row["response"] + "<|im_end|>"
        cases[name] = row

    bad = deepcopy(mapping)
    del bad[cid]
    response_case("missing_CID_no_fill", bad)
    bad = deepcopy(mapping)
    bad["C999"] = bad.pop(cid)
    response_case("unknown_CID_same_count", bad)
    response_case("duplicate_CID", good["response"][:-1] + "," + json.dumps(cid) + ":[]}")
    for name, value in [("invalid_PID", ["NOT_SUPPLIED"]), ("duplicate_PID", [pid, pid]),
                        ("numeric_PID", [1]), ("non_list_value", pid)]:
        bad = deepcopy(mapping)
        bad[cid] = value
        response_case(name, bad)
    for name, response in [("truncated_JSON", good["response"][:-1]),
                           ("JSON_fence", "```json\n" + good["response"] + "\n```"),
                           ("leading_prose", "Answer: " + good["response"]),
                           ("second_JSON_object", good["response"] + "{}")]:
        response_case(name, response)
    response_case("top_level_list", [mapping])
    row = deepcopy(good)
    row.update(generated_ids=[100, 101], output_tokens=2, natural_eos=False)
    cases["valid_JSON_without_EOS"] = row
    row = deepcopy(good)
    row.update(generated_ids=[100] * 1600, output_tokens=1600, natural_eos=False, hit_token_limit=True)
    cases["valid_JSON_at_nonEOS_cap"] = row
    row = deepcopy(good)
    row.update(generated_ids=[a.THINK_IDS[0], 100, a.EOS_IDS[-1]], has_think_marker=False)
    cases["think_ID_despite_false_flag"] = row
    row = deepcopy(good)
    row["raw_with_special_tokens"] = "<think>hidden</think>" + row["raw_with_special_tokens"]
    cases["think_special_decode"] = row
    cases["thinking_enabled"] = dict(good, thinking=True)
    cases["non_boolean_EOS_flag"] = dict(good, natural_eos="true")
    row = deepcopy(good)
    row["generated_ids"] = [a.EOS_IDS[0], 100, a.EOS_IDS[-1]]
    cases["tokens_after_EOS"] = row
    abstentions = {}
    for name, broken in cases.items():
        pairs, result = a.f1.parse_row(broken, prompt, expected, parser, a.THINK_IDS, seed=a.SEEDS[0])
        assert not pairs and not result["strict_complete"] and result["whole_row_abstention"]
        assert result["candidate_patients"] is None
        abstentions[name] = {"reasons": result["reasons"], "parser_error": result["parser_error"], "votes": 0}
    fatal = []
    for name, change in [("cached_sample", {"fresh": False}),
                         ("wrong_stream_seed", {"seed": a.SEEDS[1]}),
                         ("wrong_row_seed", {"row_seed": good["row_seed"] + 1}),
                         ("wrong_input_IDs", {"input_ids_sha256": "0" * 64}),
                         ("wrong_rendered_input", {"rendered_sha256": "0" * 64}),
                         ("wrong_document", {"pmc_id": "UNKNOWN"})]:
        broken = dict(good, **change)
        reject(lambda: a.f1.parse_row(broken, prompt, expected, parser, a.THINK_IDS, seed=a.SEEDS[0]), name)
        fatal.append(name)
    for name, rows in [("missing_row", streams[0][:-1]), ("duplicate_row", streams[0] + streams[0][:1]),
                       ("unknown_row", streams[0][:-1] + [dict(good, row_id="UNKNOWN")])]:
        reject(lambda: a.f1.parse_stream(rows, by_prompt, by_audit, parser, a.THINK_IDS, seed=a.SEEDS[0]), name)
        fatal.append(name)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    freeze, run_args, plan, complete, raw_refs = write_fixture_run(output / "fixture-formal-order", paths, streams)
    shuffled, _, _, _, _ = write_fixture_run(output / "fixture-reversed-order", paths,
                                             [list(reversed(rows)) for rows in streams])
    assert freeze["endpoints"][a.ENDPOINT]["sha256"] == shuffled["endpoints"][a.ENDPOINT]["sha256"]
    assert freeze["primary_relations"] == len(primary) and freeze["primary_added_relations"] == len(primary - base)
    reject(lambda: a.run(run_args), "synthetic_plan_for_production")
    fatal.append("synthetic_plan_for_production")
    bad = deepcopy(plan)
    bad["assembly_contract"]["formula"] = "B0 union S1 union S2"
    reject(lambda: a.verify_plan(bad, fixture_only=True), "changed_formula")
    fatal.append("changed_formula")
    bad = deepcopy(plan)
    bad["generation_kwargs"]["temperature"] = 1.0
    reject(lambda: a.verify_plan(bad, fixture_only=True), "changed_temperature")
    fatal.append("changed_temperature")
    for name, change in [("incomplete_actual", {"fresh_rows": 115}),
                         ("actual_cache", {"cached_rows": 1}),
                         ("wrong_actual_plan", {"plan_sha256": "0" * 64}),
                         ("raw_not_saved_first", {"raw_saved_before_parse": False})]:
        bad = dict(complete, **change)
        reject(lambda: a.verify_completion(bad, a.reference(run_args.plan), raw_refs, fixture_only=True), name)
        fatal.append(name)
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    report = {
        "status": "PASS_F0_CPU_CONTRACTS_NO_QUALITY", "fixture_only": True,
        "fixture_warning": "All responses and completion metadata are synthetic CPU fixtures; neither model inference nor production prediction freeze was performed",
        "sources": {name: a.reference(path) for name, path in paths.items()},
        "code": {"assemble": a.reference(a.__file__), "contracts": a.reference(__file__),
                 "f1_helpers": a.reference(a.F1_HELPERS_PATH)},
        "assembly_contract": a.ASSEMBLY_CONTRACT,
        "fixture_consensus": {"S1": len(wanted[0]), "S2": len(wanted[1]), "intersection": len(shared),
                              "primary": len(primary), "added": len(primary - base),
                              "all_B0_preserved": 315, "B0_outside_pool_preserved": 39},
        "stable_row_seed_and_order_independent_primary": True,
        "only_one_primary_prediction_and_freeze_each": True,
        "non_association_bytes_and_all_947_entities_exact": True,
        "whole_row_abstention_cases": abstentions, "fatal_rejection_cases": fatal,
        "synthetic_freezes": [a.reference(output / name / "synthetic-freeze/prediction-freeze.json")
                              for name in ("fixture-formal-order", "fixture-reversed-order")],
        "no_fallback_when_either_seed_abstains": True,
        "greedy_streams": 0, "single_seed_predictions": 0, "gold_read": False,
        "quality_calls": 0, "NN_calls": 0, "elapsed_seconds": time.monotonic() - began,
    }
    a.save_json(output / "cpu-contracts.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in a.PINNED:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "report": str(Path(args.output_dir) / "cpu-contracts.json"),
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
