#!/usr/bin/env python3
"""Discriminating CPU contracts; native G is replayed as explicitly synthetic S1/S2.

No model, torch, labels, evaluator, or actual sampled-quality endpoint is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import assemble as a


def expect_rejection(function, message):
    try:
        function()
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError(message)


def sampled_fixture(raw, ids, audit, seed):
    """Hydrate old G for tests only; this does not represent a fresh generation."""
    result = deepcopy(raw)
    result.update(generated_ids=deepcopy(ids["generated_ids"]),
                  configured_eos_ids=deepcopy(ids["configured_eos_ids"]),
                  input_ids_sha256=ids["input_ids_sha256"],
                  rendered_sha256=audit["expanded"]["rendered_sha256"],
                  seed=seed, row_seed=a.row_seed(seed, raw["row_id"]), fresh=True,
                  generation_limit=a.GENERATION_LIMIT, thinking=False, has_think_marker=False,
                  # Mocked decoded-special metadata is confined to in-memory CPU fixtures.
                  raw_with_special_tokens=raw["response"] + "<|im_end|>")
    return result


def run(args):
    started = time.monotonic()
    native = Path(args.native_dir)
    paths = {"parser_source": Path(args.parser_source), "baseline": Path(args.baseline),
             "prompts": native / "expanded80.jsonl", "row_audit": native / "row-audit.json",
             "greedy_raw": native / "terminal/responses.jsonl",
             "greedy_canonical": native / "terminal/canonical.jsonl"}
    for name, path in paths.items():
        a.require(a.sha(path) == a.PINNED[name], "Contract source SHA mismatch: " + name)
    old_lines, baseline = a.read_jsonl(paths["baseline"])
    _, prompts = a.read_jsonl(paths["prompts"])
    audit = a.read_json(paths["row_audit"])
    by_prompt, by_audit, universe, base = a.validate_universe(prompts, audit, baseline)
    parser = a.load_native_parser(paths["parser_source"])
    _, raw = a.read_jsonl(paths["greedy_raw"])
    _, canonical = a.read_jsonl(paths["greedy_canonical"])
    canonical_by_row = a.indexed(canonical)
    generated = {r["row_id"]: a.read_json(native / f"terminal/generated-ids/{by_audit[r['row_id']]['index']:03d}.json")
                 for r in prompts}
    greedy, greedy_diagnostics = a.parse_stream(raw, by_prompt, by_audit, parser, a.THINK_IDS,
                                                ids_by_row=generated)
    assert len(greedy_diagnostics) == 80
    assert all(d["strict_complete"] and d["actual_natural_eos"] for d in greedy_diagnostics)
    assert all(d["candidate_patients"] == canonical_by_row[d["row_id"]]["candidate_patients"]
               for d in greedy_diagnostics)
    assert sum(c["provenance"] == "cached_probe" for c in canonical) == 1
    assert sum(c["provenance"] == "fresh" for c in canonical) == 79
    fixtures = [[sampled_fixture(row, generated[row["row_id"]], by_audit[row["row_id"]], seed)
                 for row in raw] for seed in a.SEEDS]
    # The two fixture files are intentionally not persisted as actual fresh raw.
    sampled_pairs = []
    for records, seed in zip(fixtures, a.SEEDS):
        pairs, diagnostics = a.parse_stream(list(reversed(records)), by_prompt, by_audit, parser,
                                            a.THINK_IDS, seed=seed)
        assert pairs == greedy and len(diagnostics) == 80
        assert all(d["strict_complete"] for d in diagnostics)
        sampled_pairs.append(pairs)
    primary, diagnostic = a.consensus(base, greedy, *sampled_pairs)
    assert primary == base | greedy and diagnostic == greedy
    assert len(base & primary) == 471 and len((base - universe) & primary) == 88
    payloads = [a.assemble_prediction(old_lines, baseline, pairs) for pairs in (primary, diagnostic)]
    assert all(len(payload.splitlines()) == 16 for payload in payloads)
    assert all(sum(len(json.loads(line)["entities"]) for line in payload.splitlines()) == 1189 for payload in payloads)
    good_index = next(i for i, row in enumerate(fixtures[0])
                      if any(parser(row["response"], by_prompt[row["row_id"]]).values())
                      and len(by_prompt[row["row_id"]]["code_map"]) > 1)
    good = fixtures[0][good_index]
    prompt, expected = by_prompt[good["row_id"]], by_audit[good["row_id"]]
    mapping = parser(good["response"], prompt)
    first_cid = next(iter(mapping))
    legal_pid = prompt["patient_ids"][0]
    cases = {}

    def response_case(name, value):
        broken = deepcopy(good)
        broken["response"] = value if type(value) is str else json.dumps(value)
        broken["raw_with_special_tokens"] = broken["response"] + "<|im_end|>"
        cases[name] = broken

    bad = deepcopy(mapping)
    del bad[first_cid]
    response_case("missing_CID_no_empty_fill", bad)
    bad = deepcopy(mapping)
    bad["C999"] = bad.pop(first_cid)
    response_case("unknown_CID_same_key_count", bad)
    response_case("duplicate_CID", good["response"][:-1] + "," + json.dumps(first_cid) + ":[]}")
    for name, value in [("illegal_PID", ["NEVER_A_SUPPLIED_PID"]),
                        ("duplicate_PID", [legal_pid, legal_pid]),
                        ("numeric_PID", [1]), ("PID_value_not_list", legal_pid)]:
        bad = deepcopy(mapping)
        bad[first_cid] = value
        response_case(name, bad)
    response_case("truncated_JSON", good["response"][:-1])
    response_case("JSON_fence", "```json\n" + good["response"] + "\n```")
    response_case("leading_prose", "Answer: " + good["response"])
    response_case("trailing_second_object", good["response"] + "{}")
    response_case("top_level_list", [mapping])
    broken = deepcopy(good)
    broken["generated_ids"] = broken["generated_ids"][:-1]
    broken["output_tokens"] -= 1
    broken["natural_eos"] = False
    cases["valid_JSON_without_EOS"] = broken
    broken = deepcopy(good)
    broken["generated_ids"] = [100] * a.GENERATION_LIMIT
    broken.update(output_tokens=a.GENERATION_LIMIT, natural_eos=False, hit_token_limit=True)
    cases["valid_JSON_at_nonEOS_cap"] = broken
    broken = deepcopy(good)
    broken["generated_ids"].insert(0, a.THINK_IDS[0])
    broken["output_tokens"] += 1
    # A false marker flag must not conceal the actual think special token.
    cases["think_ID_with_false_flag"] = broken
    broken = deepcopy(good)
    broken["raw_with_special_tokens"] = "<think>hidden</think>" + broken["raw_with_special_tokens"]
    cases["think_special_decode_with_false_flag"] = broken
    broken = deepcopy(good)
    broken["thinking"] = True
    cases["thinking_mode_flag"] = broken
    broken = deepcopy(good)
    broken["natural_eos"] = "true"
    cases["non_boolean_EOS_flag"] = broken
    broken = deepcopy(good)
    broken["generated_ids"].insert(0, a.EOS_IDS[0])
    broken["output_tokens"] += 1
    cases["tokens_after_EOS"] = broken
    outcomes = {}
    for name, broken in cases.items():
        pairs, record = a.parse_row(broken, prompt, expected, parser, a.THINK_IDS, seed=a.SEEDS[0])
        assert not pairs and record["whole_row_abstention"] and record["candidate_patients"] is None, name
        assert not record["strict_complete"], name
        outcomes[name] = {"reasons": record["reasons"], "parser_error": record["parser_error"],
                          "entire_row_votes": 0}
    identity_checks = []
    for name, change in [("cached_sample_rejected", {"fresh": False}),
                         ("wrong_row_seed_rejected", {"row_seed": good["row_seed"] + 1}),
                         ("wrong_stream_seed_rejected", {"seed": a.SEEDS[1]}),
                         ("wrong_input_ids_rejected", {"input_ids_sha256": "0" * 64}),
                         ("wrong_document_rejected", {"pmc_id": "UNKNOWN"})]:
        broken = dict(good, **change)
        expect_rejection(lambda: a.parse_row(broken, prompt, expected, parser, a.THINK_IDS,
                                            seed=a.SEEDS[0]), name)
        identity_checks.append(name)
    for name, rows in [("missing_row_rejected", fixtures[0][:-1]),
                       ("duplicate_row_rejected", fixtures[0] + fixtures[0][:1])]:
        expect_rejection(lambda: a.parse_stream(rows, by_prompt, by_audit, parser, a.THINK_IDS,
                                                seed=a.SEEDS[0]), name)
        identity_checks.append(name)
    # Distinguish both preregistered formulas: a lone G vote is never sufficient,
    # S1/S2 agreement does not require G, and diagnostic does not inherit B.
    outside, g_only, g_s1, g_s2, s_both, s1_only = [tuple(["D", "P", x]) for x in "abcdef"]
    actual = a.consensus({outside}, {g_only, g_s1, g_s2}, {g_s1, s_both, s1_only}, {g_s2, s_both})
    assert actual == ({outside, s_both}, {g_s1, g_s2, s_both})
    no_fallback = a.consensus({outside}, {g_only}, set(), set())
    assert no_fallback == ({outside}, set())
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    report = {
        "status": "PASS_CPU_CONTRACTS_NO_QUALITY",
        "fixture_warning": "S1/S2 are in-memory copies of old G with synthetic metadata, never actual fresh raw or quality endpoints",
        "sources": {name: a.reference(path) for name, path in paths.items()},
        "code": {"assemble": a.reference(a.__file__), "contracts": a.reference(__file__)},
        "native_G": {"strict_complete_rows": 80, "actual_natural_eos_rows": 80,
                     "canonical_exact_rows": 80, "fresh_rows": 79, "cached_rows": 1,
                     "relations": len(greedy)},
        "replay": {"seed_streams": 2, "reordered_rows_each": 80,
                   "primary_equals_B_union_G": True, "diagnostic_equals_G": True,
                   "primary_relations": len(primary), "diagnostic_relations": len(diagnostic),
                   "baseline_relations_preserved": 471, "outside_pool_preserved": 88,
                   "entities_each": 1189, "documents_each": 16, "nonrelation_bytes_preserved": True},
        "whole_row_abstention_cases": outcomes, "fatal_identity_cases": identity_checks,
        "distinct_consensus_formula_checks": True, "no_greedy_fallback": True,
        "torch_imported": False, "gold_read": False, "quality_calls": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    a.save_json(output / "cpu-contracts.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--parser-source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "report": str(Path(args.output_dir) / "cpu-contracts.json"),
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
