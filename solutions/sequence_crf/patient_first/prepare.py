#!/usr/bin/env python3
"""CPU-only schema preparation. F1 is input-only; no targets, weights or scores."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
from patient_first import compact, inverse, parse_json, synthetic_contracts, transform, INPUT_KEYS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    # Exact convention of the existing native training token audit.
    return sha(json.dumps(value).encode())


def ref(path):
    p = Path(path)
    b = p.read_bytes()
    return {"path": str(p.resolve()), "bytes": len(b), "sha256": sha(b)}


def save(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_rows(path, values):
    with Path(path).open("x") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def token_audit(row, encoded):
    return {"row_id": row["row_id"], "pmc_id": str(row["pmc_id"]),
            "full_tokens": len(encoded["input_ids"]), "prefix_tokens": encoded["prefix_tokens"],
            "supervised_tokens": encoded["supervised_tokens"], "eos_token_id": encoded["eos_token_id"],
            "rendered_prefix_sha256": encoded["rendered_prefix_sha256"],
            "rendered_full_sha256": encoded["rendered_full_sha256"],
            "input_ids_sha256": digest(encoded["input_ids"]), "labels_sha256": digest(encoded["labels"])}


def stats(values):
    assert values
    ordered = sorted(values)
    return {"sum": sum(values), "minimum": min(values), "maximum": max(values),
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p90_nearest_rank": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)]}


class FrozenTokenizer:
    def __init__(self, tokenizer_path, template_path):
        import tokenizers
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        assert tokenizers.__version__ == "0.23.1" and jinja2.__version__ == "3.1.6"
        self.backend = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def fail(message):
            raise ValueError(message)
        env.globals["raise_exception"] = fail
        self.template = env.from_string(Path(template_path).read_text())
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self.backend.token_to_id(self.eos_token)
        assert self.eos_token_id == 248046

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert enable_thinking is False and return_dict is False
        text = self.template.render(messages=messages, tools=None,
                                    add_generation_prompt=add_generation_prompt, enable_thinking=False)
        return self.backend.encode(text, add_special_tokens=False).ids if tokenize else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    plan = parse_json(args.plan.read_text())
    references = plan["references"]
    for value in references.values():
        assert ref(value["path"]) == value, (value["path"], "SHA or size changed")
    assert ref(__file__)["sha256"] == plan["prepare_sha256"]
    assert ref(Path(__file__).with_name("patient_first.py"))["sha256"] == plan["transform_sha256"]
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU preparation limit")))
    signal.alarm(plan["cpu_seconds"])
    args.output.mkdir(parents=True, exist_ok=False)
    def load(name):
        return parse_json(Path(references[name]["path"]).read_text())
    def rows(name):
        return [parse_json(line) for line in Path(references[name]["path"]).read_text().splitlines()]
    train, queries = rows("train"), rows("F1_input")
    assert len(train) == 124 and len(queries) == 80
    assert all(set(x) == INPUT_KEYS | {"answer"} for x in train)
    assert all(set(x) == INPUT_KEYS for x in queries), "F1 file must contain only input fields"
    provenance, native, old_manifest = load("preparation_manifest"), load("native_train_audit"), load("old_training_manifest")
    train_docs = sorted({str(x["pmc_id"]) for x in train})
    query_docs = sorted({str(x["pmc_id"]) for x in queries})
    assert len(train_docs) == 32 and len(query_docs) == 16
    assert train_docs == provenance["training_document_ids"]
    assert not set(train_docs) & set(provenance["excluded_document_ids"])
    assert set(query_docs) <= set(provenance["excluded_document_ids"])
    assert not set(train_docs) & set(query_docs)
    assert provenance["training_sha256"] == references["train"]["sha256"] == native["train_sha256"]
    assert provenance["all_target_documents_excluded_from_own_generators"]
    assert provenance["no_F1_or_outer0_in_any_generator_or_SFT_training"]
    for chain in provenance["provenance"]:
        target, generator = set(chain["target_ids"]), set(chain["generator_ids"])
        assert len(target) == len(generator) == 16 and not target & generator
        assert target | generator == set(train_docs)
        assert not (target | generator) & set(query_docs)
    assert old_manifest["examples"] == 124 and old_manifest["epochs"] == 2
    assert old_manifest["total_optimizer_steps"] == 32 and old_manifest["validation_labels_used"] is False
    expected_audits = {r["row_id"]: r for r in native["rows"]}
    old_query_audits = {r["row_id"]: r for r in load("previous_F1_input_audit")}
    encode = function(references["encode_source"]["path"], "encode", {"hashlib": hashlib})
    tokenizer = FrozenTokenizer(references["tokenizer"]["path"], references["template"]["path"])
    contracts = synthetic_contracts()
    train_new, train_audit, matrix_counts, matrices = [], [], Counter(), []
    for row in train:
        assert row["row_id"] not in {x["row_id"] for x in train_new}
        new = transform(row)
        assert inverse(new) == row
        assert compact(inverse(new)) == compact(row)
        assert new["messages"][1] == row["messages"][1]
        assert new["messages"][0]["content"].split("\n")[2:] == row["messages"][0]["content"].split("\n")[2:]
        old_target, new_target = parse_json(row["answer"]), parse_json(new["answer"])
        codes, patients = list(row["code_map"]), row["patient_ids"]
        assert list(old_target) == codes and list(new_target) == patients
        assert all(ps == [p for p in patients if p in ps] for ps in old_target.values())
        assert all(cs == [c for c in codes if c in cs] for cs in new_target.values())
        old_positive = {(p, c) for c, ps in old_target.items() for p in ps}
        new_positive = {(p, c) for p, cs in new_target.items() for c in cs}
        universe = {(p, c) for p in patients for c in codes}
        assert old_positive == new_positive and universe - old_positive == universe - new_positive
        old_tokens = token_audit(row, encode(row, tokenizer))
        assert old_tokens == expected_audits[row["row_id"]], (row["row_id"], "native old tokenizer mismatch")
        new_tokens = token_audit(new, encode(new, tokenizer))
        assert new_tokens["full_tokens"] <= 6144 and old_tokens["full_tokens"] <= 6144
        matrix_counts.update({"candidate_values": len(codes), "patient_output_keys": len(patients),
                              "positive_pairs": len(old_positive), "negative_pairs": len(universe - old_positive),
                              "all_empty_rows": int(not old_positive),
                              "old_empty_CID_values": sum(not x for x in old_target.values()),
                              "new_empty_patient_values": sum(not x for x in new_target.values())})
        matrices.append({"row_id": row["row_id"], "pmc_id": str(row["pmc_id"]), "patients": patients, "codes": codes,
                         "positive_pairs": len(old_positive), "negative_pairs": len(universe - old_positive),
                         "positive_matrix_sha256": digest(sorted(old_positive)),
                         "negative_matrix_sha256": digest(sorted(universe - old_positive)),
                         "double_inverse_all_fields_and_answer_byteexact": True})
        train_audit.append({"row_id": row["row_id"], "old": old_tokens, "patient_first": new_tokens,
                            "full_token_delta": new_tokens["full_tokens"] - old_tokens["full_tokens"],
                            "prefix_token_delta": new_tokens["prefix_tokens"] - old_tokens["prefix_tokens"],
                            "label_token_delta": new_tokens["supervised_tokens"] - old_tokens["supervised_tokens"]})
        train_new.append(new)
    assert matrix_counts["candidate_values"] == 880
    assert matrix_counts["positive_pairs"] == 594 and matrix_counts["negative_pairs"] == 1912
    new_queries, query_audit = [], []
    for row in queries:
        assert row["row_id"] not in {x["row_id"] for x in new_queries}
        new = transform(row)
        assert set(new) == INPUT_KEYS and inverse(new) == row
        assert new["messages"][1] == row["messages"][1]
        audits = {}
        for label, obj in [("old", row), ("patient_first", new)]:
            text = tokenizer.apply_chat_template(obj["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            values = tokenizer.apply_chat_template(obj["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            audits[label] = {"input_tokens": len(values), "rendered_sha256": sha(text.encode()), "input_ids_sha256": digest(values)}
        expected = old_query_audits[row["row_id"]]
        assert audits["old"]["input_tokens"] == expected["original_input_tokens"]
        assert audits["old"]["rendered_sha256"] == expected["original_render_sha256"]
        assert audits["patient_first"]["input_tokens"] < 14000
        query_audit.append({"row_id": row["row_id"], "pmc_id": str(row["pmc_id"]), **audits,
                            "input_only": True, "patient_count": len(row["patient_ids"]), "candidate_count": len(row["code_map"]),
                            "candidate_patient_pairs": len(row["patient_ids"]) * len(row["code_map"]),
                            "user_and_candidate_order_unchanged": True, "inverse_all_fields_exact": True})
        new_queries.append(new)
    assert sum(x["candidate_count"] for x in query_audit) == 587
    assert sum(x["candidate_patient_pairs"] for x in query_audit) == 1557
    assert sum(x["old"]["input_tokens"] for x in query_audit) == 250696
    save_rows(args.output / "train124.patient-first.jsonl", train_new)
    save_rows(args.output / "F1.input80.patient-first.jsonl", new_queries)
    save(args.output / "training-matrix-audit.json", matrices)
    save(args.output / "train-token-audit.json", train_audit)
    save(args.output / "F1-input-token-audit.json", query_audit)
    save(args.output / "synthetic-contracts.json", contracts)
    aggregates = {}
    for kind in ("old", "patient_first"):
        aggregates[kind] = {name: stats([r[kind][name] for r in train_audit]) for name in ("full_tokens", "prefix_tokens", "supervised_tokens")}
        longest = max(train_audit, key=lambda r: r[kind]["full_tokens"])
        aggregates[kind]["longest_row"] = longest["row_id"]
    assert aggregates["old"]["full_tokens"]["sum"] == 402594
    assert aggregates["old"]["supervised_tokens"]["sum"] == 7170
    summary = {"status": "PASS_CPU_patient_first_closed_set_preparation_no_training_or_quality",
               "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
               "plan": ref(args.plan), "sources": references, "prepare_source": ref(__file__),
               "transform_source": ref(Path(__file__).with_name("patient_first.py")),
               "training": {"rows": 124, "documents": 32, "document_ids": train_docs,
                            **dict(matrix_counts), "total_pairs": 2506, "old_native_token_audit_all_fields_exact": True,
                            "double_inverse_all_fields_and_targets_byteexact": True, "token_statistics": aggregates,
                            "delta_total_tokens": aggregates["patient_first"]["full_tokens"]["sum"] - 402594,
                            "delta_label_tokens": aggregates["patient_first"]["supervised_tokens"]["sum"] - 7170,
                            "delta_prefix_tokens": stats([x["prefix_token_delta"] for x in train_audit]),
                            "truncations": 0, "supervised_EOS_one_each": True},
               "F1_input_only": {"rows": 80, "documents": 16, "document_ids": query_docs,
                                 "unique_patients": len({(r["pmc_id"], p) for r in queries for p in r["patient_ids"]}),
                                 "candidate_values": 587, "candidate_patient_pairs": 1557,
                                 "old_render_and_counts_match_previous_audit": True,
                                 "old_tokens": stats([x["old"]["input_tokens"] for x in query_audit]),
                                 "patient_first_tokens": stats([x["patient_first"]["input_tokens"] for x in query_audit]),
                                 "longest_row": max(query_audit, key=lambda x: x["patient_first"]["input_tokens"])["row_id"],
                                 "answer_fields": 0, "targets_read": False},
               "changes": ["Only first two system lines specify patient-to-CID closed-set output", "Training target matrix transposed deterministically; original patient and candidate order retained"],
               "unchanged": ["row IDs and row order", "verbatim entire user messages", "all candidate fields and code_map", "all supplied patients", "remaining clinical instructions", "all positive and negative train pair labels", "assistant-only mean-CE mask and EOS semantics"],
               "isolation": {"training_vs_F1_overlap": 0, "both_training_generator_chains_exclude_F0_F1": True,
                             "existing_raw_HPO_labels_no_canonical_remap": True, "no_gold_added_candidates": True,
                             "provenance_reused_not_retrained_generators": True},
               "runtime": {"python": sys.version.split()[0], "tokenizers": "0.23.1", "jinja2": "3.1.6",
                           "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules,
                           "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
               "no_model_weights_loaded": True, "no_GPU_or_training_or_quality": True,
               "new_native_Transformers_GPU_token_or_gradient_contract_not_yet_run": True,
               "synthetic_contracts": contracts,
               "artifacts": [ref(p) for p in sorted(args.output.iterdir()) if p.is_file()]}
    assert not any(name in sys.modules for name in ("torch", "transformers", "peft"))
    save(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "seconds": summary["elapsed_seconds"],
                      "training_new_tokens": aggregates["patient_first"]["full_tokens"],
                      "training_new_labels": aggregates["patient_first"]["supervised_tokens"],
                      "F1_new_tokens": summary["F1_input_only"]["patient_first_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
