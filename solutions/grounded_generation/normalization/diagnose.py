"""Train-only coverage controls and one fixed-policy diagnostic; never held gold."""
from collections import Counter, defaultdict
import argparse
import copy
import json
from pathlib import Path

try:
    from .strict import StrictNormalizer, normalized, read_jsonl, sha256
    from .semantic import load_policy, precheck, decide
except ImportError:
    from strict import StrictNormalizer, normalized, read_jsonl, sha256
    from semantic import load_policy, precheck, decide

TARGET_SHA256 = "bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08"
COVERAGE_SHA256 = "5c3d654631bc543c99de193627476de00cbc75e8c38f0c84c220b57ddd974a54"


def ref(path):
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha256(path)}


def put(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def filtered_result(normalizer, quote, control, pmc_id):
    if control == "strict_resubstitution":
        return normalizer.normalize(quote)
    def keep(row):
        return row["source"] != "train_alias" or (control == "leave_document_out" and row["pmc_id"] != pmc_id)
    key = normalized(quote)
    folded = {ids: [row for row in rows if keep(row)] for ids, rows in normalizer.folded.get(key, {}).items()}
    folded = {ids: rows for ids, rows in folded.items() if rows}
    exact = {ids: [row for row in rows if keep(row)] for ids, rows in normalizer.exact.get(quote, {}).items()}
    exact = {ids: rows for ids, rows in exact.items() if rows}
    clone = copy.copy(normalizer)
    clone.folded = {key: folded} if folded else {}
    clone.exact = {quote: exact} if exact else {}
    return clone.normalize(quote)


def prepare(args):
    normalizer = StrictNormalizer(args.hpo, args.train)
    policy = load_policy(args.policy)
    if sha256(args.targets) != TARGET_SHA256 or sha256(args.coverage) != COVERAGE_SHA256:
        raise ValueError("Frozen train-only target/coverage bytes changed")
    targets = read_jsonl(args.targets)
    coverage = {row["row_id"]: row for row in json.loads(Path(args.coverage).read_text())}
    if len(targets) != 87 or set(str(row["pmc_id"]) for row in targets) != set(normalizer.training_ids):
        raise ValueError("Expected all87 train32 target rows")
    cases = []
    for row in targets:
        selected = defaultdict(list)
        for item in coverage[row["row_id"]]["selected_weak_evidence"]:
            selected[json.dumps(item["chosen"]["triple"], ensure_ascii=False)].append(item)
        for triple in json.loads(row["answer"]):
            choices = selected[json.dumps(triple, ensure_ascii=False)]
            if not choices:
                raise ValueError("Target has no training provenance")
            gold = set()
            for item in choices:
                keys = item["chosen"]["keys"]
                gold.add(tuple(sorted(key[1] if key[0] == "hpo" else "-1" for key in keys)))
            cases.append({"row_id": row["row_id"], "pmc_id": row["pmc_id"], "triple": triple,
                "quote": triple[1], "offset": choices[0]["chosen"]["offset"],
                "gold_options": [list(ids) for ids in sorted(gold)],
                "source_entity_indices": sorted({item["chosen"]["entity_index"] for item in choices})})
    controls = {}
    for control in ["strict_resubstitution", "public_only", "leave_document_out"]:
        counts, eligible, guard_reasons = Counter(), set(), Counter()
        for row in cases:
            result = filtered_result(normalizer, row["quote"], control, row["pmc_id"])
            counts[result["status"]] += 1
            counts["reason:" + result["reason"]] += 1
            if result["status"] in {"mapped", "nil"}:
                counts["correct_accepted" if result["ids"] in row["gold_options"] else "incorrect_accepted"] += 1
            if result["reason"] == "unmatched":
                reason = precheck(row["quote"], policy)
                if reason:
                    guard_reasons[reason] += 1
                else:
                    eligible.add(row["quote"])
                    counts["semantic_eligible_occurrences"] += 1
        controls[control] = {"counts": dict(counts), "semantic_eligible_unique_quotes": len(eligible),
                             "precheck_abstention_reasons": dict(guard_reasons)}
    quotes = sorted({row["quote"] for row in cases})
    if not 0 < len(quotes) <= 1200:
        raise ValueError("Query budget exceeded")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    put(out / "training-cases.json", cases)
    put(out / "queries.json", [{"query_id": index, "quote": quote} for index, quote in enumerate(quotes)])
    summary = {"scope": "train32-only model-independent diagnostics; no held/dev gold, model load, fit or scores",
        "policy": ref(args.policy), "sources": {name: ref(Path(__file__).with_name(name)) for name in ["strict.py", "semantic.py", "diagnose.py"]},
        "train": ref(args.train), "targets": ref(args.targets), "coverage": ref(args.coverage),
        "documents": len(normalizer.training_ids), "target_rows": len(targets), "target_quote_occurrences": len(cases),
        "unique_quotes": len(quotes), "controls": controls,
        "max_quote_characters": max(map(len, quotes)), "max_quote_words": max(len(quote.split()) for quote in quotes),
        "queries": ref(out / "queries.json"), "cases": ref(out / "training-cases.json"),
        "leave_document_out_scope": "Excludes all supervised alias evidence from the current training document; public HPO unchanged. It is a train-only diagnostic, not independent held validation."}
    put(out / "cpu-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def evaluate(args):
    normalizer = StrictNormalizer(args.hpo, args.train)
    policy = load_policy(args.policy)
    prepared = json.loads(Path(args.prepared).read_text())
    if prepared["policy"]["sha256"] != sha256(args.policy):
        raise ValueError("Policy changed after model-independent preparation")
    if sha256(prepared["cases"]["path"]) != prepared["cases"]["sha256"]:
        raise ValueError("Training cases changed")
    cases = json.loads(Path(prepared["cases"]["path"]).read_text())
    scores = json.loads(Path(args.scores).read_text())
    expected = json.loads(Path(prepared["queries"]["path"]).read_text())
    if [(row["query_id"], row["quote"]) for row in scores] != [(row["query_id"], row["quote"]) for row in expected]:
        raise ValueError("Scored queries differ from frozen complete train-only query set")
    by_quote = {row["quote"]: row for row in scores}
    controls, mistakes = {}, []
    for control in ["strict_resubstitution", "public_only", "leave_document_out"]:
        counts, changed_unique = Counter(), set()
        for row in cases:
            strict = filtered_result(normalizer, row["quote"], control, row["pmc_id"])
            result = decide(strict, by_quote[row["quote"]], policy)
            counts[result["status"]] += 1
            counts["reason:" + result["reason"]] += 1
            accepted = result["status"] in {"mapped", "nil"}
            correct = result["ids"] in row["gold_options"]
            if accepted:
                counts["correct_accepted" if correct else "incorrect_accepted"] += 1
            if result["match"] == "public_semantic":
                counts["semantic_correct" if correct else "semantic_incorrect"] += 1
                changed_unique.add(row["quote"])
                if not correct:
                    mistakes.append({"control": control, "quote": row["quote"], "pmc_id": row["pmc_id"],
                        "gold_options": row["gold_options"], "prediction": result["ids"], "semantic": result["semantic"]})
        controls[control] = {"counts": dict(counts), "semantic_accepted_unique_quotes": len(changed_unique)}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    put(out / "train-only-mistakes.json", mistakes)
    summary = {"scope": "one fixed policy train-only diagnostic; not F1/held validation",
        "policy": ref(args.policy), "prepared": ref(args.prepared), "scores": ref(args.scores),
        "controls": controls, "unique_queries": len(scores), "threshold_search": False,
        "target_quote_occurrences": len(cases), "mistakes": ref(out / "train-only-mistakes.json")}
    put(out / "semantic-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate"])
    for name in ["hpo", "train", "policy", "output-dir"]:
        parser.add_argument("--" + name, required=True)
    for name in ["targets", "coverage", "prepared", "scores"]:
        parser.add_argument("--" + name)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
