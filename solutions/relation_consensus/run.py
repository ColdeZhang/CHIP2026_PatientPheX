"""One frozen CPU control: strong baseline union (DocRE > 0 intersect schema)."""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify(ref):
    actual = identity(ref["path"])
    assert actual == ref, (ref["path"], actual)
    return Path(ref["path"])


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(plan_path, mode):
    started = time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    refs = plan["refs"]
    verify(refs["source"])
    evaluator_path = verify(refs["evaluator"])
    verify(refs["validator"])
    sys.path.insert(0, str(evaluator_path.parent))
    ev = importlib.import_module("evaluate")
    va = importlib.import_module("validate_predictions")
    output = Path(plan["output"])
    if mode == "assemble":
        output.mkdir(parents=True, exist_ok=False)
        write(output / "assembly-start.json", {"at_utc": now(), "plan": identity(plan_path)})
        rows = {k: va.load_jsonl(verify(refs[k])) for k in ("input", "baseline", "schema")}
        rows["docre_scores"] = [json.loads(line) for line in
                                verify(refs["docre_scores"]).read_text().splitlines() if line.strip()]
        baseline = {str(d["pmc_id"]): d for d in rows["baseline"]}
        schema = {str(d["pmc_id"]): d for d in rows["schema"]}
        inputs = {str(d["pmc_id"]): d for d in rows["input"]}
        assert len(baseline) == len(schema) == len(inputs) == 16
        assert baseline.keys() == schema.keys() == inputs.keys()
        def pairs(documents):
            return {(str(d["pmc_id"]), pid, value) for d in documents
                    for pid, values in ev.associations(d).items() for value in values}
        old_pairs = pairs(rows["baseline"])
        schema_pairs = pairs(rows["schema"])
        selected, seen = set(), set()
        for row in rows["docre_scores"]:
            key = tuple(row["key"])
            assert len(key) == 3 and key not in seen
            seen.add(key)
            assert key[0] in baseline and key[1] in ev.associations(baseline[key[0]])
            assert math.isfinite(row["logit"])
            # Exact original DocRE decision, including strict inequality at zero.
            if row["logit"] > 0:
                selected.add(key)
        assert len(seen) == 2874
        additions = (selected & schema_pairs) - old_pairs
        predicted = copy.deepcopy(rows["baseline"])
        for d in predicted:
            pmc = str(d["pmc_id"])
            assert ev.associations(d).keys() == ev.associations(schema[pmc]).keys()
            for a in d["association"]:
                a["phenotype"].extend(sorted(v for doc, pid, v in additions
                                             if doc == pmc and pid == a["patient_id"]))
            assert {k: v for k, v in d.items() if k != "association"} == {
                k: v for k, v in baseline[pmc].items() if k != "association"}
        assert pairs(predicted) == old_pairs | additions
        assert sum(len(d["association"]) for d in predicted) == 39
        assert sum(len(d["entities"]) for d in predicted) == 1189
        path = output / "prediction.jsonl"
        with path.open("x", encoding="utf-8") as f:
            for d in predicted:
                f.write(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        hpo = va.read_hpo(verify(refs["hpo"]))
        format_result = va.validate_predictions(rows["input"], predicted, hpo)
        write(output / "format.json", format_result)
        assert format_result["valid"]
        write(output / "freeze.json", {
            "at_utc": now(), "plan": identity(plan_path), "prediction": identity(path),
            "format": identity(output / "format.json"), "base_pairs": len(old_pairs),
            "docre_selected": len(selected), "schema_pairs": len(schema_pairs),
            "intersection": len(selected & schema_pairs), "added": len(additions),
            "added_pairs": sorted(additions), "removed": 0, "nonassociation_objects_exact": True,
            "gold_read": False, "NN_calls": 0, "fit_calls": 0,
            "seconds": time.monotonic() - started})
    else:
        frozen = json.loads((output / "freeze.json").read_text())
        assert frozen["plan"] == identity(plan_path)
        verify(frozen["prediction"])
        if frozen["added"] == 0:
            write(output / "result.json", {"status": "CLOSED_NO_NEW_SUPPLY", "new_v2": 0})
            return
        # Exclusive intent prevents accidental duplicate successful evaluation.
        write(output / "evaluation-intent.json", {"at_utc": now(), "freeze": identity(output / "freeze.json")})
        gold = va.load_jsonl(verify(refs["gold"]))
        predicted = va.load_jsonl(frozen["prediction"]["path"])
        metrics = ev.evaluate_report(gold, predicted, body_scope="all", no_id_doc="text")
        write(output / "metrics.json", metrics)
        old = json.loads(verify(refs["baseline_metrics"]).read_text())["metrics"]
        delta = {k: metrics["metrics"][k] - old[k] for k in old}
        def pairs(documents):
            return {(str(d["pmc_id"]), pid, value) for d in documents
                    for pid, values in ev.associations(d).items() for value in values}
        added = set(map(tuple, frozen["added_pairs"]))
        tp = len(added & pairs(gold))
        write(output / "result.json", {
            "status": "COMPLETE_FIXED_INTERSECTION_CONTROL", "at_utc": now(),
            "baseline": old, "metrics": metrics["metrics"], "delta": delta,
            "added": len(added), "added_TP": tp, "added_FP": len(added)-tp,
            "new_v2": 1, "old_v2": 0, "NN_calls": 0, "fit_calls": 0,
            "retain": delta["Score"] > 0, "not_blind": True, "original_dev_read": False,
            "freeze": identity(output / "freeze.json"), "seconds": time.monotonic()-started})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("mode", choices=("assemble", "evaluate"))
    a = p.parse_args()
    run(a.plan, a.mode)
