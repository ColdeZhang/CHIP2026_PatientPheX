#!/usr/bin/env python3
"""Frozen fold0 paired HGB control; preparation/production never reads heldout gold.

`prepare` verifies sources and constructs only the common numeric matrices.
`produce` additionally requires complete, manifest-verified official-base responses,
fits both fixed trees, then freezes both predictions. Evaluation is deliberately
outside this entry point. Paths are configurable; the approved bytes are pinned.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import ast
import copy
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits


PINS = {
    "contract": "e61c1bbb2bda98647cb66243c862874a7a2361b6b22fcdd2d7db77620a1b353e",
    "train_labels": "eb133c3107c1aada2e156892ba430e5b5860315c058d6c7dd5387fe45b065f77",
    "train_prompts": "23807d0ce5fff74d923898f2ae4a0790eaaa1137354e4ed5fc7fc27b4fb5513a",
    "input_prompts": "d387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0",
    "training_features": "071ca2f2f0fd3abc6c7d216f2470c83a0d7e614c1ca35cc9b8880cf074507997",
    "input_features": "e094919aa6b23235804205f4a954573e8160972c80e2734b2d44ab4c0792e21f",
    "input": "ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37",
    "entities": "3e400ed82c5f2913e159116ec9e1ff1623ad559a14e69bc65889f9d0c2f6d25a",
    "input_responses": "42223211a18eadaaf967ea03adf684c2cd38f9e87a6987a5d6ab08532b01cd2b",
    "parser_source": "817fdc4494e7ed349085fe97c5744409d66d0f1de476089ac8c5696a84a6b4a1",
    "model_source": "74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7",
}
RUNNER_SHA = "33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669"
LLM_COLUMNS = ("llm_selected", "llm_empty", "llm_parse_failed", "llm_selected_count")
TREE_PARAMETERS = dict(categorical_features="from_dtype", class_weight=None,
    early_stopping=False, interaction_cst=None, l2_regularization=5.0,
    learning_rate=0.06, loss="log_loss", max_bins=255, max_depth=None,
    max_features=1.0, max_iter=160, max_leaf_nodes=15, min_samples_leaf=20,
    monotonic_cst=None, n_iter_no_change=10, random_state=20261023,
    scoring="loss", tol=1e-7, validation_fraction=0.1, verbose=0, warm_start=False)
THRESHOLD = 0.5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parse_json(text):
    text=text.strip()
    if '</think>' in text:
        text=text.rsplit('</think>',1)[1].strip()
    elif text.startswith('<think>'):
        return None
    text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    try:return json.loads(text)
    except json.JSONDecodeError:
        for opener,closer in [('[',']'),('{','}')]:
            a=text.find(opener);b=text.rfind(closer)
            if 0<=a<b:
                try:return json.loads(text[a:b+1])
                except json.JSONDecodeError:pass
        return None


def verify_parser(path):
    """Full source pin plus AST equality prevents accidental parser tightening."""
    require(sha(path) == PINS["parser_source"], "Frozen parser source SHA mismatch")
    functions = [n for n in ast.parse(Path(path).read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "parse_json"]
    ours = ast.parse(inspect.getsource(parse_json)).body[0]
    require(len(functions) == 1 and ast.dump(functions[0]) == ast.dump(ours),
            "Parser no longer matches frozen parse_json")


def keyed_requests(rows):
    result = {}
    for row in rows:
        rid = row["row_id"]
        require(isinstance(rid, str) and rid not in result, "Duplicate/invalid request row_id")
        require(set(row) == {"row_id", "pmc_id", "code_map", "patient_ids", "messages"},
                "Request field whitelist violated (answer/target forbidden)")
        require([m["role"] for m in row["messages"]] == ["system", "user"],
                "Only unchanged system/user messages allowed")
        require(all(isinstance(m["content"], str) for m in row["messages"]), "Invalid message content")
        patients = row["patient_ids"]
        require(patients and all(isinstance(x, str) for x in patients)
                and len(set(patients)) == len(patients), "Invalid patient_ids")
        require(isinstance(row["code_map"], dict) and row["code_map"]
                and all(isinstance(k, str) and isinstance(v, str)
                        for k, v in row["code_map"].items()), "Invalid code_map")
        result[rid] = row
    return result


def pair_rows(prompts, labelled=None):
    """Prompt order -> CID order -> supplied-patient order; no response-order join."""
    keyed_requests(prompts)
    labels = {} if labelled is None else {r["row_id"]: r for r in labelled}
    if labelled is not None:
        require(len(labels) == len(labelled) and set(labels) == {r["row_id"] for r in prompts},
                "Training label/request row coverage mismatch")
    pairs, seen = [], set()
    for prompt in prompts:
        answer = None
        if labelled is not None:
            source = labels[prompt["row_id"]]
            require({k: v for k, v in source.items() if k != "answer"} == prompt,
                    "Training requests changed beyond answer removal")
            answer = json.loads(source["answer"])
            require(isinstance(answer, dict) and set(answer) == set(prompt["code_map"]),
                    "Training answer CID mismatch")
            require(all(isinstance(v, list) and all(p in prompt["patient_ids"] for p in v)
                        for v in answer.values()), "Training answer patient mismatch")
        for cid, concept in prompt["code_map"].items():
            for patient in prompt["patient_ids"]:
                key = (str(prompt["pmc_id"]), patient, concept)
                require(key not in seen, "Duplicate permanent patient-concept key")
                seen.add(key)
                row = dict(row_id=prompt["row_id"], pmc_id=key[0], patient_id=patient,
                           concept=concept, CID=cid)
                if answer is not None:
                    row["target"] = int(patient in answer[cid])
                pairs.append(row)
    return pairs


def join_features(pairs, cached_rows, training):
    by_key = {}
    for row in cached_rows:
        key = (str(row["pmc_id"]), str(row["patient_id"]), row["concept"])
        require(key not in by_key, "Duplicate cached patient-concept key")
        require(("target" in row) == training, "Validation target present / training target missing")
        feats = row["features"]
        require(isinstance(feats, dict) and not set(feats).intersection(LLM_COLUMNS), "Invalid base features")
        require(all(isinstance(k, str) and isinstance(v, (bool, int, float))
                    and math.isfinite(float(v)) for k, v in feats.items()), "Nonfinite/non-numeric feature")
        by_key[key] = row
    selected = []
    for pair in pairs:
        key = (pair["pmc_id"], pair["patient_id"], pair["concept"])
        require(key in by_key, "Missing exact cached feature key: " + repr(key))
        row = by_key[key]
        if training:
            require(row["target"] in (False, True, 0, 1)
                    and int(row["target"]) == pair["target"], "Training label mismatch")
        selected.append(row["features"])
    return selected


def align_responses(prompts, responses):
    expected = keyed_requests(prompts)
    found = {}
    for response in responses:
        rid = response["row_id"]
        require(rid not in found and rid in expected, "Duplicate/unknown response row_id")
        require(str(response["pmc_id"]) == str(expected[rid]["pmc_id"]), "Response pmc_id mismatch")
        require(isinstance(response["response"], str), "Response text missing")
        found[rid] = response
    require(set(found) == set(expected), "Require complete response rows; no missing-row fallback")
    return found


def llm_features(prompts, responses, pairs):
    by_id = align_responses(prompts, responses)
    values, statistics = {}, {"valid_candidate_decisions": 0, "unresolved_candidate_decisions": 0}
    for prompt in prompts:
        response = by_id[prompt["row_id"]]
        parsed = None if response.get("hit_token_limit", False) else parse_json(response["response"])
        patients = set(prompt["patient_ids"])
        for code in prompt["code_map"]:
            value = parsed.get(code) if isinstance(parsed, dict) else None
            # Exactly original merger policy, including numeric patient -> str.
            valid = isinstance(value, list) and not any(str(p) not in patients for p in value)
            values[(prompt["row_id"], code)] = ({str(p) for p in value} if valid else None)
            statistics["valid_candidate_decisions" if valid else "unresolved_candidate_decisions"] += 1
    matrix = []
    for row in pairs:
        selected = values[(row["row_id"], row["CID"])]
        matrix.append([int(selected is not None and row["patient_id"] in selected),
                       int(selected is not None and not selected), int(selected is None),
                       len(selected) if selected is not None else 0])
    return np.asarray(matrix, dtype=np.float64), statistics


def verify_response_manifest(manifest, responses_path, prompts_path, rows, source):
    fp = manifest["fingerprint"]
    config = fp["configuration"]
    require("adapter" in config and config["adapter"] is None and not fp.get("adapter_files"),
            "Only official base responses accepted; adapter forbidden")
    require(fp["source_sha256"] == RUNNER_SHA, "Runner source SHA mismatch")
    require(fp["prompts_sha256"] == sha(prompts_path), "Response prompt fingerprint mismatch")
    require(fp["source_manifest_sha256"] == PINS["model_source"], "Model source manifest mismatch")
    require(manifest["output_sha256"] == sha(responses_path), "Response bytes mismatch")
    require(manifest["expected_rows"] == len(rows), "Response expected_rows mismatch")
    require(manifest["base_parameters"] == manifest["total_parameters"] == 8953803264,
            "Wrong base/adapter parameter count")
    require(config["max_input_tokens"] == 14000 and config["max_new_tokens"] == 1600
            and fp["batch_size"] == 1 and fp["thinking"] is False
            and fp["precision"] == "text_only_bfloat16_sdpa_no_quantization",
            "Inference protocol differs from fixed base58")
    require(manifest["eos_token_ids"] == [248044, 248046], "EOS policy mismatch")
    require(all(manifest["load_information"].get(k) == []
                for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Base loading was incomplete")
    expected_files = {x["name"]: {"bytes": x["size"], "sha256": x["sha256"]} for x in source["files"]}
    require(len(expected_files) == 15 and fp["model_files"] == expected_files, "Actual model file SHA mismatch")
    for name, version in {"transformers": "5.20.2", "peft": "0.21.2", "tokenizers": "0.23.1"}.items():
        require(fp["runtime"][name] == version, "Runtime mismatch: " + name)
    for row in rows:
        require(row.get("removed_demonstrations") == 0, "Prompt demonstration removed")
        require(isinstance(row.get("hit_token_limit"), bool), "Truncation status missing")
        require(0 < row["input_tokens"] <= 14000 and 0 <= row["output_tokens"] <= 1600,
                "Invalid token accounting")


def base_matrices(training, inference, expected_names=None):
    vectorizer = DictVectorizer(sparse=False)
    train_x = vectorizer.fit_transform(training)
    input_x = vectorizer.transform(inference)
    if expected_names is not None:
        require(vectorizer.feature_names_ == expected_names, "Frozen train-only vocabulary mismatch")
    require(np.isfinite(train_x).all() and np.isfinite(input_x).all(), "Nonfinite matrix")
    return vectorizer, train_x, input_x


def append_llm(base, flags):
    require(flags.shape == (len(base), 4), "LLM matrix shape mismatch")
    combined = np.concatenate([base, flags], axis=1)
    require(np.array_equal(combined[:, :base.shape[1]], base), "Base matrix differs between arms")
    return combined


def matrix_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def construct_predictions(inputs, entities, pairs, probabilities):
    require(len(pairs) == len(probabilities) and np.isfinite(probabilities).all(), "Invalid probabilities")
    base = {str(d["pmc_id"]): d for d in entities}
    require(len(base) == len(entities) == len(inputs), "Document coverage mismatch")
    selected = {}
    for doc in inputs:
        did = str(doc["pmc_id"])
        require("entities" not in doc and "association" not in doc, "Annotated validation input forbidden")
        require(did in base and did not in selected, "Input identity mismatch")
        selected[did] = {str(p["patient_id"]): set() for p in doc["patient"]}
        require(len(selected[did]) == len(doc["patient"]), "Duplicate given patient")
    for pair, probability in zip(pairs, probabilities):
        require(pair["pmc_id"] in selected and pair["patient_id"] in selected[pair["pmc_id"]],
                "Unknown prediction identity")
        if probability >= THRESHOLD:
            selected[pair["pmc_id"]][pair["patient_id"]].add(pair["concept"])
    outputs = []
    for doc in inputs:
        did = str(doc["pmc_id"])
        row = copy.deepcopy(base[did])
        row["association"] = [{"patient_id": p["patient_id"],
                               "phenotype": sorted(selected[did][str(p["patient_id"])])}
                              for p in doc["patient"]]
        require(row["entities"] == base[did]["entities"], "Entity pool changed")
        outputs.append(row)
    return outputs


def freeze_predictions(out, predictions):
    require(set(predictions) == {"without_llm", "with_llm"}, "Both arms required before freeze")
    files = {}
    for arm in ("without_llm", "with_llm"):
        path = out / (arm + ".predictions.jsonl")
        require(not path.exists(), "Refusing to overwrite predictions")
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in predictions[arm]))
        files[arm] = {"path": str(path), "sha256": sha(path)}
    freeze = {"both_predictions_complete": True, "heldout_gold_read": False,
              "evaluation_performed": False, "source_sha256": sha(__file__), "files": files}
    write_json(out / "predictions-freeze.json", freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "produce"])
    for key in PINS:
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    for key in ("input_response_manifest", "train_responses", "train_response_manifest"):
        parser.add_argument("--" + key.replace("_", "-"), required=key.startswith("input"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    require(not out.exists(), "Use a new unique output directory")
    hashes = {}
    for key, expected in PINS.items():
        path = getattr(args, key)
        hashes[key] = {"path": path, "sha256": sha(path)}
        require(hashes[key]["sha256"] == expected, "Frozen bytes mismatch: " + key)
    verify_parser(args.parser_source)
    contract = json.loads(Path(args.contract).read_text())
    require(contract["estimator"]["estimator"] == TREE_PARAMETERS
            and contract["fit_controls"]["threshold"] == THRESHOLD, "Tree protocol changed")
    train_prompts, input_prompts = read_rows(args.train_prompts), read_rows(args.input_prompts)
    train_pairs = pair_rows(train_prompts, read_rows(args.train_labels))
    input_pairs = pair_rows(input_prompts)
    train_docs = {r["pmc_id"] for r in train_pairs}; input_docs = {r["pmc_id"] for r in input_pairs}
    require(len(train_prompts) == 204 and len(input_prompts) == 58
            and len(train_pairs) == 4127 and len(input_pairs) == 1166
            and len(train_docs) == 48 and len(input_docs) == 16 and not train_docs & input_docs,
            "Frozen split/candidate counts differ")
    y = np.asarray([r["target"] for r in train_pairs], dtype=np.int64)
    require(y.sum() == 967, "Training positive count changed")
    training = join_features(train_pairs, read_rows(args.training_features), True)
    inference = join_features(input_pairs, read_rows(args.input_features), False)
    vectorizer, x, z = base_matrices(training, inference, contract["base_features"]["training_columns"])
    require(x.shape == (4127, 299) and z.shape == (1166, 299), "Common matrix shape mismatch")
    source = json.loads(Path(args.model_source).read_text())
    input_responses = read_rows(args.input_responses)
    input_manifest = json.loads(Path(args.input_response_manifest).read_text())
    verify_response_manifest(input_manifest, args.input_responses, args.input_prompts, input_responses, source)
    input_flags, input_stats = llm_features(input_prompts, input_responses, input_pairs)
    inputs, entities = read_rows(args.input), read_rows(args.entities)
    by_doc = {str(d["pmc_id"]): d for d in inputs}
    require(set(by_doc) == input_docs, "Stripped input document mismatch")
    require(all("entities" not in d and "association" not in d for d in inputs), "Validation labels forbidden")
    require(sum(len(d["entities"]) for d in entities) == 947, "Entity count mismatch")
    for prompt in input_prompts:
        require(set(prompt["patient_ids"]) == {str(p["patient_id"]) for p in by_doc[str(prompt["pmc_id"])]["patient"]},
                "Prompt/given patient mismatch")
    metadata = dict(phase=args.phase, source_sha256=sha(__file__), files=hashes,
        command=vars(args), runtime=dict(numpy=np.__version__, sklearn=sklearn.__version__),
        input_response_manifest_sha256=sha(args.input_response_manifest),
        train_rows=len(train_pairs), input_rows=len(input_pairs), positives=int(y.sum()),
        train_features=vectorizer.feature_names_, train_matrix_sha256=matrix_sha(x),
        input_matrix_sha256=matrix_sha(z), target_array_sha256=matrix_sha(y),
        pair_order_sha256=hashlib.sha256(json.dumps([train_pairs, input_pairs], sort_keys=True).encode()).hexdigest(),
        validation_llm_statistics=input_stats, estimator=TREE_PARAMETERS, threshold=THRESHOLD,
        heldout_gold_read=False, original_dev_gold_read=False, tree_fitted=False,
        wider_occurrence_evidence_shared=True, evaluation_performed=False)
    if args.phase == "prepare":
        out.mkdir(parents=True)
        write_json(out / "preparation.json", metadata)
        print(json.dumps({"prepared": True, "tree_fitted": False, "output": str(out)}))
        return
    require(args.train_responses and args.train_response_manifest, "Complete base204 responses required")
    responses = read_rows(args.train_responses)
    train_manifest = json.loads(Path(args.train_response_manifest).read_text())
    verify_response_manifest(train_manifest, args.train_responses, args.train_prompts, responses, source)
    train_flags, train_stats = llm_features(train_prompts, responses, train_pairs)
    augmented_x, augmented_z = append_llm(x, train_flags), append_llm(z, input_flags)
    metadata.update(train_response_manifest_sha256=sha(args.train_response_manifest),
        train_responses_sha256=sha(args.train_responses), training_llm_statistics=train_stats,
        augmented_train_matrix_sha256=matrix_sha(augmented_x), augmented_input_matrix_sha256=matrix_sha(augmented_z),
        llm_columns=list(LLM_COLUMNS), base_prefix_identical=True)
    predictions, models, times = {}, {}, {}
    with threadpool_limits(limits=1):
        for arm, xx, zz in [("without_llm", x, z), ("with_llm", augmented_x, augmented_z)]:
            started = time.monotonic()
            model = HistGradientBoostingClassifier(**TREE_PARAMETERS).fit(xx, y)
            probabilities = model.predict_proba(zz)[:, 1]
            predictions[arm] = construct_predictions(inputs, entities, input_pairs, probabilities)
            models[arm] = model
            times[arm] = time.monotonic() - started
    out.mkdir(parents=True)
    freeze = freeze_predictions(out, predictions)
    # Only local trusted output is serialized; this entry point never loads pickles.
    import joblib
    joblib.dump(dict(models=models, vectorizer=vectorizer, llm_columns=LLM_COLUMNS,
                     threshold=THRESHOLD), out / "models.joblib")
    metadata.update(tree_fitted=True, fit_and_prediction_seconds=times,
        models_sha256=sha(out / "models.joblib"), predictions_freeze_sha256=sha(out / "predictions-freeze.json"),
        predictions=freeze["files"])
    write_json(out / "production.json", metadata)
    print(json.dumps({"produced": True, "evaluated": False, "output": str(out)}))


if __name__ == "__main__":
    main()
