"""CPU-only input preparation; importing this module does not run preparation.

The only extracted executable upstream code is is_number/tokenize and the
unchanged annotate_text method (plus its exact pre-get_match enumeration).
No upstream module import, TensorFlow, fastText, model, labels, or fit is used.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ast_sha(node):
    return sha256(ast.dump(node, include_attributes=False).encode("utf-8"))


def _native_parts(source_text):
    tree = ast.parse(source_text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NCR"]
    require(len(classes) == 1, "Expected exactly one NCR class")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef)
               and n.name == "annotate_text"]
    require(len(methods) == 1, "Expected exactly one NCR.annotate_text")
    native = methods[0]
    stops = [i for i, n in enumerate(native.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "matches" for t in n.targets)]
    require(len(stops) == 1, "Expected one matches assignment at the NN boundary")
    stop = stops[0]
    expected_call = ast.parse("matches = [x[0] for x in self.get_match(candidates, 1)]").body[0]
    require(ast.dump(native.body[stop], include_attributes=False) ==
            ast.dump(expected_call, include_attributes=False), "Native NN boundary changed")
    body = copy.deepcopy(native.body[:stop])
    candidate = copy.deepcopy(native)
    candidate.name = "enumerate_native"
    candidate.body = body + [ast.Return(value=ast.Tuple(elts=[
        ast.Name(id="candidates", ctx=ast.Load()),
        ast.Name(id="candidates_info", ctx=ast.Load())], ctx=ast.Load()))]
    plain = []
    for name in ("is_number", "tokenize"):
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(found) == 1, "Expected exactly one native " + name)
        plain.append(copy.deepcopy(found[0]))
    return native, body, candidate, plain


def extract_native(source_text):
    """Return (enumerate_native, tokenize, complete_native_annotate_text).

    Both methods are unbound: call enum(None, text), or annotate(adapter, text,
    threshold=0.8). The latter requires adapter.ont.root_id and adapter.get_match;
    it is returned intact and is never called by this preparation program.
    """
    native, body, candidate, plain = _native_parts(source_text)
    module = ast.fix_missing_locations(ast.Module(
        body=plain + [candidate, copy.deepcopy(native)], type_ignores=[]))
    scope = {"re": re}
    exec(compile(module, "<exact native NCR methods>", "exec"), scope)
    require(ast_sha(ast.Module(body=candidate.body[:-1], type_ignores=[])) ==
            ast_sha(ast.Module(body=native.body[:len(body)], type_ignores=[])),
            "Enumeration prefix AST changed")
    return scope["enumerate_native"], scope["tokenize"], scope["annotate_text"]


def file_ref(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def load_ref(ref):
    require(isinstance(ref, dict) and all(k in ref for k in ("path", "bytes", "sha256")),
            "File reference must contain path, bytes, sha256")
    data = Path(ref["path"]).read_bytes()
    require(len(data) == ref["bytes"] and sha256(data) == ref["sha256"],
            "File reference mismatch: " + str(ref["path"]))
    return data


def publish_bytes(path, data):
    """Exclusive publication; never replace a previous run's output."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(str(temporary), str(path))
    temporary.unlink()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def check_numpy():
    # Enforce the same explicit packed layout used by the downstream cache.
    import numpy as np
    require(np.__version__ == "1.23.5", "Expected numpy 1.23.5, got " + np.__version__)
    dtype = np.dtype([("column", "<u2"), ("score", "<f4")], align=False)
    offsets = {name: dtype.fields[name][1] for name in dtype.names}
    require(dtype.itemsize == 6 and offsets == {"column": 0, "score": 2}
            and not dtype.isalignedstruct, "Packed top1 dtype contract failed")
    values = np.zeros(2, dtype=dtype)
    values["column"] = [0, 14260]
    values["score"] = [0.8, 1.0]
    expected = struct.pack("<HfHf", 0, 0.8, 14260, 1.0)
    require(values.tobytes() == expected, "Packed dtype byte order mismatch")
    require(np.array_equal(np.frombuffer(expected, dtype=dtype), values),
            "Packed dtype roundtrip mismatch")
    return {"version": np.__version__, "dtype_descr": dtype.descr,
            "dtype_itemsize": dtype.itemsize, "dtype_offsets": offsets,
            "packed_roundtrip": True, "synthetic_rows": 2}


def run(plan_path, output_dir):
    started = time.monotonic()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "preparation.json"
    # Existing receipts are immutable. Report refusal on stderr rather than
    # overwriting the evidence of a successful or failed previous invocation.
    require(not receipt_path.exists() and not receipt_path.with_suffix(".json.tmp").exists(),
            "Existing preparation receipt; use a fresh output directory")
    report = {"status": "failed_input_only_preparation", "stage": "initialization",
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "artifacts": {}, "counts": {}, "pmc_ids": [], "refs": {},
              "competition_labels_read": False, "task_NN_run": False,
              "model_loaded": False, "fit_run": False, "quality_evaluation": False}
    success = False
    try:
        report["refs"]["source"] = file_ref(__file__)
        report["refs"]["plan"] = file_ref(plan_path)
        for name in ("queries.jsonl", "columns.json"):
            require(not (output_dir / name).exists() and
                    not (output_dir / (name + ".tmp")).exists(),
                    "Existing output; use a fresh directory: " + name)
        require(not any(name == "tensorflow" or name.startswith("tensorflow.")
                        or name == "fasttext" or name.startswith("fasttext.")
                        for name in sys.modules), "Model runtime already imported")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        report["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
        report["stage"] = "plan_and_numpy"
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if "source_sha256" in plan:
            require(report["refs"]["source"]["sha256"] == plan["source_sha256"],
                    "Preparation source does not match plan")
        frozen_keys = {"rows": 408272, "bytes": 16445155,
                       "sha256": "1924f3691da766a81de963690d9a72bbd4f030ff34c3e191c082c64bc6ff7b96"}
        require(all(plan["expected_keys"].get(k) == v for k, v in frozen_keys.items()),
                "Plan changed frozen token-key contract")
        expected = plan["expected_counts"]
        expected_documents = expected.get("documents", expected.get("docs"))
        require(expected_documents == 64 and expected.get("docs", 64) == 64 and
                expected["passages"] == 2301 and expected["candidate_occurrences"] == 683438,
                "Plan changed frozen input census counts")
        report["numpy"] = check_numpy()
        report["stage"] = "verified_sources_and_ontology"
        source_text = load_ref(plan["native_source"]).decode("utf-8")
        input_bytes = load_ref(plan["input"])
        ontology = json.loads(load_ref(plan["ontology"]))
        report["refs"].update({key: plan[key] for key in ("input", "native_source", "ontology")})
        enumerate_native, tokenize, _annotate_text = extract_native(source_text)
        native, body, _candidate, plain = _native_parts(source_text)
        report["ast_sha256"] = {
            "enumeration_body": ast_sha(ast.Module(body=body, type_ignores=[])),
            "tokenize": ast_sha(plain[1]), "annotate_text": ast_sha(native)}
        require(report["ast_sha256"]["enumeration_body"] ==
                "44a7eff872d5bf5d98466a887064c4533fbe57d1052c9ee5f35eda4d5caba0b9",
                "Native enumeration differs from frozen census AST")
        concepts = ontology["concepts"]
        require(isinstance(concepts, list) and len(concepts) == 14260 and
                all(isinstance(label, str) and re.fullmatch(r"HP:[0-9]{7}", label)
                    for label in concepts) and len(set(concepts)) == len(concepts),
                "Expected original ordered 14260 unique ontology concepts")
        root_id = ontology["root_id"]
        require(root_id in concepts and "None" not in concepts, "Invalid original ontology root/head")
        labels = concepts + ["None"]
        report["head"] = {"concept_count": len(concepts), "label_count": len(labels),
                          "root_id": root_id,
                          "order_sha256": sha256("\n".join(labels).encode("utf-8"))}
        norm = re.compile(r"[\\/\r\n\t-]")
        contracts = [("Seizures.", [(0, 8)]),
                     ("A B C", [(0, 1), (0, 3), (0, 5), (2, 3), (2, 5), (4, 5)]),
                     ("ab-cd", [(0, 2), (0, 5), (3, 5)]),
                     ("aa  bb", [(0, 2), (4, 6)]),
                     ("Caf\u00e9 x", [(0, 3), (5, 6)]), ("", [])]
        for text, spans in contracts:
            queries, info = enumerate_native(None, text)
            require([(a, b) for a, b, _ in info] == spans and len(queries) == len(info),
                    "Synthetic native coordinate contract failed")
            require(all(norm.sub(" ", text[a:b]) == q for q, (a, b, _) in zip(queries, info)),
                    "Synthetic native normalization contract failed")
        report["synthetic_coordinate_contracts_passed"] = len(contracts)
        report["stage"] = "input_only_native_enumeration"
        docs = [json.loads(line) for line in input_bytes.splitlines()]
        require(len(docs) == expected_documents and
                all(isinstance(doc, dict) and set(doc) ==
                    {"pmc_id", "pmid", "patient", "full_text"} for doc in docs),
                "Expected 64 four-field input-only documents")
        ids = [doc["pmc_id"] for doc in docs]
        require(len(set(ids)) == expected_documents, "Duplicate document IDs")
        report["pmc_ids"] = ids
        counts = {"documents": len(docs), "passages": 0, "candidate_occurrences": 0,
                  "unique_token_keys": 0}
        report["counts"] = counts
        unique = set()
        enum_started = time.monotonic()
        for doc_index, doc in enumerate(docs):
            require(isinstance(doc["full_text"], list), "full_text must be a list")
            for passage_index, passage in enumerate(doc["full_text"]):
                context = "document %s passage %s" % (doc_index, passage_index)
                require(isinstance(passage, dict) and
                        set(passage) <= {"section_type", "type", "offset", "text", "title"}
                        and type(passage.get("offset")) is int and
                        isinstance(passage.get("text"), str), "Invalid input passage: " + context)
                text = passage["text"]
                queries, info = enumerate_native(None, text)
                require(len(queries) == len(info) and
                        len({(a, b) for a, b, _ in info}) == len(info),
                        "Candidate cardinality or coordinate duplication: " + context)
                counts["passages"] += 1
                counts["candidate_occurrences"] += len(queries)
                for query, (start, end, _chunk) in zip(queries, info):
                    require(0 <= start < end <= len(text) and
                            norm.sub(" ", text[start:end]) == query,
                            "Original local character offset mismatch: " + context)
                    tokens = tokenize(query)[:49]
                    require(1 <= len(tokens) <= 7, "Unexpected native token length: " + context)
                    key = " ".join(tokens)
                    require(tokenize(key) == tokens,
                            "Native token-key roundtrip failed (INT/FLOAT included): " + context)
                    unique.add(key)
        counts["unique_token_keys"] = len(unique)
        require(counts["passages"] == expected["passages"] and
                counts["candidate_occurrences"] == expected["candidate_occurrences"] and
                len(unique) == frozen_keys["rows"], "Full native census counts changed")
        report["enumeration_seconds"] = time.monotonic() - enum_started
        report["stage"] = "publishing_verified_key_and_head_files"
        query_path = output_dir / "queries.jsonl"
        temporary = output_dir / "queries.jsonl.tmp"
        digest = hashlib.sha256()
        byte_count = 0
        with temporary.open("xb") as handle:
            for key in sorted(unique):
                # Check the emitted representatives, in addition to every occurrence.
                require(" ".join(tokenize(key)) == key, "Emitted representative is not native-stable")
                line = json_bytes({"query": key})
                handle.write(line)
                digest.update(line)
                byte_count += len(line)
            handle.flush()
            os.fsync(handle.fileno())
        require(byte_count == frozen_keys["bytes"] and digest.hexdigest() == frozen_keys["sha256"],
                "Regenerated queries differ from exact frozen census bytes")
        os.link(str(temporary), str(query_path))
        temporary.unlink()
        report["artifacts"]["queries"] = file_ref(query_path)
        columns_path = output_dir / "columns.json"
        publish_bytes(columns_path, json_bytes({"labels": labels, "root_id": root_id}))
        report["artifacts"]["columns"] = file_ref(columns_path)
        report["all_candidate_offsets_roundtrip"] = True
        report["all_occurrence_and_unique_token_keys_roundtrip"] = True
        report["exact_frozen_query_bytes"] = True
        report["native_annotate_text_extracted_but_not_called"] = True
        require(not any(name == "tensorflow" or name.startswith("tensorflow.")
                        or name == "fasttext" or name.startswith("fasttext.")
                        for name in sys.modules), "Unexpected model runtime imported")
        report["status"] = "complete_input_only_preparation"
        report["stage"] = "complete"
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["timing"] = {"total_seconds": time.monotonic() - started,
                            "enumeration_seconds": report.pop("enumeration_seconds", None)}
        publish_bytes(receipt_path, (json.dumps(report, ensure_ascii=False, indent=2,
                                              allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": report["status"], "stage": report["stage"],
                      "preparation": file_ref(receipt_path), "counts": report["counts"]}))
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return run(args.plan, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
