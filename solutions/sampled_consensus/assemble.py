#!/usr/bin/env python3
"""Freeze the two preregistered F1 consensus predictions using CPU only.

Primary: B union (S1 intersection S2). Diagnostic: majority(G, S1, S2).
This module has no evaluator or label input. Concepts remain literal strings from
the original code_map, including its three non-HPO candidate strings.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


PINNED = {
    "baseline": "2a65056f8c47738b8a5d8f54a8e36bbc68e68e9a3657a237c56ccd9f3241b7d0",
    "prompts": "2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085",
    "row_audit": "35f9a00ecabf0b0114d756d31b13fc5a947155cb3c3ec8f6a36392f1f0674abb",
    "greedy_raw": "fbc6f5bee29591a4308325c5e19d65fb23d09d1950a328e416ec290fbbc8201f",
    "greedy_canonical": "8442a1258422a79a226d5867d3995253356a4fc02372de430da4d58c70b492a1",
    "parser_source": "062cbf90e04cec32e635975a2f903efa5fe7c1774fe568f2066b23dda26d824f",
}
SEEDS = (20261024, 20261025)
EOS_IDS = [248044, 248046]
THINK_IDS = [248068, 248069]
GENERATION_LIMIT = 1600
ENDPOINTS = ("primary-B-union-S1-intersect-S2", "diagnostic-M2of3-G-S1-S2")
POLICY = {
    "primary": "B union (S1 intersection S2)",
    "diagnostic": "(G intersection S1) union (G intersection S2) union (S1 intersection S2)",
    "invalid_response": "whole-row abstention; no JSON repair, missing-key fill, or greedy fallback",
    "alignment": "row_id; streams must each cover the same complete 80-row F1 universe",
    "entities": "preserve all 1189 B entities and every non-association byte",
    "baseline": "primary preserves all 471 B relations, including 88 outside the candidate pool",
    "seed_evaluation": "none; freeze exactly the two complete consensus endpoints together",
}
THINK_RE = re.compile(r"</?think(?:\s[^>]*)?>", re.IGNORECASE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def read_jsonl(path):
    data = Path(path).read_bytes()
    require(data.endswith(b"\n"), "Incomplete JSONL final line: " + str(path))
    lines = data.decode("utf-8").splitlines(keepends=True)
    require(lines and all(line.strip() for line in lines), "Empty JSONL row")
    records = [json.loads(line, object_pairs_hook=unique_object) for line in lines]
    require(all(type(row) is dict for row in records), "JSONL rows must be objects")
    return lines, records


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load_native_parser(path):
    """Execute the exact two original parser functions, without model imports."""
    require(sha(path) == PINNED["parser_source"], "Original native parser source SHA differs")
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"unique_object", "parse_complete"}]
    require([node.name for node in functions] == ["unique_object", "parse_complete"],
            "Original native parser function layout differs")
    namespace = {"json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_complete"]


def indexed(records, key="row_id"):
    result = {}
    for row in records:
        identity = row.get(key)
        require(type(identity) is str and identity and identity not in result,
                "Missing, duplicate, or nonstring identity: " + repr(identity))
        result[identity] = row
    return result


def relation_set(documents):
    result = set()
    for document in documents:
        pmc = document["pmc_id"]
        seen = set()
        for association in document["association"]:
            require(set(association) == {"patient_id", "phenotype"}, "Unexpected association schema")
            pid, concepts = association["patient_id"], association["phenotype"]
            require(type(pid) is str and pid not in seen, "Invalid/duplicate baseline patient")
            require(type(concepts) is list and all(type(h) is str for h in concepts)
                    and len(concepts) == len(set(concepts)), "Invalid/duplicate baseline concept")
            seen.add(pid)
            result.update((pmc, pid, concept) for concept in concepts)
    return result


def validate_universe(prompts, audit, baseline):
    by_prompt, by_audit = indexed(prompts), indexed(audit)
    by_document = indexed(baseline, "pmc_id")
    require(len(prompts) == len(audit) == 80 and len(baseline) == 16,
            "Expected complete F1: 80 rows and 16 documents")
    require(set(by_prompt) == set(by_audit), "Prompt/audit coverage differs")
    patients, concepts, universe = {}, set(), set()
    for index, row in enumerate(prompts):
        require(set(row) == {"row_id", "pmc_id", "messages", "code_map", "patient_ids"},
                "Unexpected pure-input prompt schema")
        expected = by_audit[row["row_id"]]
        require(expected["index"] == index and expected["pmc_id"] == row["pmc_id"]
                and expected["expanded_row_sha256"] == jsha(row), "Prompt identity/audit mismatch")
        pmc, code_map, pids = row["pmc_id"], row["code_map"], row["patient_ids"]
        require(pmc in by_document and type(code_map) is dict and code_map
                and list(code_map) == [f"C{i:03d}" for i in range(len(code_map))], "Invalid CID map")
        require(type(pids) is list and pids and all(type(pid) is str for pid in pids)
                and len(pids) == len(set(pids)), "Invalid supplied PID map")
        require(patients.setdefault(pmc, pids) == pids, "Inconsistent per-document PID order")
        require(set(pids) == {a["patient_id"] for a in by_document[pmc]["association"]},
                "Prompt/baseline patient coverage mismatch")
        for concept in code_map.values():
            require(type(concept) is str and (pmc, concept) not in concepts,
                    "Duplicate/nonliteral concept across rows")
            concepts.add((pmc, concept))
            universe.update((pmc, pid, concept) for pid in pids)
    base_relations = relation_set(baseline)
    require(set(patients) == set(by_document) and sum(map(len, patients.values())) == 39,
            "F1 document/patient universe differs")
    require(len(concepts) == 587 and len(universe) == 1557, "F1 candidate universe differs")
    require(sum(len(d["entities"]) for d in baseline) == 1189, "B entity count differs")
    require(len(base_relations) == 471 and len(base_relations - universe) == 88,
            "B relations or out-of-pool relations differ")
    return by_prompt, by_audit, universe, base_relations


def row_seed(seed, row_id):
    return int.from_bytes(hashlib.sha256(f"{seed}|{row_id}".encode()).digest()[:8], "big") & ((1 << 63) - 1)


def parse_row(raw, prompt, audit, parse_complete, think_token_ids, *, seed=None, ids=None):
    """Identity defects stop assembly; response defects abstain the entire row."""
    rid = prompt["row_id"]
    require(raw.get("row_id") == rid and raw.get("pmc_id") == prompt["pmc_id"],
            "Raw/prompt identity mismatch: " + rid)
    require(type(raw.get("response")) is str, "Missing raw response: " + rid)
    require(type(raw.get("input_tokens")) is int
            and raw["input_tokens"] == audit["expanded"]["tokens"], "Input length mismatch: " + rid)
    require(type(raw.get("removed_demonstrations")) is int and raw["removed_demonstrations"] == 0,
            "Prompt removal is forbidden: " + rid)
    evidence = raw if seed is not None else ids
    require(type(evidence) is dict and evidence.get("input_ids_sha256")
            == audit["expanded"]["input_ids_sha256"], "Input token identity mismatch: " + rid)
    if seed is not None:
        require(type(raw.get("seed")) is int and raw["seed"] == seed
                and type(raw.get("row_seed")) is int and raw["row_seed"] == row_seed(seed, rid),
                "Seed identity mismatch: " + rid)
        require(raw.get("fresh") is True, "Each sampled row must be fresh: " + rid)
        require(raw.get("rendered_sha256") == audit["expanded"]["rendered_sha256"],
                "Rendered input identity mismatch: " + rid)
        require(type(raw.get("generation_limit")) is int and raw["generation_limit"] == GENERATION_LIMIT,
                "Generation limit differs: " + rid)
        require(type(raw.get("raw_with_special_tokens")) is str,
                "Missing raw decode with special tokens: " + rid)
    else:
        require(evidence.get("row_id") == rid, "Greedy generated-ID row mismatch: " + rid)
    generated = evidence.get("generated_ids")
    require(type(generated) is list and all(type(token) is int and token >= 0 for token in generated),
            "Missing/invalid generated token evidence: " + rid)
    require(evidence.get("configured_eos_ids") == EOS_IDS, "EOS configuration differs: " + rid)
    require(type(raw.get("output_tokens")) is int and raw["output_tokens"] == len(generated),
            "Output length/evidence mismatch: " + rid)
    natural_eos = bool(generated and generated[-1] in EOS_IDS)
    hit_cap = len(generated) >= GENERATION_LIMIT and not natural_eos
    has_think = bool(set(generated) & set(think_token_ids) or THINK_RE.search(raw["response"])
                     or THINK_RE.search(raw.get("raw_with_special_tokens", "")))
    errors = []
    if not natural_eos or raw.get("natural_eos") is not True or evidence.get("natural_eos") is not True:
        errors.append("no_natural_eos")
    if hit_cap or raw.get("hit_token_limit") is not False or evidence.get("hit_token_limit") is not False:
        errors.append("token_cap_or_invalid_cap_flag")
    if len(generated) > GENERATION_LIMIT or any(token in EOS_IDS for token in generated[:-1]):
        errors.append("invalid_generation_boundary")
    if has_think or (seed is not None and (raw.get("thinking") is not False
                                         or raw.get("has_think_marker") is not False)):
        errors.append("thinking_violation")
    parsed, parser_error = None, None
    try:
        parsed = parse_complete(raw["response"], prompt)
    except (ValueError, TypeError) as exc:
        parser_error = type(exc).__name__ + ": " + str(exc)
        errors.append("strict_parser_failure")
    accepted = not errors
    # A syntactically valid fragment from an invalid row never receives a vote.
    mapping = parsed if accepted else None
    relations = {(prompt["pmc_id"], pid, prompt["code_map"][cid])
                 for cid, pids in (mapping or {}).items() for pid in pids}
    diagnostic = {
        "row_id": rid, "pmc_id": prompt["pmc_id"], "seed": seed,
        "row_seed": raw.get("row_seed"), "fresh": raw.get("fresh"),
        "strict_complete": accepted, "whole_row_abstention": not accepted,
        "reasons": errors, "parser_error": parser_error,
        "candidate_patients": mapping, "relations": sorted(relations),
        "candidate_count": len(prompt["code_map"]), "candidate_pair_count": len(prompt["code_map"]) * len(prompt["patient_ids"]),
        "input_tokens": raw["input_tokens"], "output_tokens": len(generated),
        "actual_natural_eos": natural_eos, "actual_hit_token_limit": hit_cap,
        "actual_has_think_marker": has_think, "input_ids_sha256": evidence["input_ids_sha256"],
        "generated_ids_sha256": jsha(generated), "response_sha256": hashlib.sha256(raw["response"].encode()).hexdigest(),
        "raw_record_sha256": jsha(raw),
    }
    return relations, diagnostic


def parse_stream(records, by_prompt, by_audit, parse_complete, think_token_ids, *, seed=None, ids_by_row=None):
    by_raw = indexed(records)
    require(set(by_raw) == set(by_prompt), "Incomplete/extra sampled row coverage")
    relations, diagnostics = set(), []
    for rid in sorted(by_prompt):
        pairs, diagnostic = parse_row(by_raw[rid], by_prompt[rid], by_audit[rid], parse_complete,
                                      think_token_ids, seed=seed,
                                      ids=None if ids_by_row is None else ids_by_row[rid])
        relations.update(pairs)
        diagnostics.append(diagnostic)
    return relations, diagnostics


def consensus(base, greedy, first, second):
    return base | (first & second), (greedy & first) | (greedy & second) | (first & second)


def association_span(line):
    """Locate the top-level association value, preserving all other UTF-8 bytes."""
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    position = len(line) - len(line.lstrip())
    require(line[position:position + 1] == "{", "Expected document JSON object")
    position += 1
    found, keys = None, set()
    while True:
        while position < len(line) and line[position].isspace():
            position += 1
        if line[position:position + 1] == "}":
            position += 1
            break
        key, position = decoder.raw_decode(line, position)
        require(type(key) is str and key not in keys, "Duplicate/nonstring document key")
        keys.add(key)
        while position < len(line) and line[position].isspace():
            position += 1
        require(line[position:position + 1] == ":", "Missing document colon")
        position += 1
        while position < len(line) and line[position].isspace():
            position += 1
        begin = position
        value, position = decoder.raw_decode(line, position)
        if key == "association":
            require(type(value) is list, "Invalid association value")
            found = (begin, position)
        while position < len(line) and line[position].isspace():
            position += 1
        if line[position:position + 1] == ",":
            position += 1
            continue
        require(line[position:position + 1] == "}", "Invalid document separator")
        position += 1
        break
    require(found is not None and not line[position:].strip(), "Missing association/trailing document data")
    return found


def assemble_prediction(baseline_lines, baseline, pairs):
    output = []
    for line, doc in zip(baseline_lines, baseline):
        associations = [{"patient_id": a["patient_id"],
                         "phenotype": sorted(h for pmc, pid, h in pairs
                                             if pmc == doc["pmc_id"] and pid == a["patient_id"])}
                        for a in doc["association"]]
        begin, end = association_span(line)
        candidate = line[:begin] + json.dumps(associations, ensure_ascii=False) + line[end:]
        output.append(candidate)
        new_begin, new_end = association_span(candidate)
        require(line[:begin] == candidate[:new_begin] and line[end:] == candidate[new_end:],
                "Non-association bytes changed")
    parsed = [json.loads(line) for line in output]
    require(relation_set(parsed) == pairs, "Prediction lost or invented relation identity")
    require(all(old["entities"] == new["entities"] for old, new in zip(baseline, parsed)),
            "Baseline entities changed")
    return "".join(output).encode("utf-8")


def save_bytes(path, data):
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save_json(path, value):
    save_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def run(args):
    refs = {}
    for name, expected in PINNED.items():
        path = getattr(args, name)
        refs[name] = reference(path)
        require(refs[name]["sha256"] == expected, "Pinned source differs: " + name)
    require(sorted(args.think_token_ids) == THINK_IDS,
            "Think token IDs differ from the pinned native tokenizer")
    baseline_lines, baseline = read_jsonl(args.baseline)
    _, prompts = read_jsonl(args.prompts)
    audit = read_json(args.row_audit)
    by_prompt, by_audit, universe, base = validate_universe(prompts, audit, baseline)
    parser = load_native_parser(args.parser_source)
    _, greedy_records = read_jsonl(args.greedy_raw)
    _, greedy_canonical = read_jsonl(args.greedy_canonical)
    canonical = indexed(greedy_canonical)
    require(set(canonical) == set(by_prompt), "Original greedy canonical coverage differs")
    ids_records, ids_refs = {}, []
    for row in prompts:
        index = by_audit[row["row_id"]]["index"]
        path = Path(args.greedy_ids_dir) / f"{index:03d}.json"
        ids_records[row["row_id"]] = read_json(path)
        ids_refs.append(reference(path))
    greedy, greedy_diagnostics = parse_stream(greedy_records, by_prompt, by_audit, parser,
                                             args.think_token_ids, ids_by_row=ids_records)
    for diagnostic in greedy_diagnostics:
        rid = diagnostic["row_id"]
        expected = canonical[rid]
        require(diagnostic["strict_complete"] and expected["strict_complete"] is True
                and expected["abstention"] is None
                and diagnostic["candidate_patients"] == expected["candidate_patients"],
                "Original native greedy strict parse/canonical drift: " + rid)
        provenance = "cached_probe" if by_audit[rid]["index"] == 19 else "fresh"
        require(expected["provenance"] == provenance, "Original greedy provenance differs")
        diagnostic["provenance"] = provenance
    streams, all_diagnostics, stream_stats = {}, [], {}
    for name, records, seed in [("G", greedy_records, None),
                                ("S1", read_jsonl(args.seed1_raw)[1], SEEDS[0]),
                                ("S2", read_jsonl(args.seed2_raw)[1], SEEDS[1])]:
        if seed is None:
            pairs, diagnostics = greedy, greedy_diagnostics
        else:
            pairs, diagnostics = parse_stream(records, by_prompt, by_audit, parser,
                                              args.think_token_ids, seed=seed)
        require(pairs <= universe, "Model relations escaped the fixed candidate/PID universe")
        streams[name] = pairs
        for diagnostic in diagnostics:
            diagnostic["stream"] = name
        all_diagnostics.extend(diagnostics)
        reasons = Counter(reason for item in diagnostics for reason in item["reasons"])
        stream_stats[name] = {"rows": len(diagnostics), "strict_complete_rows": sum(d["strict_complete"] for d in diagnostics),
                              "whole_row_abstentions": sum(d["whole_row_abstention"] for d in diagnostics),
                              "reason_counts": dict(reasons), "relations": len(pairs),
                              "fresh_rows": 79 if seed is None else 80, "cached_rows": 1 if seed is None else 0}
    refs["seed1_raw"] = reference(args.seed1_raw)
    refs["seed2_raw"] = reference(args.seed2_raw)
    primary, diagnostic = consensus(base, streams["G"], streams["S1"], streams["S2"])
    require(base <= primary and base - universe <= primary, "Primary lost baseline relations")
    require(primary - base <= universe and diagnostic <= universe, "Consensus escaped candidate universe")
    payloads = [assemble_prediction(baseline_lines, baseline, pairs) for pairs in (primary, diagnostic)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in zip(ENDPOINTS, payloads):
        save_bytes(output_dir / (name + ".pred.jsonl"), payload)
    save_bytes(output_dir / "row-diagnostics.jsonl",
               b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                        for row in all_diagnostics))
    save_json(output_dir / "greedy-generated-ids-manifest.json", ids_refs)
    endpoint_refs = {name: reference(output_dir / (name + ".pred.jsonl")) for name in ENDPOINTS}
    freeze = {
        "status": "BOTH_COMPLETE_PREDICTIONS_FROZEN_NO_QUALITY", "policy": POLICY,
        "source": reference(__file__), "references": refs,
        "greedy_generated_ids_manifest": reference(output_dir / "greedy-generated-ids-manifest.json"),
        "row_diagnostics": reference(output_dir / "row-diagnostics.jsonl"),
        "seeds": list(SEEDS), "row_seed_rule": "SHA256(f'{seed}|{row_id}') first 8 bytes big-endian & ((1<<63)-1)",
        "think_token_ids": sorted(args.think_token_ids), "eos_token_ids": EOS_IDS,
        "streams": stream_stats, "model_raw_rows": 240, "sampled_fresh_rows": 160,
        "documents": 16, "patients": 39, "entities_each": 1189,
        "candidate_concepts": 587, "candidate_patient_pairs": 1557,
        "baseline_relations": len(base), "baseline_outside_candidate_pool": len(base - universe),
        "baseline_retained_primary": len(base & primary),
        "baseline_outside_candidate_pool_retained_primary": len((base - universe) & primary),
        "primary_relations": len(primary), "primary_added_relations": len(primary - base),
        "diagnostic_relations": len(diagnostic), "endpoints": endpoint_refs,
        "nonrelation_bytes_preserved": True, "gold_read": False, "quality_calls": 0,
    }
    save_json(output_dir / "paired-predictions-freeze.json", freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PINNED:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--greedy-ids-dir", required=True)
    parser.add_argument("--seed1-raw", required=True)
    parser.add_argument("--seed2-raw", required=True)
    parser.add_argument("--think-token-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "endpoints": result["endpoints"],
                      "freeze": str(Path(args.output_dir).resolve() / "paired-predictions-freeze.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
