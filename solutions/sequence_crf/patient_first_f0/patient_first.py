"""Closed-set patient-first serialization; no model, data lookup, or scoring."""
import copy
import json

OLD_TASK = "Determine which supplied patients actually have each candidate clinical phenotype in a biomedical report."
NEW_TASK = "Determine which candidate clinical phenotypes each supplied patient actually has in a biomedical report."
OLD_SCHEMA = "Return one JSON object mapping EVERY candidate code to a list of supplied patient IDs. Return [] when no supplied patient is supported. Use only listed candidate codes and patient IDs."
NEW_SCHEMA = "Return one JSON object mapping EVERY supplied patient ID to a list of candidate codes. Return [] when no candidate phenotype is supported for that patient. Use only listed patient IDs and candidate codes."
INPUT_KEYS = {"row_id", "pmc_id", "code_map", "patient_ids", "messages"}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON object key")
        out[key] = value
    return out


def parse_json(text):
    if not isinstance(text, str):
        raise ValueError("JSON text must be a string")
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def ids(values, kind, nonempty=False):
    if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(kind + " must be a list of nonempty strings")
    if len(values) != len(set(values)) or (nonempty and not values):
        raise ValueError(kind + " duplicate or empty")


def validate_input(row):
    if not isinstance(row, dict) or set(row) not in (INPUT_KEYS, INPUT_KEYS | {"answer"}):
        raise ValueError("unexpected row schema")
    ids(row["patient_ids"], "patient IDs", nonempty=True)
    if not isinstance(row["code_map"], dict):
        raise ValueError("code map must be an object")
    ids(list(row["code_map"]), "candidate IDs")
    if any(not isinstance(v, str) or not v for v in row["code_map"].values()):
        raise ValueError("candidate meaning IDs must be strings")
    if not isinstance(row["messages"], list) or len(row["messages"]) != 2:
        raise ValueError("expected system and user messages")
    for message, role in zip(row["messages"], ("system", "user")):
        if set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str):
            raise ValueError("unexpected message schema")


def validate_matrix(value, keys, values):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("missing or unknown matrix key")
    allowed = set(values)
    for choices in value.values():
        ids(choices, "matrix values")
        if not set(choices) <= allowed:
            raise ValueError("unknown matrix value")


def transpose_old_answer(text, codes, patients):
    ids(codes, "candidate IDs")
    ids(patients, "patient IDs", nonempty=True)
    old = parse_json(text)
    validate_matrix(old, codes, patients)
    return {p: [c for c in codes if p in old[c]] for p in patients}


def invert_patient_answer(text, codes, patients):
    """Strict whitelist parser, then canonical old-direction projection."""
    ids(codes, "candidate IDs")
    ids(patients, "patient IDs", nonempty=True)
    new = parse_json(text)
    validate_matrix(new, patients, codes)
    return {c: [p for p in patients if c in new[p]] for c in codes}


def replace_schema(system, inverse=False):
    lines = system.split("\n")
    expected = [NEW_TASK, NEW_SCHEMA] if inverse else [OLD_TASK, OLD_SCHEMA]
    replacement = [OLD_TASK, OLD_SCHEMA] if inverse else [NEW_TASK, NEW_SCHEMA]
    if lines[:2] != expected or len(lines) < 3:
        raise ValueError("source system schema mismatch")
    return "\n".join(replacement + lines[2:])


def transform(row):
    validate_input(row)
    result = copy.deepcopy(row)
    result["messages"][0]["content"] = replace_schema(row["messages"][0]["content"])
    if "answer" in row:
        result["answer"] = compact(transpose_old_answer(row["answer"], list(row["code_map"]), row["patient_ids"]))
    return result


def inverse(row):
    validate_input(row)
    result = copy.deepcopy(row)
    result["messages"][0]["content"] = replace_schema(row["messages"][0]["content"], inverse=True)
    if "answer" in row:
        result["answer"] = compact(invert_patient_answer(row["answer"], list(row["code_map"]), row["patient_ids"]))
    return result


def synthetic_contracts():
    """Discriminating whitelist/empty/ambiguity cases, not implementation mirroring."""
    codes, patients = ["C001", "C0010"], ["P1", "P10"]
    old = '{"C001":["P1","P10"],"C0010":["P10"]}'
    expected = {"P1": ["C001"], "P10": ["C001", "C0010"]}
    assert transpose_old_answer(old, codes, patients) == expected
    assert compact(invert_patient_answer(compact(expected), codes, patients)) == old
    assert transpose_old_answer('{"C001":[],"C0010":[]}', codes, patients) == {"P1": [], "P10": []}
    assert invert_patient_answer('{"P1":[],"P10":[]}', [], patients) == {}
    assert invert_patient_answer('{"P10":["C0010"],"P1":["C001"]}', codes, patients) == {"C001": ["P1"], "C0010": ["P10"]}
    valid = ["exact_P1_P10_C001_C0010", "complete_double_inverse", "all_empty_preserved", "empty_candidate_set", "JSON_key_order_semantically_irrelevant"]
    invalid = {
        "duplicate_patient_key": '{"P1":[],"P1":[],"P10":[]}',
        "missing_patient": '{"P1":[]}',
        "unknown_patient": '{"P1":[],"P10":[],"P2":[]}',
        "unknown_CID": '{"P1":["C002"],"P10":[]}',
        "duplicate_CID_value": '{"P1":["C001","C001"],"P10":[]}',
        "null_list": '{"P1":null,"P10":[]}',
        "numeric_value": '{"P1":[1],"P10":[]}',
        "bool_value": '{"P1":[true],"P10":[]}',
        "string_instead_of_list": '{"P1":"C001","P10":[]}',
        "nested_list": '{"P1":[["C001"]],"P10":[]}',
        "markdown": '```json\n{"P1":[],"P10":[]}\n```',
        "trailing_second_object": '{"P1":[],"P10":[]}{}',
        "nonfinite": '{"P1":[NaN],"P10":[]}',
        "old_schema_is_not_new_schema": '{"C001":["P1"],"C0010":[]}',
    }
    for name, text in invalid.items():
        try:
            invert_patient_answer(text, codes, patients)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid " + name)
    checks = [
        ("duplicate_patient_whitelist", lambda: invert_patient_answer('{"P1":[]}', codes, ["P1", "P1"])),
        ("duplicate_CID_whitelist", lambda: invert_patient_answer('{"P1":[],"P10":[]}', ["C001", "C001"], patients)),
        ("empty_patient_whitelist", lambda: invert_patient_answer('{}', codes, [])),
        ("old_duplicate_CID_key", lambda: transpose_old_answer('{"C001":[],"C001":[],"C0010":[]}', codes, patients)),
        ("old_unknown_patient", lambda: transpose_old_answer('{"C001":["P2"],"C0010":[]}', codes, patients)),
        ("old_duplicate_patient_value", lambda: transpose_old_answer('{"C001":["P1","P1"],"C0010":[]}', codes, patients)),
    ]
    for name, func in checks:
        try:
            func()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid " + name)
    return {"valid_cases": valid, "rejected_cases": list(invalid) + [x[0] for x in checks], "all_passed": True}
