"""Strict NEW-suffix-only contract for a proposed artificial-boundary run.

No JSON from the old thinking trace is parsed or repaired. The boundary is
operator-supplied; only the new suffix's terminal EOS can finish this new run.
"""
import json


def strict_final_text(text, codes, patients):
    def unique_pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None, "invalid_complete_final_JSON"
    if not isinstance(value, dict) or set(value) != set(codes):
        return None, "candidate_keys"
    if not all(isinstance(v, list) and all(isinstance(pid, str) and pid in patients for pid in v)
               and len(v) == len(set(v)) for v in value.values()):
        return None, "patient_values"
    return value, None


def extract_final(all_ids, replay_input_ids, tokenizer, codes, patients, max_new_tokens=512):
    """Accept actual generate output IDs, never a concatenated display string."""
    if all_ids[:len(replay_input_ids)] != replay_input_ids:
        return None, "replay_prefix_changed"
    suffix = all_ids[len(replay_input_ids):]
    if not suffix or len(suffix) > max_new_tokens:
        return None, "invalid_new_suffix_length"
    if suffix[-1] not in [248044, 248046]:
        return None, "missing_final_EOS_or_token_limit"
    if any(token in [248044, 248045, 248046, 248068, 248069] for token in suffix[:-1]):
        return None, "new_role_or_thinking_or_embedded_EOS"
    # Strip exactly the terminating EOS; skip_special_tokens=False prevents
    # swallowing any illegal hidden roles or multimodal tokens inside JSON.
    final = tokenizer.decode(suffix[:-1], skip_special_tokens=False)
    return strict_final_text(final, codes, patients)


def synthetic_contract_check(tokenizer):
    """Schema/phase boundary counterexamples only; no real patient assignments."""
    valid = '{"X0":[],"X1":["Z1"]}'
    cases = [
        ("valid_complete", valid, None),
        ("missing_CID", '{"X0":[]}', "candidate_keys"),
        ("extra_CID", '{"X0":[],"X1":[],"X2":[]}', "candidate_keys"),
        ("duplicate_CID", '{"X0":[],"X0":[],"X1":[]}', "invalid_complete_final_JSON"),
        ("illegal_PID", '{"X0":[],"X1":["Z2"]}', "patient_values"),
        ("duplicate_PID", '{"X0":[],"X1":["Z1","Z1"]}', "patient_values"),
        ("partial_JSON", '{"X0":[],"X1":[', "invalid_complete_final_JSON"),
        ("markdown_wrapper", '```json\n' + valid + '\n```', "invalid_complete_final_JSON"),
        ("extra_text", valid + '\nDone.', "invalid_complete_final_JSON"),
    ]
    names = []
    for name, text, reason in cases:
        _, actual = strict_final_text(text, {"X0", "X1"}, {"Z1"})
        assert actual == reason, (name, actual)
        names.append(name)
    # Deliberately invalid old-thought JSON is ignored, never used as fallback.
    prefix = tokenizer.encode('old thought {"X0":["Z2"]}\n</think>\n\n', add_special_tokens=False).ids
    final_ids = tokenizer.encode(valid, add_special_tokens=False).ids
    value, reason = extract_final(prefix + final_ids + [248046], prefix, tokenizer, {"X0", "X1"}, {"Z1"})
    assert value == {"X0": [], "X1": ["Z1"]} and reason is None
    names.append("old_thought_JSON_never_parsed")
    for name, ids, reason in [
        ("prefix_mutation", [0] + prefix[1:] + final_ids + [248046], "replay_prefix_changed"),
        ("no_EOS", prefix + final_ids, "missing_final_EOS_or_token_limit"),
        ("new_think", prefix + [248068] + final_ids + [248046], "new_role_or_thinking_or_embedded_EOS"),
        ("fake_close", prefix + [248069] + final_ids + [248046], "new_role_or_thinking_or_embedded_EOS"),
    ]:
        _, actual = extract_final(ids, prefix, tokenizer, {"X0", "X1"}, {"Z1"})
        assert actual == reason, (name, actual)
        names.append(name)
    return names
