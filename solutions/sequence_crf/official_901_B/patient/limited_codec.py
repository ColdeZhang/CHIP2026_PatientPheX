"""Decode explicit patient facts only; omissions are abstentions, never answers.

Pure CPU classification. The caller owns raw persistence and complete input-row
coverage. It must save each original record before invoking this function.
"""
import json
import math

from patient_first import validate_input

EOS_IDS = frozenset({248044, 248046})


class _Rejected(ValueError):
    pass


def _reject(reason):
    raise _Rejected(reason)


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            _reject('duplicate_json_key')
        output[key] = value
    return output


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _reject('nonfinite_json')
    return result


def classify_final(record, prompt):
    """Return complete/partial/abstain plus observed PIDs and explicit PID/CID pairs.

    Valid partial objects retain only their explicitly observed keys. An explicit
    [] is a negative statement for that observed patient; an omitted patient has
    no statement. Every invalid row returns *zero* retained keys/pairs, including
    apparently valid fields before an invalid field. Invalid prompt definitions
    raise: they are caller/data errors, not model abstentions.
    """
    validate_input(prompt)
    if 'answer' in prompt:
        raise ValueError('Inference input contains an answer')
    def abstain(reason):
        return dict(status='abstain', reason=reason,
                    observed_key_subset=[], selected_pairs=[])
    if not isinstance(record, dict):
        return abstain('record_not_object')
    if record.get('row_id') != prompt['row_id'] or str(record.get('pmc_id')) != str(prompt['pmc_id']):
        return abstain('response_identity_mismatch')
    tokens = record.get('generated_token_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        return abstain('invalid_generated_token_ids')
    if tokens[-1] not in EOS_IDS or any(token in EOS_IDS for token in tokens[:-1]):
        return abstain('missing_or_multiple_actual_EOS')
    if record.get('output_tokens') != len(tokens):
        return abstain('token_count_mismatch')
    if record.get('hit_token_limit') is not False:
        return abstain('token_limit_or_unknown_terminal_status')
    if record.get('removed_demonstrations') != 0:
        return abstain('prompt_removal')
    # Optional owner annotations never override original actual token evidence.
    if 'terminal_eos_id' in record and (type(record['terminal_eos_id']) is not int
                                      or record['terminal_eos_id'] != tokens[-1]):
        return abstain('terminal_eos_annotation_mismatch')
    if 'actual_last_token_eos' in record and record['actual_last_token_eos'] is not True:
        return abstain('terminal_eos_annotation_mismatch')
    text = record.get('response')
    if not isinstance(text, str):
        return abstain('response_not_text')
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: _reject('nonfinite_json'),
                           parse_float=_finite_float)
    except _Rejected as error:
        return abstain(str(error))
    except (ValueError, RecursionError, OverflowError):
        return abstain('invalid_complete_JSON')
    if not isinstance(value, dict):
        return abstain('JSON_not_object')
    patients = prompt['patient_ids']; codes = prompt['code_map']
    if not set(value) <= set(patients):
        return abstain('unknown_patient_key')
    for choices in value.values():
        if not isinstance(choices, list):
            return abstain('patient_value_not_list')
        if any(not isinstance(cid, str) for cid in choices):
            return abstain('CID_not_string')
        if len(choices) != len(set(choices)):
            return abstain('duplicate_CID')
        if not set(choices) <= set(codes):
            return abstain('unknown_CID')
    observed = [pid for pid in patients if pid in value]
    if not observed:
        return abstain('no_patient_keys')
    # Canonical ordering of *observed* facts only; no missing-key completion.
    selected = [[pid, cid] for pid in observed for cid in codes if cid in value[pid]]
    complete = len(observed) == len(patients)
    return dict(status='complete' if complete else 'partial',
                reason='all_patient_keys_observed' if complete else 'omitted_patients_abstain',
                observed_key_subset=observed, selected_pairs=selected)
