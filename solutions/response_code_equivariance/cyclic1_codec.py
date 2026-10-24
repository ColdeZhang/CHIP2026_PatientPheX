"""One fixed output-handle rotation; never rename a clinical identity or CID.

For patient index i, its new handle is PT((i + 1) % n + 1).
Example: old PT001->p0/PT002->p1/PT003->p2 becomes
PT001->p2/PT002->p0/PT003->p1. n=1 is the exact original representation.
"""
import copy
import json
import original_response_codec as original

VIEW = 'patient-response-code-cyclic1-v1'


def expected_mapping(row):
    patients = row['patient_ids']
    if not patients or any(type(x) is not str for x in patients) or len(set(patients)) != len(patients):
        raise ValueError('Require unique original clinical patient IDs')
    codes = [f'PT{i:03d}' for i in range(1, len(patients) + 1)]
    if set(codes) & (set(patients) | set(row['code_map'])):
        raise ValueError('Response code collision with clinical PID or CID')
    # Keep table/code order; rotate assignments, not the original patient list.
    return {code: patients[(i - 1) % len(patients)] for i, code in enumerate(codes)}


def table_text(mapping):
    return original.TABLE_PREFIX + json.dumps(
        [{'response_code': code, 'original_patient_id': pid} for code, pid in mapping.items()],
        ensure_ascii=False, separators=(',', ':'))


def validate_view(row):
    expected = expected_mapping(row)
    if ('answer' in row or row['response_representation'] != original.REPRESENTATION
            or list(row['response_code_map'].items()) != list(expected.items())
            or row['response_patient_codes'] != list(expected)):
        raise ValueError('Require exact cyclic1 held-input identity map')
    if not row['messages'][-1]['content'].endswith(table_text(expected)):
        raise ValueError('Cyclic1 table and decoder mapping differ')
    return expected


def transform(row):
    if 'answer' in row:
        raise ValueError('Only fixed answer-free held80 is in this control')
    if [x['role'] for x in row['messages']] != ['system', 'user']:
        raise ValueError('Unexpected original coded message schema')
    old = dict(zip(row['response_patient_codes'], row['patient_ids']))
    codes = [f'PT{i:03d}' for i in range(1, len(row['patient_ids']) + 1)]
    if row['response_patient_codes'] != codes or list(row['response_code_map'].items()) != list(old.items()):
        raise ValueError('Input must be the original unshifted coded view')
    suffix = table_text(old)
    user = row['messages'][-1]['content']
    if not user.endswith(suffix):
        raise ValueError('Original coded suffix differs; no heuristic splitting')
    prefix = user[:-len(suffix)]
    if any(code in prefix for code in codes):
        raise ValueError('Response alias collides with original evidence')
    shifted = expected_mapping(row)
    result = copy.deepcopy(row)
    if len(row['patient_ids']) > 1:
        result['response_code_map'] = shifted
        result['messages'][-1]['content'] = prefix + table_text(shifted)
    validate_view(result)
    return result


def decode_response(text, row):
    mapping = validate_view(row)
    # Exact existing strict JSON grammar: complete CID keys, unique known PT
    # values, no partial/set decoder, fuzzy correction, sorting or supplements.
    matrix = original.parse_matrix(text, row['code_map'], mapping)
    return {cid: [mapping[code] for code in values] for cid, values in matrix.items()}
