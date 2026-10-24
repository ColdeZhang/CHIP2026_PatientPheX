"""Byte-identical original fulltext-normal strict codec; no repair/fallback."""
import json

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result

def parse_complete(text, row):
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != set(row['code_map']):
        raise ValueError('Expected exactly the supplied candidate keys')
    allowed = set(row['patient_ids'])
    for patients in value.values():
        if not isinstance(patients, list):
            raise ValueError('Patient value must be a list')
        if any(type(pid) is not str or pid not in allowed for pid in patients):
            raise ValueError('Unknown patient ID or wrong type')
        if len(patients) != len(set(patients)):
            raise ValueError('Duplicate patient ID')
    return value
