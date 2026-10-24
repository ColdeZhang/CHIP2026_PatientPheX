"""Original strict normal/fulltext codec, selected functions unchanged."""
import json

def unique_object(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError('Duplicate JSON key')
        obj[key] = value
    return obj


def parse_complete(text, row):
    # Exact original normal/fulltext strict policy, not the later R1 codec.
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
