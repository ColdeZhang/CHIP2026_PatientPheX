"""Critical half-open interval cases for the fixed addition policy."""
import argparse
import json
from common import write_json
from evaluate_control import compatible


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); a = ap.parse_args()
    def entity(start, end, hid='HP:0000001'):
        return {'offset': start, 'length': end-start, 'identifier': hid}
    existing = [entity(10, 20)]
    cases = [
        ('left_disjoint', entity(0, 9), True),
        ('touching_left', entity(0, 10), True),
        ('touching_right', entity(20, 25), True),
        ('same_span_same_id', entity(10, 20), False),
        ('same_span_different_id', entity(10, 20, 'HP:0000002'), False),
        ('proper_inner_new_id', entity(11, 19, 'HP:0000002'), True),
        ('proper_outer_new_id', entity(9, 21, 'HP:0000002'), True),
        ('nested_same_id', entity(11, 19), False),
        ('cross_left', entity(9, 19, 'HP:0000002'), False),
        ('cross_right', entity(11, 21, 'HP:0000002'), False),
    ]
    for name, candidate, expected in cases:
        assert compatible(candidate, existing) == expected, name
    assert not compatible(entity(9, 21, 'HP:0000002'), existing + [entity(20, 30)])
    result = {'status': 'PASS', 'cases': len(cases)+1, 'scope': 'Synthetic geometry only; no task gold or v2'}
    write_json(a.output, result); print(json.dumps(result))


if __name__ == '__main__': main()
