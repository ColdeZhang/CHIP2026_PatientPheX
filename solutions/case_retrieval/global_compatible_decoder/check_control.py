"""Small distinguishing contracts for exact objective, ties and raw preservation."""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

import census
import control


def main():
    start = time.monotonic()
    checks = []
    def solve(weights, edges, reference, order=None):
        return control.solve_component(list(range(len(weights))), edges, weights, set(reference),
                                       order or [(i, 1, 'HP:1', i) for i in range(len(weights))])[0]
    # A three-node path: higher single-item probability is not the best set.
    assert solve([Fraction(1,2), Fraction(3,10), Fraction(3,10)], [(0,1),(0,2)], [0]) == {1,2}
    checks.append('strict_global_gain_on_three_node_path')
    assert solve([Fraction(1,2), Fraction(1,4), Fraction(1,4)], [(0,1),(0,2)], [0]) == {0}
    assert solve([Fraction(1,2), Fraction(1,4), Fraction(1,4)], [(0,1),(0,2)], [1,2]) == {1,2}
    checks.append('exact_utility_tie_preserves_old_nested_intersection')
    assert solve([Fraction(1), Fraction(1)], [(0,1)], [], [(2,1,'HP:2',0),(1,2,'HP:1',1)]) == {1}
    checks.append('lexical_span_hpo_original_index_tie')
    low = .6
    high = math.nextafter(low, math.inf)
    assert solve([Fraction.from_float(low), Fraction.from_float(high)], [(0,1)], [0]) == {1}
    checks.append('one_float_ulp_is_not_erased_by_epsilon')
    assert solve([Fraction(1), Fraction(2)], [], []) == {0,1}
    checks.append('compatible_distinct_candidates_retained')
    line = ' { "text":"中文 \\"entities\\" { }", "association" : [{"patient_id":"P1","phenotype":["HP:1"]}], "entities" : [{"offset": 1}], "tail":{"entities":[99]} }\r\n'
    changed = control.replace_entities(line, [{'offset':2,'text':'中文'}])
    a,b = control.field_span(line)
    x,y = control.field_span(changed)
    assert line[:a] == changed[:x] and line[b:] == changed[y:] and changed.endswith('\r\n')
    assert control.replace_entities(line, json.loads(line)['entities']) == line
    checks.append('literal_top_level_only_unicode_escaped_key_crlf_preserved')
    try:
        control.field_span('{"entities":[],"entities":[]}')
        raise AssertionError('Duplicate keys not rejected')
    except AssertionError as exc:
        assert str(exc) != 'Duplicate keys not rejected'
    checks.append('duplicate_top_level_keys_rejected')
    result = dict(status='PASS', seconds=time.monotonic()-start, checks=checks,
                  sources=[census.ref(__file__),census.ref(control.__file__),census.ref(census.__file__)],
                  gpu_used=False, real_data_read=False)
    control.write(Path(sys.argv[1]), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
