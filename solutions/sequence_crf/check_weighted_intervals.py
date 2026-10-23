#!/usr/bin/env python3
"""Synthetic contracts and independent exhaustive scheduling oracle; no data labels."""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
from decode_weighted_intervals import decode, greedy, key


def candidate(start, length, probability, identifier='HP:0000001'):
    return dict(offset=start, length=length, selector_probability=probability,
                identifier=identifier, text='x' * length)


def exhaustive(rows, reference):
    eligible = [c for c in rows if c['selector_probability'] > .3]
    ref = {key(c) for c in reference}
    options = []
    for bits in itertools.product([False, True], repeat=len(eligible)):
        chosen = sorted((c for c, take in zip(eligible, bits) if take), key=key)
        if any(a['offset'] + a['length'] > b['offset'] for a, b in zip(chosen, chosen[1:])):
            continue
        utility = sum((Fraction(str(c['selector_probability'])) - Fraction(3, 10) for c in chosen), Fraction(0))
        keys = tuple(key(c) for c in chosen)
        options.append((-utility, -len(set(keys) & ref), keys))
    return min(options)[2]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    checks = []
    cases = {
        'global_beats_greedy': ([candidate(0, 10, .9), candidate(0, 5, .7), candidate(5, 5, .7)], [(0, 5), (5, 5)]),
        'empty': ([], []),
        'no_positive_gain': ([candidate(0, 2, .3), candidate(2, 2, .2)], []),
        'touching_allowed': ([candidate(0, 2, .6), candidate(2, 2, .6)], [(0, 2), (2, 2)]),
        'equal_utility_prefers_reference': ([candidate(0, 10, .9), candidate(0, 5, .6), candidate(5, 5, .6)], [(0, 10)]),
        'same_boundary_reference': ([candidate(0, 2, .6, 'HP:0000002'), candidate(0, 2, .6)], [(0, 2)]),
    }
    for name, (rows, expected) in cases.items():
        reference = greedy(rows)
        got, diagnostic = decode(rows, reference)
        assert [(e['offset'], e['length']) for e in got] == expected
        assert tuple(key(e) for e in got) == exhaustive(rows, reference)
        if name == 'same_boundary_reference': assert got[0]['identifier'] == 'HP:0000002'
        if name == 'global_beats_greedy': assert diagnostic['utility_gain'] == .2
        checks.append(name)
    rows = [candidate(0, 2, .6, 'HP:0000002'), candidate(0, 2, .6)]
    assert decode(rows, [])[0][0]['identifier'] == 'HP:0000001'
    checks.append('stable_id_tie_without_reference')
    rng = random.Random(20261023)
    for i in range(80):
        rows = [candidate(rng.randrange(12), rng.randrange(1, 8), rng.choice([.2, .3, .4, .5, .6, .7, .8]),
                          f'HP:{j+1:07}') for j in range(rng.randrange(9))]
        reference = greedy(rows)
        expected = exhaustive(rows, reference)
        actual = tuple(key(e) for e in decode(rows, reference)[0])
        assert actual == expected, i
        rng.shuffle(rows)
        assert tuple(key(e) for e in decode(rows, reference)[0]) == expected
    checks += ['80_independent_exhaustive_optimum_and_tie_cases', 'permutation_invariance_with_fixed_reference']
    for bad in [candidate(-1, 2, .6), candidate(0, 0, .6), candidate(0, 2, float('nan')), candidate(0, 2, 1.1)]:
        try: decode([bad], [])
        except AssertionError: pass
        else: raise AssertionError('invalid row accepted')
    duplicate = candidate(0, 2, .6)
    try: decode([duplicate, duplicate], [])
    except AssertionError: pass
    else: raise AssertionError('duplicate accepted')
    checks += ['invalid_coordinates_probability_rejected', 'duplicate_keys_rejected']
    result = dict(status='PASS', checks=checks, gold_read=False, new_fit=False, gpu_used=False)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
