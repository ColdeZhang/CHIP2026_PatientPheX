#!/usr/bin/env python3
"""Exact weighted interval scheduling over frozen selector probabilities."""
from bisect import bisect_right
from fractions import Fraction
import math


def key(candidate):
    return candidate['offset'], candidate['length'], candidate['identifier']


def entity(candidate):
    return {k: candidate[k] for k in ['identifier', 'offset', 'length', 'text']} | {
        'type': 'Phenotype', 'note': None}


def greedy(candidates, cut=.3):
    """The frozen original decoder, with probability read from each row."""
    selected = []
    for c in sorted(candidates, key=lambda c: (-c['selector_probability'], -c['length'], c['offset'])):
        p = c['selector_probability']
        assert math.isfinite(float(p)) and 0 <= p <= 1
        if p < cut or any(c['offset'] < e['offset'] + e['length'] and
                          e['offset'] < c['offset'] + c['length'] for e in selected):
            continue
        selected.append(entity(c))
    return sorted(selected, key=lambda e: e['offset'])


def better(left, right):
    """State: exact utility, reference intersection count, sorted entity keys."""
    if left[:2] != right[:2]:
        return left if left[:2] > right[:2] else right
    return left if left[2] < right[2] else right


def decode(candidates, reference, cut=.3):
    assert cut == .3, 'This frozen control supports only the declared cut .3'
    threshold = Fraction(3, 10)
    lookup = {}
    for c in candidates:
        k = key(c)
        assert type(c['offset']) is int and c['offset'] >= 0
        assert type(c['length']) is int and c['length'] > 0
        assert isinstance(c['identifier'], str) and isinstance(c['text'], str)
        assert k not in lookup, 'Duplicate candidate keys are ambiguous'
        p = c['selector_probability']
        assert isinstance(p, (int, float)) and not isinstance(p, bool)
        assert math.isfinite(p) and 0 <= p <= 1
        lookup[k] = c
    refs = {key(e) for e in reference}
    assert len(refs) == len(reference) and refs <= lookup.keys()
    eligible = [(k, Fraction(str(c['selector_probability'])) - threshold)
                for k, c in lookup.items() if c['selector_probability'] > cut]
    eligible.sort(key=lambda item: (item[0][0] + item[0][1], *item[0]))
    ends = [k[0] + k[1] for k, _ in eligible]
    states = [(Fraction(0), 0, ())]
    for i, (k, utility) in enumerate(eligible):
        # Half-open coordinates: an interval ending at this start is compatible.
        predecessor_count = bisect_right(ends, k[0], 0, i)
        prev = states[predecessor_count]
        take = (prev[0] + utility, prev[1] + int(k in refs), tuple(sorted((*prev[2], k))))
        states.append(better(take, states[-1]))
    final = states[-1]
    selected = [entity(lookup[k]) for k in final[2]]
    assert all(a['offset'] + a['length'] <= b['offset'] for a, b in zip(selected, selected[1:]))
    reference_utility = sum((max(Fraction(0), Fraction(str(lookup[k]['selector_probability'])) - threshold)
                             for k in refs), Fraction(0))
    assert final[0] >= reference_utility
    return selected, dict(eligible=len(eligible), at_cut=sum(c['selector_probability'] == cut for c in candidates),
                         greedy_utility=str(reference_utility), wis_utility=str(final[0]),
                         utility_gain=float(final[0] - reference_utility), retained_greedy=final[1])
