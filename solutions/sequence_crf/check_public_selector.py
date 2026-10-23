#!/usr/bin/env python3
"""Synthetic boundary contracts for the new public-source pool adapter."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
import select_public_candidates as method


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    old = [dict(offset=0, length=4, identifier='HP:0000001', text='pain',
                probability=.8, similarity=.99, exact=True, is_base=True,
                crf=True, crf_span=True, semantic_source=True),
           dict(offset=10, length=3, identifier='HP:0000002', text='red',
                probability=.7, similarity=.95, exact=False, is_base=False,
                crf=False, crf_span=False, semantic_source=True)]
    snapshot = copy.deepcopy(old)
    public = [dict(offset=0, length=4, identifier='HP:0000001', text='pain',
                   public_present=True, public_score=.96, public_neural=True,
                   public_dictionary=False, public_abbreviation=False, public_origin_unknown=False),
              dict(offset=0, length=4, identifier='HP:0000003', text='pain',
                   public_present=True, public_score=.97, public_neural=False,
                   public_dictionary=True, public_abbreviation=False, public_origin_unknown=False),
              dict(offset=20, length=4, identifier='HP:0000004', text='blue',
                   public_present=True, public_score=.99, public_neural=False,
                   public_dictionary=False, public_abbreviation=True, public_origin_unknown=False)]
    public[0].update(public_raw_hpo_ids=['HP:0000001'],public_matched_neural_head_columns=[2])
    duplicate = dict(public[0], public_score=.98, public_dictionary=True,
                     public_raw_hpo_ids=['HP:0000007'],public_matched_neural_head_columns=[7])
    pool = method.merge_pool(old, public+[duplicate])
    assert old == snapshot
    assert [method.key(c) for c in pool[:2]] == [method.key(c) for c in old]
    assert all({k: c[k] for k in o} == o for c, o in zip(pool, old))
    assert len(pool) == 4 and pool[0]['public_score'] == .98
    assert pool[0]['public_neural'] and pool[0]['public_dictionary']
    assert pool[0]['public_raw_hpo_ids'] == ['HP:0000001','HP:0000007']
    assert pool[0]['public_matched_neural_head_columns'] == [2,7]
    assert pool[2]['crf_span'] and not pool[2]['crf']
    assert not pool[2]['semantic_source'] and pool[2]['probability'] == .5 and pool[2]['similarity'] == 0
    assert pool[3]['public_abbreviation'] and not pool[3]['crf_span']
    chosen = method.select(pool, [.3, .4, .9, .39])
    assert [x['identifier'] for x in chosen] == ['HP:0000003', 'HP:0000002']
    # A conflicting new ID can replace old, .4 is inclusive, other old source
    # records are not protected. An overlapping lower score cannot coexist.
    chosen = method.select(pool, [.8, .4, .9, .5])
    assert [x['identifier'] for x in chosen] == ['HP:0000003','HP:0000002','HP:0000004']
    crossing = dict(pool[2], offset=2, length=10, text='in      re')
    assert len(method.select([pool[0], crossing], [.9, .8])) == 1
    failures = []
    for label, candidates in [('nonfinite', [dict(public[0], public_score=math.nan)]),
                              ('wrong_surface', [dict(public[0], text='ache')])]:
        try:
            method.merge_pool(old, candidates)
        except AssertionError:
            failures.append(label)
        else:
            raise AssertionError(label)
    doc = {'pmc_id': 'x', 'full_text': [{'offset': 30, 'text': 'pain'}]}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'synthetic.jsonl'
        row = dict(public[0], offset=30)
        path.write_text(json.dumps({'pmc_id':'x','candidates':[dict(row,identifier='HP:0000001')]})+'\n')
        assert method.public_rows(path, {'x': doc})['x'][0]['offset'] == 30
        path.write_text(json.dumps({'pmc_id':'x','candidates':[dict(row,identifier='HP:0000001',offset=31)]})+'\n')
        try:
            method.public_rows(path, {'x': doc})
        except AssertionError:
            failures.append('global_offset')
        else:
            raise AssertionError('bad offset accepted')
    result = dict(status='PASS', source_sha256=method.sha(__file__),
        selector_sha256=method.sha(method.__file__), synthetic_only=True,
        checked=['old source fields/order immutable','exact duplicate score max and source union',
                 'same-boundary different ID retained as alternatives','explicit missing Sap/CRF scores',
                 'whole-pool deletion/replacement','.4 inclusive and greedy overlap exclusion',
                 'global passage offset and exact surface',
                 'old-head IDs/columns retained as non-feature provenance'], rejected=failures,
        models_loaded=False, task_annotations_read=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
