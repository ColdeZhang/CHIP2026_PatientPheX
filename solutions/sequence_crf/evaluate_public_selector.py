#!/usr/bin/env python3
"""One fixed outer0 assessment after the common baseline/new prediction freeze."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import evaluate_exact_gate_control as common
base, evaluate, validator = common.base, common.evaluate, common.validator


def breakdown(gold, before, after, scores):
    groups = defaultdict(Counter)
    for truth, old, new in zip(gold, before, after):
        did = truth['pmc_id']
        g = evaluate.positive_units(truth, truth, 'all')
        old_units = evaluate.positive_units(old, truth, 'all')
        new_units = evaluate.positive_units(new, truth, 'all')
        added, removed = common.difference(new_units, old_units), common.difference(old_units, new_units)
        cs = {(c['offset'],c['length'],c['identifier']): c for c in scores[did]}
        # Assign selected new atomic units to the exact selected candidate's
        # origin. Existing-pool reselection is distinct from truly novel keys.
        partition = defaultdict(lambda: defaultdict(set))
        for e in new['entities']:
            span = (e['offset'],e['length'])
            ids = set(e['identifier'].split(';')) & added.get(span,set())
            if not ids:
                continue
            c = cs[(e['offset'],e['length'],e['identifier'])]
            role = 'added_from_existing_pool' if c['crf'] or c['semantic_source'] else 'added_from_public_only_pool'
            partition[role][span].update(ids)
        for span, ids in removed.items():
            a,b = span[0],span[0]+span[1]
            overlap = [e for e in new['entities'] if a < e['offset']+e['length'] and e['offset'] < b]
            if any((e['offset'],e['length']) == span for e in overlap):
                role = 'removed_same_boundary_different_id'
            elif overlap:
                role = 'removed_overlapping_boundary_replacement'
            else:
                role = 'removed_without_new_overlap'
            partition[role][span].update(ids)
        for role, units in partition.items():
            tp,fp,_ = evaluate.mention_counts(g,units)
            groups[role].update(tp=tp,fp=fp,atomic_units=sum(map(len,units.values())))
    return {k:dict(v) for k,v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    a = p.parse_args()
    root = Path(a.directory)
    plan = json.loads((root/'plan.json').read_text())
    freeze = json.loads((root/'complete-freeze.json').read_text())
    assert freeze['plan_sha256'] == base.sha(root/'plan.json')
    assert freeze['evaluation_gold_read'] is False
    assert set(freeze['predictions']) == {'baseline','augmented'}
    for info in freeze['predictions'].values():
        assert base.sha(info['path']) == info['sha256']
    for info in freeze['support'].values():
        assert base.sha(info['path']) == info['sha256']
    for path,digest in plan['sources'].items():
        assert base.sha(path) == digest
    baseline_path = freeze['predictions']['baseline']['path']
    new_path = freeze['predictions']['augmented']['path']
    assert base.sha(baseline_path) == plan['baseline_prediction']['sha256']
    assert base.sha(plan['baseline_metrics']['path']) == plan['baseline_metrics']['sha256']
    prior_report = json.loads(Path(plan['baseline_metrics']['path']).read_text())
    assert prior_report['metrics']['Score'] == .7103590603023253
    before,after = base.read_jsonl(baseline_path),base.read_jsonl(new_path)
    # First evaluation-label read occurs only after all frozen source/output
    # contracts above pass. Training labels were restricted to outer train48.
    gold_all = base.read_jsonl(plan['gold'])
    gold_map = {d['pmc_id']:d for d in gold_all}
    assert len(gold_map) == 16 and [d['pmc_id'] for d in before] == [d['pmc_id'] for d in after]
    gold = [gold_map[d['pmc_id']] for d in before]
    report = evaluate.evaluate_report(gold,after,'all','text')
    check = validator.validate_predictions(gold,after,validator.read_hpo(plan['hpo']))
    assert check['valid'] and not check['error_count']
    scores = {d['pmc_id']:d['candidates'] for d in base.read_jsonl(freeze['support']['scores']['path'])}
    result = dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.sha(__file__),evaluator_sha256=base.sha(evaluate.__file__),
        validator_sha256=base.sha(validator.__file__),
        plan_sha256=base.sha(root/'plan.json'),freeze_sha256=base.sha(root/'complete-freeze.json'),
        gold_sha256=base.sha(plan['gold']),baseline_metrics_reused=True,
        baseline=prior_report['metrics'],augmented=report['metrics'],
        deltas={k:report['metrics'][k]-prior_report['metrics'][k] for k in report['metrics']},
        changes=common.changes(gold,before,after),entity_change_origins=breakdown(gold,before,after,scores),
        errors=check['error_count'],warnings=check['warning_count'],
        limitations=['New public candidate source plus six corresponding features: method increment, not one feature causal effect.',
            'Outer0 has participated in earlier development; originaldev remains untouched.',
            'Public-only missing Sap values use explicit old CRF-only placeholders, not measured Sap confidence.',
            'Change-group TP/FP uses v2 NIL matching separately; complete report counts govern total performance.',
            'Old entities can be removed/replaced; no source/threshold/ordering sweep.'])
    for name,value in [('augmented.metrics.json',report),('augmented.validation.json',check),('results.json',result)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
