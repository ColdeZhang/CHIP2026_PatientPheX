"""One evaluation after all four wider-pool predictions have been frozen."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def changes(evaluate, gold, before, after):
    count = Counter()
    for g, a, b in zip(gold, before, after):
        gu, au, bu = [evaluate.positive_units(x, g, 'all') for x in [g, a, b]]
        for role, left, right in [('added', bu, au), ('removed', au, bu)]:
            diff = {span: ids-right.get(span, set()) for span, ids in left.items()
                    if ids-right.get(span, set())}
            tp, fp, _ = evaluate.mention_counts(gu, diff)
            count['entity_'+role+'_tp'] += tp
            count['entity_'+role+'_fp'] += fp
        gt, at, bt = [evaluate.associations(x) for x in [g, a, b]]
        assert gt.keys() == at.keys() == bt.keys()
        for patient in gt:
            for role, diff in [('added', bt[patient]-at[patient]), ('removed', at[patient]-bt[patient])]:
                tp, fp, _ = evaluate.compare_sets(gt[patient], diff)
                count['relation_'+role+'_tp'] += tp
                count['relation_'+role+'_fp'] += fp
    return dict(count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    root = Path(plan['output_dir'])
    assert not (root/'evaluation-start.json').exists()
    assert not (root/'results.json').exists()
    freeze = json.loads((root/'complete-freeze.json').read_text())
    assert freeze['plan_sha256'] == sha(a.plan)
    assert freeze['all_four_predictions_complete'] and not freeze['new_quality_evaluation_started']
    assert [x['fold'] for x in freeze['folds']] == [0, 1, 2, 3]
    for p, digest in plan['sources'].items():
        assert sha(p) == digest
    for fold in freeze['folds']:
        for role, item in fold.items():
            if role != 'fold':
                assert sha(item['path']) == item['sha256'], role
    for role in ['prepared_manifest', 'prepared_candidates']:
        assert sha(freeze[role]['path']) == freeze[role]['sha256']
    prior = plan['old095_pooled_metrics']
    assert sha(prior['path']) == prior['sha256']
    assert sha(plan['hpo']['path']) == plan['hpo']['sha256']
    old_pooled = json.loads(Path(prior['path']).read_text())
    assert old_pooled['metrics']['Score'] == .7226510219708908
    sys.path.insert(0, plan['evaluation_source'])
    import evaluate
    import validate_predictions as validator
    start = {'started_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
             'plan_sha256': sha(a.plan), 'complete_freeze_sha256': sha(root/'complete-freeze.json'),
             'all_four_predictions_verified_before_quality_labels': True}
    (root/'evaluation-start.json').write_text(json.dumps(start, indent=2)+'\n')
    # New endpoint labels are first opened only after all four complete hashes.
    gold_all = read(plan['train64_gold'])
    assert sha(plan['train64_gold']) == plan['train64_gold_sha256']
    truth = {d['pmc_id']: d for d in gold_all}
    assert len(truth) == len(gold_all) == 64
    hpo = validator.read_hpo(plan['hpo']['path'])
    reports, new_map, old_map = [], {}, {}
    for spec, frozen in zip(plan['folds'], freeze['folds']):
        after = read(frozen['new_prediction']['path'])
        before = read(frozen['old095_prediction']['path'])
        assert [d['pmc_id'] for d in after] == [d['pmc_id'] for d in before]
        gold = [truth[d['pmc_id']] for d in after]
        prior_metric = json.loads(Path(spec['old095_metrics']['path']).read_text())
        report = evaluate.evaluate_report(gold, after, 'all', 'text')
        valid = validator.validate_predictions(gold, after, hpo)
        assert valid['valid'] and valid['error_count'] == 0
        item = {'fold': spec['fold'], 'baseline095': prior_metric['metrics'], 'union': report['metrics'],
                'deltas': {k: report['metrics'][k]-prior_metric['metrics'][k] for k in report['metrics']},
                'changes': changes(evaluate, gold, before, after), 'errors': valid['error_count'],
                'warnings': valid['warning_count']}
        target = root/('fold'+str(spec['fold']))
        for name, value in [('metrics.json', report), ('validation.json', valid), ('results.json', item)]:
            (target/name).write_text(json.dumps(value, indent=2)+'\n')
        reports.append(item)
        assert not new_map.keys() & {d['pmc_id'] for d in after}
        new_map.update({d['pmc_id']: d for d in after})
        old_map.update({d['pmc_id']: d for d in before})
    assert new_map.keys() == old_map.keys() == truth.keys()
    after = [new_map[d['pmc_id']] for d in gold_all]
    before = [old_map[d['pmc_id']] for d in gold_all]
    pooled = evaluate.evaluate_report(gold_all, after, 'all', 'text')
    (root/'pooled.metrics.json').write_text(json.dumps(pooled, indent=2)+'\n')
    (root/'pooled.prediction.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in after))
    result = {'finished_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(a.plan),
              'source_sha256': sha(__file__), 'freeze_sha256': sha(root/'complete-freeze.json'),
              'evaluation_started': start, 'folds': reports,
              'pooled': {'baseline095': old_pooled['metrics'], 'union': pooled['metrics'],
                         'deltas': {k: pooled['metrics'][k]-old_pooled['metrics'][k] for k in pooled['metrics']},
                         'changes': changes(evaluate, gold_all, before, after)},
              'unweighted_mean': {arm: {k: sum(f[arm][k] for f in reports)/4 for k in pooled['metrics']}
                                  for arm in ['baseline095', 'union']},
              'scope': 'Four development folds; new wider candidate AND provenance increment, not a simple single threshold change.',
              'original_dev_read': False, 'official_actions': False,
              'selection': 'One prespecified union and fixed HGB160/.4; no quality-guided threshold/feature search.'}
    (root/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'fold_deltas': [x['deltas']['Score'] for x in reports], 'pooled': result['pooled']}))


if __name__ == '__main__':
    main()
