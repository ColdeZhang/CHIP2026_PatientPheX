#!/usr/bin/env python3
"""Evaluate only after every declared exact-gate control prediction is frozen."""
import argparse
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import sys

import run as base
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import evaluate
import validate_predictions as validator


def difference(left, right):
    return {span: ids - right.get(span, set()) for span, ids in left.items()
            if ids - right.get(span, set())}


def changes(gold, before, after):
    count = Counter()
    for truth, old, new in zip(gold, before, after):
        assert truth['pmc_id'] == old['pmc_id'] == new['pmc_id']
        g = evaluate.positive_units(truth, truth, 'all')
        a = evaluate.positive_units(old, truth, 'all')
        b = evaluate.positive_units(new, truth, 'all')
        for role, values in [('added', difference(b, a)), ('removed', difference(a, b))]:
            tp, fp, _ = evaluate.mention_counts(g, values)
            count['entity_' + role + '_tp'] += tp
            count['entity_' + role + '_fp'] += fp
        ga, aa, ba = map(evaluate.associations, [truth, old, new])
        assert ga.keys() == aa.keys() == ba.keys()
        for patient in ga:
            for role, values in [('added', ba[patient] - aa[patient]),
                                 ('removed', aa[patient] - ba[patient])]:
                tp, fp, _ = evaluate.compare_sets(ga[patient], values)
                count['relation_' + role + '_tp'] += tp
                count['relation_' + role + '_fp'] += fp
    return dict(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    plan = json.loads((root / 'plan.json').read_text())
    freeze = json.loads((root / 'complete-freeze.json').read_text())
    assert freeze['plan_sha256'] == base.sha(root / 'plan.json')
    assert len(freeze['artifacts']) == 6
    for item in freeze['artifacts']:
        assert base.sha(item['path']) == item['sha256']
        assert base.sha(Path(item['path']).with_suffix('.manifest.json')) == item['manifest_sha256']
    for path, expected in plan['sources'].items():
        assert base.sha(path) == expected
    # No heldout labels are read before all six frozen artifacts are verified.
    parent = json.loads(Path(plan['baseline_manifest']['path']).read_text())
    assert base.sha(plan['baseline_manifest']['path']) == plan['baseline_manifest']['sha256']
    train_path = parent['configuration']['train']
    assert base.sha(train_path) == parent['files']['train']
    gold_all = base.read_jsonl(train_path)
    gold = {str(d['pmc_id']): d for d in gold_all}
    assert len(gold) == 48 and set(gold) == set(parent['training_document_ids'])
    hpo = validator.read_hpo(parent['configuration']['hpo'])
    combined = {'baseline': [], 'include-exact': []}
    results = []
    for fold in plan['folds']:
        directory = root / ('fold' + str(fold['fold']))
        before = base.read_jsonl(directory / 'baseline.jsonl')
        after = base.read_jsonl(directory / 'include-exact.jsonl')
        ids = [str(d['pmc_id']) for d in before]
        assert ids == [str(d['pmc_id']) for d in after]
        assert set(ids) == set(fold['input_ids']) and not set(ids) & set(fold['train_ids'])
        truth = [gold[i] for i in ids]
        item = {'fold': fold['fold'], 'arms': {}, 'changes': changes(truth, before, after)}
        for name, prediction in [('baseline', before), ('include-exact', after)]:
            report = evaluate.evaluate_report(truth, prediction, 'all', 'text')
            report['prediction_sha256'] = base.sha(directory / (name + '.jsonl'))
            report['training48_gold_source_sha256'] = base.sha(train_path)
            check = validator.validate_predictions(truth, prediction, hpo)
            assert check['valid'] and not check['error_count']
            (directory / (name + '.metrics.json')).write_text(json.dumps(report, indent=2) + '\n')
            (directory / (name + '.validation.json')).write_text(json.dumps(check, indent=2) + '\n')
            item['arms'][name] = {'metrics': report['metrics'], 'counts': report['counts'],
                                  'prediction_sha256': report['prediction_sha256'],
                                  'errors': check['error_count'], 'warnings': check['warning_count']}
            combined[name].extend(prediction)
        item['metric_deltas'] = {k: item['arms']['include-exact']['metrics'][k] - item['arms']['baseline']['metrics'][k]
                                 for k in item['arms']['baseline']['metrics']}
        results.append(item)
    pooled = {name: evaluate.evaluate_report(gold_all, rows, 'all', 'text')
              for name, rows in combined.items()}
    for name, report in pooled.items():
        (root / (name + '.pooled.metrics.json')).write_text(json.dumps(report, indent=2) + '\n')
    combined_by_id = {name: {str(d['pmc_id']): d for d in rows} for name, rows in combined.items()}
    result = {'scope': plan['scope'], 'selection_disclosure': plan['selection_disclosure'],
              'evaluated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': base.sha(__file__), 'evaluator_sha256': base.sha(evaluate.__file__),
              'validator_sha256': base.sha(validator.__file__), 'plan_sha256': base.sha(root / 'plan.json'),
              'all_predictions_freeze_sha256': base.sha(root / 'complete-freeze.json'),
              'frozen_at_utc': freeze['frozen_at_utc'], 'gold_sha256': base.sha(train_path), 'folds': results,
              'unweighted_fold_mean': {name: {k: sum(x['arms'][name]['metrics'][k] for x in results) / 3
                                             for k in pooled[name]['metrics']} for name in pooled},
              'pooled': {name: {'metrics': value['metrics'], 'counts': value['counts']} for name, value in pooled.items()},
              'pooled_changes': changes(gold_all, *[[combined_by_id[name][str(d['pmc_id'])] for d in gold_all]
                                                   for name in ['baseline', 'include-exact']]),
              'metric_deltas_pooled': {k: pooled['include-exact']['metrics'][k] - pooled['baseline']['metrics'][k]
                                       for k in pooled['baseline']['metrics']},
              'limitations': ['Same train48 used for aggregate mechanism selection; development control, not blind test.',
                              'No originaldev, GPU, new fitting or threshold/order search.',
                              'Added/removed entity TPFP is evaluated separately using v2 NIL matching; net pooled counts are authoritative.']}
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'fold_score_deltas': [x['metric_deltas']['Score'] for x in results],
                      'pooled_metrics': {k: v['metrics'] for k, v in pooled.items()},
                      'changes': result['pooled_changes']}))


if __name__ == '__main__':
    main()
