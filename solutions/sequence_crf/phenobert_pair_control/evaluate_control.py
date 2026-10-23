#!/usr/bin/env python3
"""One new-arm v2 comparison after four public-pair predictions freeze; reuse old reports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(item):
    path = Path(item['path'])
    assert path.is_absolute(), str(path)
    assert sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes'], str(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    assert rows and all(isinstance(row, dict) for row in rows), str(path)
    ids = [row['pmc_id'] for row in rows]
    assert all(isinstance(did, str) and did for did in ids), str(path)
    assert len(ids) == len(set(ids)), str(path)
    return rows


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def nonentities(document):
    return {key: value for key, value in document.items() if key != 'entities'}


def changes(evaluate, gold, before, after):
    """Describe changed mention IDs, using the frozen evaluator's gold policy."""
    count = Counter({kind + '_' + role + '_' + outcome: 0
                     for kind in ['entity', 'relation']
                     for role in ['added', 'removed'] for outcome in ['tp', 'fp']})
    assert len(gold) == len(before) == len(after)
    for g, a, b in zip(gold, before, after):
        assert g['pmc_id'] == a['pmc_id'] == b['pmc_id']
        gu, au, bu = [evaluate.positive_units(row, g, 'all') for row in [g, a, b]]
        for role, left, right in [('added', bu, au), ('removed', au, bu)]:
            diff = {span: ids - right.get(span, set()) for span, ids in left.items()
                    if ids - right.get(span, set())}
            tp, fp, _ = evaluate.mention_counts(gu, diff)
            count['entity_' + role + '_tp'] += tp
            count['entity_' + role + '_fp'] += fp
        gt, at, bt = [evaluate.associations(row) for row in [g, a, b]]
        assert gt.keys() == at.keys() == bt.keys()
        assert at == bt
    return dict(count)


def numeric_deltas(after, before):
    assert after.keys() == before.keys()
    return {key: numeric_deltas(value, before[key]) if isinstance(value, dict)
            else value - before[key] for key, value in after.items()}


def assert_same_relations(baseline, top3):
    assert baseline['counts']['relation'] == top3['counts']['relation']
    assert baseline['patients'] == top3['patients']
    for metric in ['micF1', 'macF1']:
        assert baseline['metrics'][metric] == top3['metrics'][metric]
        for mode in baseline['sensitivity']:
            assert baseline['sensitivity'][mode][metric] == top3['sensitivity'][mode][metric]
    assert [row['pmc_id'] for row in baseline['per_document']] == [
        row['pmc_id'] for row in top3['per_document']]
    for left, right in zip(baseline['per_document'], top3['per_document']):
        assert left['relation'] == right['relation']
        assert left['patients'] == right['patients']


def compare(evaluate, validator, hpo, gold, before, after, baseline_report, baseline_validation):
    validations = {'baseline': baseline_validation}
    reports = {'baseline': baseline_report}
    valid = validator.validate_predictions(gold, after, hpo)
    assert valid['valid'] and valid['error_count'] == valid['warning_count'] == 0, valid
    assert baseline_validation['valid'] and baseline_validation['error_count'] == baseline_validation['warning_count'] == 0
    validations['pair'] = valid
    reports['pair'] = evaluate.evaluate_report(gold, after, 'all', 'text')
    baseline, top3 = reports['baseline'], reports['pair']
    assert_same_relations(baseline, top3)
    result = {arm: report['metrics'] for arm, report in reports.items()}
    result.update(deltas=numeric_deltas(top3['metrics'], baseline['metrics']),
                  count_deltas=numeric_deltas(top3['counts'], baseline['counts']),
                  sensitivity_deltas=numeric_deltas(top3['sensitivity'], baseline['sensitivity']),
                  changes=changes(evaluate, gold, before, after),
                  errors={arm: value['error_count'] for arm, value in validations.items()},
                  warnings={arm: value['warning_count'] for arm, value in validations.items()},
                  relations_identical=True)
    return reports, validations, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = read_json(plan_path)
    root = Path(plan['output_dir'])
    assert root.is_absolute()
    assert not (root / 'evaluation-start.json').exists()
    assert not (root / 'results.json').exists()
    freeze_path = root / 'complete-freeze.json'
    freeze = read_json(freeze_path)
    assert freeze['plan_sha256'] == sha(plan_path)
    assert freeze['all_four_predictions_complete'] is True
    assert freeze['all_four_fixed_span_predictions_complete'] is True
    assert freeze['new_quality_evaluation_started'] is False
    assert [fold['fold'] for fold in plan['folds']] == [0, 1, 2, 3]
    assert [fold['fold'] for fold in freeze['folds']] == [0, 1, 2, 3]

    # Verify all sixteen prediction references before reading either quality labels or scores.
    for spec, frozen in zip(plan['folds'], freeze['folds']):
        assert frozen['baseline_prediction']['sha256'] == plan['comparison_top3'][spec['fold']]['prediction']['sha256']
        assert Path(frozen['baseline_prediction']['path']).resolve() == Path(
            plan['comparison_top3'][spec['fold']]['prediction']['path']).resolve()
        checked(plan['comparison_top3'][spec['fold']]['prediction'])
        checked(frozen['new_prediction'])
        checked(frozen['fixed_span_prediction'])
        checked(frozen['fixed_span_decisions'])
        assert frozen['retained_bioformer_prediction'] == spec['baseline_prediction']
        checked(frozen['retained_bioformer_prediction'])

    sources = {str(Path(path).resolve()): digest for path, digest in plan['sources'].items()}
    assert len(sources) == len(plan['sources'])
    assert str(Path(__file__).resolve()) in sources
    for path, digest in sources.items():
        assert sha(path) == digest, path
    fit_source = Path(__file__).resolve().with_name('fit_control.py')
    assert str(fit_source) in sources
    run_source = str(Path(__file__).resolve().with_name('run_control.py'))
    assert run_source in sources and freeze['source_sha256'] == sources[run_source]
    prepared_path = checked(plan['prepared_manifest'])
    if 'prepared_manifest' in freeze:
        assert checked(freeze['prepared_manifest']).resolve() == prepared_path.resolve()
        assert freeze['prepared_manifest']['sha256'] == plan['prepared_manifest']['sha256']
    prepared = read_json(prepared_path)
    assert [fold['fold'] for fold in prepared['folds']] == [0, 1, 2, 3]
    hpo_path = checked(plan['hpo'])

    fold_rows, seen = [], set()
    for spec, frozen, prep in zip(plan['folds'], freeze['folds'], prepared['folds']):
        manifest_path = checked(frozen['new_manifest'])
        checked(frozen['scores'])
        manifest = read_json(manifest_path)
        assert manifest['source_sha256'] == sources[str(fit_source)]
        assert manifest['plan_sha256'] == sha(plan_path)
        assert manifest['prepared_manifest_sha256'] == plan['prepared_manifest']['sha256']
        assert manifest['actual_iterations'] == 160 and manifest['early_stopping_performed'] is False
        assert manifest['threshold'] == .3
        for role, name in [('new_prediction', 'augmented.jsonl'), ('scores', 'selector-scores.jsonl')]:
            assert manifest['outputs'][name] == frozen[role]['sha256']
            assert Path(frozen[role]['path']).resolve() == (manifest_path.parent / name).resolve()
        checked(prep['train'])
        prepared_input = read_rows(checked(prep['input']))
        assert len(prepared_input) == 16
        for row in prepared_input:
            assert row['origin_index'] is None
            assert set(row['input']) == {'pmc_id', 'pmid', 'full_text', 'patient'}
            assert row['pmc_id'] == row['input']['pmc_id']
        before = read_rows(frozen['baseline_prediction']['path'])
        after = read_rows(frozen['new_prediction']['path'])
        ids = [row['pmc_id'] for row in after]
        assert len(ids) == 16 and ids == [row['pmc_id'] for row in before]
        assert ids == [row['pmc_id'] for row in prepared_input]
        assert not seen.intersection(ids)
        seen.update(ids)
        for old, new in zip(before, after):
            assert nonentities(old) == nonentities(new), old['pmc_id']
            assert 'association' in old
        bioformer = read_rows(frozen['retained_bioformer_prediction']['path'])
        fixed = read_rows(frozen['fixed_span_prediction']['path'])
        assert [d['pmc_id'] for d in fixed] == ids == [d['pmc_id'] for d in bioformer]
        for a, b in zip(bioformer, fixed):
            assert nonentities(a) == nonentities(b)
            assert [{k:v for k,v in e.items() if k != 'identifier'} for e in a['entities']] == [{k:v for k,v in e.items() if k != 'identifier'} for e in b['entities']]
        fold_rows.append((spec['fold'], before, after, prepared_input, bioformer, fixed))
    assert len(seen) == 64

    sys.path.insert(0, str(Path(plan['evaluation_source']).resolve()))
    evaluate = importlib.import_module('evaluate')
    validator = importlib.import_module('validate_predictions')
    for module in [evaluate, validator]:
        path = str(Path(module.__file__).resolve())
        assert Path(path).parent == Path(plan['evaluation_source']).resolve()
        assert path in sources and sha(path) == sources[path]
    assert evaluate.VERSION == 'patientphex-public-approx-v2'
    hpo = validator.read_hpo(hpo_path)
    assert hpo['version'] == 'hp/releases/2026-06-23'
    expected_outputs = [root / ('fold' + str(fold)) / name
                        for fold in range(4)
                        for name in ['baseline.metrics.json', 'pair.metrics.json',
                                     'baseline.validation.json', 'pair.validation.json', 'results.json',
                                     'fixed-span.metrics.json', 'fixed-span.validation.json', 'fixed-span.results.json']]
    expected_outputs += [root / name for name in ['pooled.baseline.metrics.json', 'pooled.pair.metrics.json',
                        'pooled.baseline.validation.json', 'pooled.pair.validation.json',
                        'pooled.baseline.prediction.jsonl', 'pooled.pair.prediction.jsonl',
                        'pooled.fixed-span.metrics.json', 'pooled.fixed-span.validation.json', 'pooled.fixed-span.prediction.jsonl']]
    assert all(not path.exists() for path in expected_outputs)
    start = {'started_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
             'plan_sha256': sha(plan_path), 'complete_freeze_sha256': sha(freeze_path),
             'all_four_predictions_verified_before_quality_labels': True,
             'baseline_full_v2_recomputed': False, 'new_quality_evaluation_started': True}
    # Exclusive creation is the final guard: interrupted evaluation is not retried.
    write_json(root / 'evaluation-start.json', start)

    gold_item = plan['train64_gold']
    if isinstance(gold_item, str):
        gold_item = {'path': gold_item, 'sha256': plan['train64_gold_sha256']}
    gold_all = read_rows(checked(gold_item))
    truth = {row['pmc_id']: row for row in gold_all}
    assert len(truth) == 64 and truth.keys() == seen
    fold_results, old_map, new_map = [], {}, {}
    fixed_results, bio_map, fixed_map = [], {}, {}
    for fold, before, after, prepared_input, bioformer, fixed in fold_rows:
        gold = [truth[row['pmc_id']] for row in after]
        for doc, prepared_row in zip(gold, prepared_input):
            assert {key: value for key, value in doc.items() if key not in ['entities', 'association']} == prepared_row['input']
        old = plan['baseline_reports']['folds'][fold]
        assert old['fold'] == fold
        reports, validations, result = compare(evaluate, validator, hpo, gold, before, after,
            read_json(checked(old['metrics'])), read_json(checked(old['validation'])))
        result['fold'] = fold
        target = root / ('fold' + str(fold))
        for arm in ['baseline', 'pair']:
            write_json(target / (arm + '.metrics.json'), reports[arm])
            write_json(target / (arm + '.validation.json'), validations[arm])
        write_json(target / 'results.json', result)
        fold_results.append(result)
        old_map.update({row['pmc_id']: row for row in before})
        new_map.update({row['pmc_id']: row for row in after})
        old_fixed = plan['retained_bioformer_reports']['folds'][fold]
        fr, fv, fc = compare(evaluate, validator, hpo, gold, bioformer, fixed,
            read_json(checked(old_fixed['metrics'])), read_json(checked(old_fixed['validation'])))
        fc['fold'] = fold
        write_json(target / 'fixed-span.metrics.json', fr['pair'])
        write_json(target / 'fixed-span.validation.json', fv['pair'])
        write_json(target / 'fixed-span.results.json', fc)
        fixed_results.append(fc)
        bio_map.update({row['pmc_id']: row for row in bioformer})
        fixed_map.update({row['pmc_id']: row for row in fixed})
    before = [old_map[row['pmc_id']] for row in gold_all]
    after = [new_map[row['pmc_id']] for row in gold_all]
    old = plan['baseline_reports']['pooled']
    reports, validations, pooled = compare(evaluate, validator, hpo, gold_all, before, after,
        read_json(checked(old['metrics'])), read_json(checked(old['validation'])))
    for arm, rows in [('baseline', before), ('pair', after)]:
        write_json(root / ('pooled.' + arm + '.metrics.json'), reports[arm])
        write_json(root / ('pooled.' + arm + '.validation.json'), validations[arm])
        with (root / ('pooled.' + arm + '.prediction.jsonl')).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    bioformer = [bio_map[row['pmc_id']] for row in gold_all]
    fixed = [fixed_map[row['pmc_id']] for row in gold_all]
    old_fixed = plan['retained_bioformer_reports']['pooled']
    fr, fv, fixed_pooled = compare(evaluate, validator, hpo, gold_all, bioformer, fixed,
        read_json(checked(old_fixed['metrics'])), read_json(checked(old_fixed['validation'])))
    write_json(root / 'pooled.fixed-span.metrics.json', fr['pair'])
    write_json(root / 'pooled.fixed-span.validation.json', fv['pair'])
    with (root / 'pooled.fixed-span.prediction.jsonl').open('x') as stream:
        for row in fixed: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {'finished_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(plan_path),
              'source_sha256': sha(__file__), 'freeze_sha256': sha(freeze_path),
              'evaluation_started': start, 'quality_evaluation_complete': True,
              'gold_sha256': gold_item['sha256'], 'evaluator_version': evaluate.VERSION,
              'folds': fold_results, 'pooled': pooled,
              'fixed_span': {'folds': fixed_results, 'pooled': fixed_pooled, 'used_for_cut_selection': False},
              'unweighted_mean': {arm: {metric: sum(fold[arm][metric] for fold in fold_results) / 4
                                      for metric in pooled[arm]} for arm in ['baseline', 'pair']},
              'outputs': {str(path.relative_to(root)): sha(path) for path in expected_outputs},
              'scope': 'Four train64 development folds; local public v2 all/text, not an official score.',
              'selection': 'Same fixed top3 candidates and old features; three public pair means only, HGB160 and cutoff .3; separate prespecified nine-cut training calibration.',
              'baseline_relation_provenance': 'The frozen cutoff-.3 files retain their original cutoff-.4 association fields.',
              'baseline_full_v2_recomputed': False, 'relations_identical': True,
              'original_dev_read': False, 'official_actions': False}
    write_json(root / 'results.json', result)
    print(json.dumps({'fold_deltas': [fold['deltas']['Score'] for fold in fold_results], 'pooled': pooled}))


if __name__ == '__main__':
    main()
