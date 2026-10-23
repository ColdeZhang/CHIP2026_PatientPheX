"""Bind a completed public cache to the already authorized immutable protocol.

No model, task labels, fit or quality evaluation; exclusive output creation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import features


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=features.sha(path), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--cache-manifest', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.protocol).read_text())
    assert plan['root_authorized'] is True and plan['execution_authorized'] is False
    assert plan['fit_infer_CPU_cap_seconds'] == 300
    assert plan['calibration']['cuts'] == [i / 10 for i in range(1, 10)]
    assert plan['calibration']['CPU_cap_seconds'] == 30
    cache = json.loads(Path(a.cache_manifest).read_text())
    assert cache['status'] == 'PASS_COMPLETE_PUBLIC_PAIR_TRAIN64_INPUT_ONLY_CACHE'
    assert cache['actual_unique_tuples'] == cache['requested_unique_tuples'] == 135000
    assert cache['missing_keys'] == cache['extra_keys'] == cache['duplicate_keys'] == 0
    assert cache['all_finite_normalized'] is True
    assert cache['no_dev_input'] is cache['no_training'] is cache['no_gold_or_quality_evaluation'] is True
    assert cache['atomic_request'] == plan['pair_cache']['atomic_request']
    assert cache['hpo'] == plan['hpo']
    plan['pair_cache']['scores'] = cache['scores']
    plan['pair_cache']['cache_manifest'] = ref(a.cache_manifest)
    actual = features.load(plan['pair_cache'])
    assert len(actual.by_atomic_pair) == 137784 and len(actual.model_scores) == 135000
    plan['cache_cpu_binding'] = dict(unique_raw_pairs=len(actual.by_atomic_pair),
                                   unique_native_inputs=len(actual.model_scores),
                                   actual_load_contract_passed=True,
                                   candidate_features_not_yet_fit=True,
                                   source_sha256=features.sha(features.__file__))
    plan['protocol'] = ref(a.protocol)
    plan['execution_authorized'] = True
    plan['status'] = 'READY_FROZEN_ACTUAL_CACHE_BOUND_NO_FITS_YET'
    plan['scope'] = ('Authorized four CPU HGB fits on strict train48 / three nested train32 pools, '
                     'then once-only new fixed .3 quality and prespecified training-side nine-cut calibration. '
                     'No new neural forward, originaldev input/labels or official action.')
    plan['authorization_boundary'] = ('Root authorized the unchanged protocol; complete public cache is now '
                                      'bound. Four fits share 300 seconds plus TERM5; calibration cap30.')
    plan['evaluation'] = ('All four new full predictions and probabilities, with matching frozen top3 '
                          'controls, freeze before new fixed-cut quality. Reuse old metric artifacts. '
                          'Nine cuts are predeclared; all36 decode outputs freeze before calibration labels.')
    plan['frozen_at'] = datetime.now(timezone.utc).isoformat()
    source_dir = Path(__file__).resolve().parent
    for path in source_dir.glob('*.py'):
        plan['sources'][str(path)] = features.sha(path)
    snapshot = Path(plan['ontology_source'])
    plan['evaluation_source'] = str(snapshot / 'evaluation')
    # Use immutable old top3 reports instead of recomputing their quality.
    old_root = Path(plan['comparison_top3'][0]['manifest']['path']).parents[2]
    plan['baseline_reports'] = {
        'folds': [dict(fold=i, metrics=ref(old_root / f'fold{i}/top3.metrics.json'),
                       validation=ref(old_root / f'fold{i}/top3.validation.json'))
                  for i in range(4)],
        'pooled': dict(metrics=ref(old_root / 'pooled.top3.metrics.json'),
                       validation=ref(old_root / 'pooled.top3.validation.json')),
        'complete_freeze': ref(old_root / 'complete-freeze.json'),
        'results': ref(old_root / 'results.json')}
    plan['retained_bioformer_reports'] = {
        'folds': [dict(fold=i, metrics=ref(old_root / f'fold{i}/baseline.metrics.json'),
                       validation=ref(old_root / f'fold{i}/baseline.validation.json'))
                  for i in range(4)],
        'pooled': dict(metrics=ref(old_root / 'pooled.baseline.metrics.json'),
                       validation=ref(old_root / 'pooled.baseline.validation.json'))}
    old_freeze = json.loads(Path(plan['baseline_reports']['complete_freeze']['path']).read_text())
    assert len(old_freeze['folds']) == 4
    for comparison, frozen in zip(plan['comparison_top3'], old_freeze['folds']):
        assert comparison['fold'] == frozen['fold']
        assert comparison['prediction']['sha256'] == frozen['new_prediction']['sha256']
        assert comparison['scores']['sha256'] == frozen['scores']['sha256']
        for name in ['manifest', 'prediction', 'scores']:
            features.checked(comparison[name])
    for path, digest in plan['sources'].items():
        assert features.sha(path) == digest, path
    with Path(a.output).open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan=ref(a.output), cache=ref(a.cache_manifest),
                         no_fit=True, no_quality=True)))


if __name__ == '__main__':
    main()
