"""Bind existing resources to the one authorized original-pool pair control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def checked(item):
    path = Path(item['path'])
    assert path.is_absolute() and sha(path) == item['sha256'], str(path)
    return path


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['prepare', 'fit', 'calibration'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--legacy-plan')
    ap.add_argument('--pair-plan')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    source = Path(__file__).resolve().parent
    feature_source = source.parent / 'phenobert_pair_control/features.py'
    features = ['phenobert_not_match', 'phenobert_related', 'phenobert_match']
    if args.phase == 'prepare':
        old = read(args.legacy_plan)
        pair = read(args.pair_plan)
        checked(old['old_prepared_manifest'])
        checked(pair['pair_cache']) if 'sha256' in pair['pair_cache'] else None
        # Copy only frozen source bytes. Do not import a mutable other worktree.
        snapshots = root / 'snapshots'
        snapshots.mkdir(exist_ok=False)
        dependencies = {}
        for original, digest in old['sources'].items():
            path = Path(original)
            assert sha(path) == digest
            if path.parent.name == 'public_sap_old_pool_control':
                continue
            if path.name in ['evaluate.py', 'validate_predictions.py']:
                target = snapshots / 'evaluation' / path.name
            elif path.parent.name == 'legacy':
                target = snapshots / 'legacy' / path.name
            else:
                target = snapshots / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            assert sha(target) == digest
            dependencies[str(target)] = digest
        old_state_path = Path(old['output_dir']) / 'run-state.json'
        old_state = read(old_state_path)
        assert old_state['status'] == 'complete_four_fits_and_common_freeze_no_quality'
        seconds = old_state['fit_infer_elapsed_seconds']
        estimated = 1.5 * seconds + 10
        assert estimated <= 120
        prep = dict(
            scope='Same original Bioformer pools plus three public pair means, no top3/Sap fields',
            frozen_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(source / 'prepare_pools.py'),
            feature_source=ref(feature_source),
            old_prepared_manifest=old['old_prepared_manifest'],
            pair_cache=pair['pair_cache'],
            output_dir=str(root / 'prepared'),
            estimated_four_fit_infer_seconds=estimated,
            estimate_basis={'same_old_pool_two_feature_control': ref(old_state_path),
                            'actual_seconds': seconds, 'multiplier': 1.5, 'overhead_seconds': 10},
            no_fit=True, no_new_forward=True, no_quality=True)
        write(root / 'prepare-plan.json', prep)
        write(root / 'source-snapshots.json', dependencies)
        write(root / 'origin-plans.json', {'legacy': ref(args.legacy_plan), 'pair': ref(args.pair_plan)})
        print(json.dumps(ref(root / 'prepare-plan.json')))
    elif args.phase == 'fit':
        origins = read(root / 'origin-plans.json')
        old = read(checked(origins['legacy']))
        pair = read(checked(origins['pair']))
        prep = read(root / 'prepared/manifest.json')
        assert prep['old_keys_fields_types_order_exact'] is True
        assert prep['new_features'] == features
        assert prep['top3_new_keys'] is prep['top3_six_features'] is False
        assert prep['estimated_four_fit_infer_seconds'] <= 120
        dependencies = read(root / 'source-snapshots.json')
        dependencies.update({str(path): sha(path) for path in source.glob('*.py')})
        dependencies[str(feature_source)] = sha(feature_source)
        plan = dict(old)
        plan.update(experiment='Original Bioformer pools plus three PhenoBERT pair means only',
                    frozen_at=datetime.now(timezone.utc).isoformat(),
                    output_dir=str(root / 'formal-control'),
                    prepared_manifest=ref(root / 'prepared/manifest.json'),
                    pair_cache=pair['pair_cache'], features=features,
                    ontology_source=str(root / 'snapshots'),
                    evaluation_source=str(root / 'snapshots/evaluation'),
                    sources=dependencies, source_check=ref(root / 'source-diff-review.json'),
                    origin_plans=origins,
                    old_calibration_results=ref(Path(old['output_dir']).parents[2] /
                        'sequence_crf/public-bioformer-cut-calibration/results.json'),
                    selection='Only three public pair features. HGB160 and old keys/fields/order unchanged. '
                              'Nine fixed cuts choose pooled (menF1+docF1)/2, tie nearest .4 then higher. '
                              'No fixed-ID arm, no originaldev, no new neural inference.',
                    execution_authorized=True)
        plan.pop('public_feature_manifest')
        write(root / 'formal-plan.json', plan)
        print(json.dumps(ref(root / 'formal-plan.json')))
    else:
        fit_plan = read(root / 'formal-plan.json')
        freeze = read(root / 'formal-control/complete-freeze.json')
        assert freeze['plan_sha256'] == sha(root / 'formal-plan.json')
        assert freeze['all_four_predictions_complete'] is True
        folds = []
        for item in freeze['folds']:
            for field in ['scores', 'new_manifest', 'new_prediction', 'baseline_prediction']:
                checked(item[field])
            folds.append(dict(fold=item['fold'], scores=item['scores'], manifest=item['new_manifest'],
                              baseline_prediction=item['new_prediction'], old_template=item['baseline_prediction']))
        evaluator = Path(fit_plan['evaluation_source']) / 'evaluate.py'
        gold = read(checked(read(root / 'origin-plans.json')['pair']))['train64_gold']
        plan = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
                    scope='Prespecified nine-cut calibration of fixed old pool plus three public pair means',
                    output_dir=str(root / 'calibration'), cuts=fit_plan['cuts'], CPU_cap_seconds=30,
                    TERM_grace_seconds=5, python=fit_plan['python'], environment=fit_plan['environment'],
                    sources={str(source / 'calibrate_cut.py'): sha(source / 'calibrate_cut.py'), str(evaluator): sha(evaluator)},
                    evaluation_source=fit_plan['evaluation_source'], train64=gold,
                    old_calibration_results=fit_plan['old_calibration_results'],
                    fit_plan=ref(root / 'formal-plan.json'),
                    complete_freeze=ref(root / 'formal-control/complete-freeze.json'), folds=folds)
        write(root / 'cut-plan.json', plan)
        print(json.dumps(ref(root / 'cut-plan.json')))


if __name__ == '__main__':
    main()
