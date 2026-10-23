"""Four fixed HGB fits share 120 seconds, then freeze all eight full predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(item):
    path = Path(item['path'])
    assert path.is_absolute() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    args = ap.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    plan_sha = sha(plan_path)
    root = Path(plan['output_dir'])
    assert root.is_absolute() and root.exists()
    assert not (root / 'complete-freeze.json').exists()
    assert not (root / 'evaluation-start.json').exists()
    assert plan['fit_infer_CPU_cap_seconds'] == 120 and plan['TERM_grace_seconds'] == 5
    assert str(Path(__file__).resolve()) in plan['sources']
    for path, digest in plan['sources'].items():
        assert sha(path) == digest, path
    prep = read(checked(plan['prepared_manifest']))
    old_pools = read(checked(plan['old_prepared_manifest']))
    assert [f['fold'] for f in old_pools['folds']] == [0, 1, 2, 3]
    assert prep['old_prepared_manifest'] == plan['old_prepared_manifest']
    assert prep['new_features'] == ['phenobert_not_match', 'phenobert_related', 'phenobert_match']
    assert prep['estimated_four_fit_infer_seconds'] <= 120
    for new_fold, old_fold in zip(prep['folds'], old_pools['folds']):
        assert new_fold['fold'] == old_fold['fold']
        for field in ('old_train_rows', 'old_input_rows', 'train_ids', 'input_ids', 'nested_input_ids'):
            assert new_fold[field] == old_fold[field], ('unchanged_old_pool', field)
        for role in ('train', 'input'):
            checked(new_fold[role])
            checked(old_fold[role])
    assert [f['fold'] for f in prep['folds']] == [0, 1, 2, 3]
    assert [f['fold'] for f in plan['folds']] == [0, 1, 2, 3]
    fit_source = Path(__file__).resolve().with_name('fit_control.py')
    assert str(fit_source) in plan['sources']
    for fold in plan['folds']:
        for role in ['old_fit_manifest', 'reference', 'baseline_prediction']:
            checked(fold[role])
        assert not (root / ('fold' + str(fold['fold'])) / 'fitted').exists()
    state = dict(started_at=now(), plan_sha256=plan_sha, source_sha256=sha(__file__),
                 prepared_manifest=plan['prepared_manifest'], jobs=[], status='running',
                 shared_CPU_seconds=120, TERM_grace_seconds=5, evaluation_started=False)
    with (root / 'run-state.json').open('x') as stream:
        json.dump(state, stream, indent=2)

    def save():
        (root / 'run-state.json').write_text(json.dumps(state, indent=2) + '\n')

    env = dict(os.environ, **plan['environment'])
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert all(env[name] == '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    started = time.monotonic()
    try:
        for fold in plan['folds']:
            target = root / ('fold' + str(fold['fold']))
            target.mkdir(exist_ok=False)
            argv = [plan['python'], str(fit_source), '--plan', str(plan_path), '--fold', str(fold['fold'])]
            job = dict(fold=fold['fold'], argv=argv, started_at=now())
            state['jobs'].append(job)
            save()
            remaining = 120 - (time.monotonic() - started)
            assert remaining > 0
            with (target / 'fit.stdout').open('x') as out, (target / 'fit.stderr').open('x') as err:
                child = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
                job['pid'] = child.pid
                save()
                begin = time.monotonic()
                try:
                    code = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    job['timed_out'] = True
                    raise
                finally:
                    job.update(returncode=child.returncode, seconds=time.monotonic()-begin, finished_at=now())
                    save()
            assert code == 0, fold['fold']
        state['fit_infer_elapsed_seconds'] = time.monotonic() - started
        assert state['fit_infer_elapsed_seconds'] <= 120
        for path, digest in plan['sources'].items():
            assert sha(path) == digest, path
        assert sha(plan_path) == plan_sha
        frozen, seen = [], set()
        for fold in plan['folds']:
            target = root / ('fold' + str(fold['fold'])) / 'fitted'
            manifest = read(target / 'manifest.json')
            assert manifest['source_sha256'] == plan['sources'][str(fit_source)]
            assert manifest['plan_sha256'] == plan_sha
            assert manifest['prepared_manifest_sha256'] == plan['prepared_manifest']['sha256']
            assert manifest['actual_iterations'] == 160 and manifest['threshold'] == .3
            assert manifest['early_stopping_performed'] is False
            assert manifest['new_features'] == ['phenobert_not_match', 'phenobert_related', 'phenobert_match']
            assert manifest['old_prepared_manifest'] == plan['old_prepared_manifest']
            assert manifest['assertions']['old_pool_fields_keys_order_exact'] is True
            assert manifest['assertions']['only_three_pair_features_added'] is True
            for name, digest in manifest['outputs'].items():
                assert sha(target / name) == digest
            old = [json.loads(x) for x in checked(fold['baseline_prediction']).read_text().splitlines()]
            new = [json.loads(x) for x in (target / 'augmented.jsonl').read_text().splitlines()]
            assert [d['pmc_id'] for d in old] == [d['pmc_id'] for d in new]
            ids = {d['pmc_id'] for d in new}
            assert len(new) == len(ids) == 16 and not ids & seen
            seen |= ids
            for a, b in zip(old, new):
                assert {k:v for k,v in a.items() if k != 'entities'} == {k:v for k,v in b.items() if k != 'entities'}
            frozen.append(dict(fold=fold['fold'], new_prediction=ref(target / 'augmented.jsonl'),
                               new_manifest=ref(target / 'manifest.json'), scores=ref(target / 'selector-scores.jsonl'),
                               baseline_prediction=fold['baseline_prediction']))
        assert len(seen) == 64
        freeze = dict(frozen_at=now(), plan_sha256=plan_sha, source_sha256=sha(__file__),
                      prepared_manifest=plan['prepared_manifest'],
                      old_prepared_manifest=plan['old_prepared_manifest'], folds=frozen,
                      all_four_predictions_complete=True, new_quality_evaluation_started=False,
                      nonentity_fields_equal=True,
                      feature_increment=['phenobert_not_match', 'phenobert_related', 'phenobert_match'],
                      old_pool_fields_keys_order_exact=True,
                      association_provenance='Both arms preserve frozen cut .3 templates whose association fields originate from .4.')
        with (root / 'complete-freeze.json').open('x') as stream:
            json.dump(freeze, stream, indent=2)
            stream.write('\n')
        state.update(status='complete_four_fits_and_common_freeze_no_quality', freeze=ref(root / 'complete-freeze.json'))
    except BaseException as exc:
        state.update(status='failed_preserved', failure=repr(exc))
        raise
    finally:
        state['finished_at'] = now()
        save()
    print(json.dumps(state))


if __name__ == '__main__':
    main()
