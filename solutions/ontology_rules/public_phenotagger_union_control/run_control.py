"""Four fixed source-HGB fits share one 120-second CPU budget; then freeze."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    p = Path(p)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    root = Path(plan['output_dir'])
    assert not (root/'run-state.json').exists()
    assert not (root/'complete-freeze.json').exists()
    for p, digest in plan['sources'].items():
        assert sha(p) == digest, p
    prep = root/'prepared'
    pm = json.loads((prep/'manifest.json').read_text())
    assert pm['plan_sha256'] == sha(a.plan) and pm['unique_keys'] == 8221
    assert pm['output_sha256'] == sha(prep/'candidates.jsonl')
    assert [x['fold'] for x in plan['folds']] == [0, 1, 2, 3]
    state = {'started_at': now(), 'plan_sha256': sha(a.plan), 'source_sha256': sha(__file__),
             'prepared_manifest': ref(prep/'manifest.json'), 'jobs': [], 'status': 'running',
             'shared_CPU_seconds': 120, 'TERM_grace_seconds': 5, 'evaluation_started': False}
    def save():
        (root/'run-state.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    started = time.monotonic()
    env = dict(os.environ)
    env.update(plan['environment'])
    try:
        for fold in plan['folds']:
            for role in ['reference', 'old095_prediction', 'old095_metrics', 'old095_manifest']:
                item = fold[role]
                assert sha(item['path']) == item['sha256'], role
            old_manifest = json.loads(Path(fold['old095_manifest']['path']).read_text())
            assert old_manifest['outputs']['augmented.jsonl'] == fold['old095_prediction']['sha256']
            target = root/('fold'+str(fold['fold']))/'fitted'
            target.parent.mkdir()
            argv = [plan['python'], fold['selector_source'], '--ontology-source', plan['ontology_source'],
                    '--reference', fold['reference']['path'], '--public', str(prep/'candidates.jsonl'),
                    '--public-manifest', str(prep/'manifest.json'), '--output-dir', str(target)]
            job = {'fold': fold['fold'], 'argv': argv, 'started_at': now()}
            state['jobs'].append(job)
            save()
            left = 120-(time.monotonic()-started)
            assert left > 0
            with (target.parent/'fit.stdout').open('x') as out, (target.parent/'fit.stderr').open('x') as err:
                child = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
                job['pid'] = child.pid
                save()
                t = time.monotonic()
                try:
                    rc = child.wait(timeout=left)
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
                    job['returncode'] = child.returncode
                    job['elapsed_seconds'] = time.monotonic()-t
                    job['finished_at'] = now()
                    save()
            assert rc == 0
        state['fit_infer_elapsed_seconds'] = time.monotonic()-started
        assert state['fit_infer_elapsed_seconds'] <= 120
        frozen = []
        seen = set()
        for fold in plan['folds']:
            target = root/('fold'+str(fold['fold']))/'fitted'
            manifest = json.loads((target/'manifest.json').read_text())
            assert manifest['source_sha256'] == plan['sources'][fold['selector_source']]
            assert manifest['public_manifest_sha256'] == sha(prep/'manifest.json')
            assert manifest['public_sha256'] == sha(prep/'candidates.jsonl')
            for name, digest in manifest['outputs'].items():
                assert sha(target/name) == digest
            rows = [json.loads(x) for x in (target/'augmented.jsonl').read_text().splitlines()]
            old = [json.loads(x) for x in Path(fold['old095_prediction']['path']).read_text().splitlines()]
            assert [d['pmc_id'] for d in rows] == [d['pmc_id'] for d in old]
            ids = {d['pmc_id'] for d in rows}
            assert len(rows) == len(ids) == 16 and not ids & seen
            seen |= ids
            frozen.append({'fold': fold['fold'], 'new_prediction': ref(target/'augmented.jsonl'),
                           'new_manifest': ref(target/'manifest.json'), 'scores': ref(target/'selector-scores.jsonl'),
                           'old095_prediction': fold['old095_prediction'], 'old095_metrics': fold['old095_metrics']})
        assert len(seen) == 64
        freeze = {'frozen_at': now(), 'plan_sha256': sha(a.plan), 'source_sha256': sha(__file__),
                  'prepared_manifest': ref(prep/'manifest.json'), 'prepared_candidates': ref(prep/'candidates.jsonl'),
                  'folds': frozen, 'all_four_predictions_complete': True, 'new_quality_evaluation_started': False,
                  'training_labels_scope': 'Each new HGB fit only its frozen outertrain48, using three train32-to-inner16 OOF pools.'}
        (root/'complete-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
        state['freeze'] = ref(root/'complete-freeze.json')
        state['status'] = 'complete_four_fits_and_common_freeze_no_quality'
    except BaseException as exc:
        state['status'] = 'failed_preserved'
        state['failure'] = repr(exc)
        raise
    finally:
        state['finished_at'] = now()
        save()
    print(json.dumps(state))


if __name__ == '__main__':
    main()
