"""Bounded fresh BCE/focal control with explicit process-group timeout handling."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--stage', choices=['cpu', 'smoke', 'formal'], required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for path, expected in plan['required_sha256'].items():
        assert sha(path) == expected, path
    state_path = Path(args.stage+'.state.json')
    assert not state_path.exists(), 'Use immutable new output paths, no implicit retry/resume'
    state = {'stage': args.stage, 'started_at_utc': now(), 'plan_sha256': sha(args.plan),
             'wrapper_sha256': sha(__file__), 'gold_read': False, 'stages': []}
    if args.stage != 'cpu':
        cpu = json.loads(Path('cpu-preflight.json').read_text())
        pure = json.loads(Path('cpu-loss.json').read_text())
        assert cpu['status'] == 'PASS_CPU_readiness' and cpu['all_original_rows_collate_pass']
        assert pure['status'] == 'PASS_CPU_synthetic_only' and not pure['cuda_initialized']
        assert cpu['sources']['trainer'] == plan['required_sha256']['train_gliner_loss_control.py']
        assert pure['loss_source_sha256'] == plan['required_sha256']['focal_loss.py']
        free = shutil.disk_usage('.').free
        assert free >= plan['minimum_free_disk_bytes']
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
        assert plan['gpu_uuid'] not in apps, apps
        state.update(fresh_gpu_empty=True, compute_apps=apps, free_bytes_before=free,
                     cpu_preflight_sha256=sha('cpu-preflight.json'), cpu_loss_sha256=sha('cpu-loss.json'))
    if args.stage == 'formal':
        smoke = json.loads(Path('smoke/run_manifest.json').read_text())
        assert smoke['smoke_completed'] and smoke['parameters'] == 195175936
        assert smoke['fingerprint']['source_sha256'] == plan['required_sha256']['train_gliner_loss_control.py']
        assert smoke['fingerprint']['configuration']['loss_choice'] == 'focal'
        assert math.isfinite(smoke['smoke_gradient_norm']) and smoke['smoke_gradient_norm'] > 0
        assert smoke['max_gpu_allocated_bytes'] < 22*1024**3
        assert not list(Path('smoke').rglob('*.pt')) and not list(Path('smoke').rglob('*.bin'))
        state['smoke_manifest_sha256'] = sha('smoke/run_manifest.json')
        assert not Path('bce').exists() and not Path('focal').exists()
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    train_started = None
    for stage in plan[args.stage]:
        cap = stage['timeout_seconds']
        if stage['kind'] == 'train':
            if train_started is None:
                train_started = time.monotonic()
                state['training_started_at_utc'] = now()
                state['training_deadline_utc'] = (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=plan['shared_training_timeout_seconds'])).isoformat()
            cap = min(cap, plan['shared_training_timeout_seconds']-(time.monotonic()-train_started))
            assert cap > 0, 'Shared training budget exhausted; do not extend'
        row = {'name': stage['name'], 'kind': stage['kind'], 'started_at_utc': now(),
               'command': stage['command'], 'effective_timeout_seconds': cap}
        state['stages'].append(row)
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        started = time.monotonic()
        env = os.environ.copy()
        env.update(stage['env'])
        proc = subprocess.Popen(stage['command'], env=env, start_new_session=True)
        row['pid'] = proc.pid
        try:
            code = proc.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            row['timed_out'] = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
                row['killed_after_term_grace'] = True
        row.update(exit_code=code, finished_at_utc=now(), seconds=time.monotonic()-started)
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        assert code == 0 and not row.get('timed_out'), row
        if stage['kind'] == 'train':
            tr = json.loads(Path(stage['arm']+'/run_manifest.json').read_text())
            assert tr['completed_epochs'] == 10 and tr['total_steps'] == 2880 and not tr['arguments']['resume']
            assert tr['fingerprint']['base_weight_sha256'] == plan['base_weight_sha256']
            assert tr['fingerprint']['configuration']['loss_choice'] == stage['arm']
            row['training_manifest_sha256'] = sha(stage['arm']+'/run_manifest.json')
        if stage['kind'] == 'inference':
            arm = stage['arm']
            tr = json.loads(Path(arm+'/run_manifest.json').read_text())
            inf_dir = Path(arm+'-inference')
            inf = json.loads((inf_dir/'manifest.json').read_text())
            weight = tr['epoch_results'][-1]['model_sha256']
            assert inf['files'][arm+'/epoch-010/pytorch_model.bin'] == weight
            assert inf['input_sha256'] == plan['input_sha256']
            assert inf['candidates_sha256'] == sha(inf_dir/'candidates.jsonl')
            freeze = {'frozen_at_utc': now(), 'candidate_sha256': inf['candidates_sha256'],
                      'input_sha256': inf['input_sha256'], 'model_sha256': weight,
                      'actual_manifest_sha256': sha(inf_dir/'manifest.json'),
                      'training_manifest_sha256': sha(arm+'/run_manifest.json'), 'gold_read': False}
            (inf_dir/'prediction-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
            row['raw_freeze_sha256'] = sha(inf_dir/'prediction-freeze.json')
        state_path.write_text(json.dumps(state, indent=2)+'\n')
    state.update(complete=True, finished_at_utc=now(), free_bytes_after=shutil.disk_usage('.').free)
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'stage': args.stage, 'complete': True, 'state_sha256': sha(state_path)}), flush=True)


if __name__ == '__main__':
    main()
