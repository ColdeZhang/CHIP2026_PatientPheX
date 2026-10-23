#!/usr/bin/env python3
"""Execute one frozen context-control stage with independent hard deadline."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc)


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--stage', required=True, choices=['cpu', 'smoke', 'train', 'infer', 'link'])
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['training_epochs'] == 10 and plan['training_steps'] == 2880
    assert plan['training_rows'] == 1150 and plan['fresh_public_base'] is True
    stage = plan['stages'][args.stage]
    assert stage['timeout_seconds'] == {'cpu':240, 'smoke':300, 'train':3600, 'infer':300, 'link':300}[args.stage]
    assert plan['terminate_grace_seconds'] == 30
    state_path = Path(plan['state_dir'])/(args.stage+'.state.json')
    state_path.parent.mkdir(parents=True, exist_ok=True)
    assert not state_path.exists(), 'Never overwrite or silently retry a stage'
    assert not Path(stage['output']).exists(), 'New stage requires fresh output'
    for name, expected in plan['source_and_input_sha256'].items():
        assert sha(name) == expected, name
    env = dict(os.environ); env.update(stage.get('environment', {}))
    record = {'stage':args.stage, 'plan_sha256':sha(args.plan), 'wrapper_sha256':sha(__file__),
              'command':stage['command'], 'environment_override':stage.get('environment', {}),
              'timeout_seconds':stage['timeout_seconds'], 'terminate_grace_seconds':30,
              'complete':False, 'heldout_gold_read':False}
    if args.stage == 'cpu':
        assert env.get('CUDA_VISIBLE_DEVICES') == ''
    else:
        assert env.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid']
        apps = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                              '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
        assert not any(s.strip().startswith(plan['gpu_uuid']+',') for s in apps.splitlines()), 'Assigned GPU occupied'
        gpu = subprocess.run(['nvidia-smi','--id='+plan['gpu_uuid'], '--query-gpu=uuid,name,memory.free',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()
        assert int(gpu.rsplit(',',1)[1]) > 20000
        record.update(gpu_preflight=gpu, compute_apps_preflight=apps)
        if args.stage in ['smoke', 'train']:
            cpu_path = Path(plan['stages']['cpu']['output'])
            cpu = json.loads(cpu_path.read_text())
            assert cpu['strict_state_load'] and cpu['counts']['rows'] == 1150
            record['cpu_result_sha256'] = sha(cpu_path)
        if args.stage == 'train':
            smoke_path = Path(plan['stages']['smoke']['output'])/'run_manifest.json'
            smoke = json.loads(smoke_path.read_text())
            assert smoke['smoke_completed'] and smoke['completed_epochs'] == 0
            assert smoke['arguments']['smoke_only'] and not smoke['arguments']['resume']
            assert not (smoke_path.parent/'latest-state.pt').exists()
            cost_path = Path(plan['cost_projection_path'])
            cost = json.loads(cost_path.read_text())
            assert cost['smoke_manifest_sha256'] == sha(smoke_path)
            assert cost['projected_training_seconds'] <= 3600 and cost['proceed'] is True
            record.update(smoke_manifest_sha256=sha(smoke_path), cost_projection_sha256=sha(cost_path))
    started = now()
    record.update(started_at_utc=started.isoformat(),
                  deadline_utc=(started+timedelta(seconds=stage['timeout_seconds'])).isoformat())
    save(state_path, record)
    t = time.monotonic()
    with state_path.with_suffix('.stdout.log').open('w') as stdout, state_path.with_suffix('.stderr.log').open('w') as stderr:
        process = subprocess.Popen(stage['command'], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        record['pid'] = process.pid; save(state_path, record)
        try:
            rc = process.wait(timeout=stage['timeout_seconds'])
        except subprocess.TimeoutExpired:
            record['timeout'] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            rc = process.returncode
    record.update(return_code=rc, finished_at_utc=now().isoformat(), seconds=time.monotonic()-t)
    if rc == 0 and not record.get('timeout'):
        output = Path(stage['output'])
        manifest = output if output.is_file() else output/('run_manifest.json' if args.stage in ['train','smoke'] else 'manifest.json')
        record['output_manifest_sha256'] = sha(manifest)
        if args.stage == 'train':
            report = json.loads(manifest.read_text())
            assert report['completed_epochs'] == 10 and report['total_steps'] == 2880
            assert report['rows'] == 1150 and not report['arguments']['resume']
            record['final_model_sha256'] = sha(output/'epoch-010/pytorch_model.bin')
            record['final_state_sha256'] = sha(output/'latest-state.pt')
        record['complete'] = True
    save(state_path, record)
    print(json.dumps(record), flush=True)
    assert record['complete'], 'Frozen stage failed; do not retry silently'


if __name__ == '__main__':
    main()
