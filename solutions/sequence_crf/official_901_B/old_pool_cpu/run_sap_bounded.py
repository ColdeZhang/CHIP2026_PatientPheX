#!/usr/bin/env python3
"""Single-start bounded driver for the frozen Sap CPU/GPU query jobs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def gpu_inventory():
    return {'gpus': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True),
            'apps': subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
             '--format=csv,noheader,nounits'], text=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--stage', choices=['cpu', 'B'], required=True)
    args = parser.parse_args()
    assert sha(args.plan) == args.plan_sha256
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['driver_sha256']
    assert sha(plan['source_path']) == plan['source_sha256']
    stage = plan['stages'][args.stage]
    assert (stage['timeout_seconds'], stage['grace_seconds']) == ((180, 10) if args.stage == 'cpu' else (600, 30))
    root = Path(plan['remote_root'])
    root.mkdir(exist_ok=True)
    state = root/(args.stage+'.driver.started.json')
    with state.open('x') as handle:
        json.dump({'plan_sha256': args.plan_sha256, 'pid': os.getpid(), 'stage': args.stage,
                   'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    process = None
    terminal = {'stage': args.stage, 'plan_sha256': args.plan_sha256, 'status': 'failed_before_start'}
    begin = time.monotonic()
    try:
        assert shutil.disk_usage(root).free >= 2 * 1024**3
        assert not Path(stage['output_dir']).exists()
        environment = os.environ.copy()
        environment.update(stage['environment'])
        if args.stage == 'B':
            cpu_terminal = json.loads((root/'cpu.driver.terminal.json').read_text())
            assert cpu_terminal['returncode'] == 0 and cpu_terminal['status'] == 'complete'
            cpu_manifest = json.loads(Path(plan['stages']['cpu']['output_dir'], 'manifest.json').read_text())
            assert cpu_manifest['document_count'] == 16 and cpu_manifest['candidate_count'] == 12811
            inventory = gpu_inventory()
            save(root/'B.prelaunch-gpu.json', inventory)
            uuid = plan['gpu_uuid']
            matching = [line.split(', ') for line in inventory['gpus'].splitlines() if uuid in line]
            assert len(matching) == 1 and matching[0][0] == '0' and int(matching[0][2]) <= 64
            assert uuid not in inventory['apps']
        with (root/(args.stage+'.stdout.log')).open('xb') as stdout, (root/(args.stage+'.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(stage['argv'], cwd=root, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            save(root/(args.stage+'.child.json'), {'pid': process.pid, 'pgid': process.pid,
                 'argv': stage['argv'], 'timeout_seconds': stage['timeout_seconds'], 'grace_seconds': stage['grace_seconds']})
            terminal['child_pid'] = process.pid
            try:
                returncode = process.wait(timeout=stage['timeout_seconds'])
                terminal['status'] = 'complete' if returncode == 0 else 'child_failed'
            except subprocess.TimeoutExpired:
                terminal['status'] = 'timeout_term'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=stage['grace_seconds'])
                except subprocess.TimeoutExpired:
                    terminal['status'] = 'timeout_kill'
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
            terminal['returncode'] = returncode
        if returncode == 0:
            manifest_path = Path(stage['output_dir'])/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            assert manifest['status'] == 'complete'
            assert manifest['document_count'] == stage['expected_documents']
            assert manifest['candidate_count'] == stage['expected_candidates']
            terminal['manifest_sha256'] = sha(manifest_path)
    except BaseException as exc:
        terminal.update(exception_type=type(exc).__name__, exception=str(exc), status='driver_failed')
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=stage['grace_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        terminal['seconds'] = time.monotonic()-begin
        terminal['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if args.stage == 'B':
            terminal['post_gpu'] = gpu_inventory()
        save(root/(args.stage+'.driver.terminal.json'), terminal)
        print(json.dumps(terminal), flush=True)
    if terminal.get('returncode') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
