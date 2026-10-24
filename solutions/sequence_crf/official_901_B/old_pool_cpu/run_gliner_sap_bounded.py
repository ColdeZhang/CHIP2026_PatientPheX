#!/usr/bin/env python3
"""Single sequential GLiNER-query Sap follow-up, fixed 180+30 seconds."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.plan) == args.plan_sha256
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['driver_sha256']
    assert sha(plan['source_path']) == plan['source_sha256']
    assert sha(plan['base_driver_path']) == plan['base_driver_sha256']
    spec = importlib.util.spec_from_file_location('frozen_bounded_helpers', plan['base_driver_path'])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    root = Path(plan['remote_root'])
    root.mkdir(exist_ok=True)
    with (root/'launch.lock').open('x') as handle:
        handle.write(str(os.getpid())+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    helper.save(root/'driver.started.json', {'plan_sha256': args.plan_sha256, 'pid': os.getpid(), 'started_unix': time.time()})
    assert not Path(plan['output_dir']).exists()
    assert (plan['timeout_seconds'], plan['grace_seconds']) == (180, 30)
    parent_terminal = json.loads(Path(plan['prerequisite_terminal']).read_text())
    assert parent_terminal['status'] == 'complete' and parent_terminal['returncode'] == 0
    assert parent_terminal['manifest_sha256'] == plan['prerequisite_manifest_sha256']
    assert sha(plan['prerequisite_manifest']) == plan['prerequisite_manifest_sha256']
    assert shutil.disk_usage(root).free >= 1024**3
    inventory = helper.gpu_inventory()
    helper.save(root/'prelaunch-gpu.json', inventory)
    uuid = plan['gpu_uuid']
    matches = [line.split(', ') for line in inventory['gpus'].splitlines() if uuid in line]
    assert len(matches) == 1 and matches[0][0] == '0' and int(matches[0][2]) <= 64
    assert uuid not in inventory['apps']
    environment = os.environ.copy()
    environment.update(plan['environment'])
    begin = time.monotonic()
    process = None
    terminal = {'plan_sha256': args.plan_sha256, 'status': 'starting'}
    try:
        with (root/'stdout.log').open('xb') as stdout, (root/'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(plan['argv'], cwd=root, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            helper.save(root/'child.json', {'pid': process.pid, 'pgid': process.pid,
                 'argv': plan['argv'], 'timeout_seconds': 180, 'grace_seconds': 30})
            terminal['child_pid'] = process.pid
            try:
                returncode = process.wait(timeout=180)
                terminal['status'] = 'complete' if returncode == 0 else 'child_failed'
            except subprocess.TimeoutExpired:
                terminal['status'] = 'timeout_term'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminal['status'] = 'timeout_kill'
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
            terminal['returncode'] = returncode
        if returncode == 0:
            manifest_path = Path(plan['output_dir'])/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            assert manifest['status'] == 'complete' and manifest['document_count'] == 100
            assert manifest['candidate_count'] == 7048
            terminal['manifest_sha256'] = sha(manifest_path)
    except BaseException as exc:
        terminal.update(status='driver_failed', exception_type=type(exc).__name__, exception=str(exc))
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        terminal['seconds'] = time.monotonic()-begin
        terminal['ended_unix'] = time.time()
        terminal['post_gpu'] = helper.gpu_inventory()
        helper.save(root/'driver.terminal.json', terminal)
        print(json.dumps(terminal), flush=True)
    if terminal.get('returncode') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
