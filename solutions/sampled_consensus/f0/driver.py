"""Once-only native driver; fresh GPU gate and one absolute TERM/KILL budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def gpu_state(uuid):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
    rows = [line for line in gpu.splitlines() if line.split(',')[1].strip() == uuid]
    assert len(rows) == 1
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    return {'at_utc': datetime.now(timezone.utc).isoformat(), 'gpu': rows[0],
            'target_compute_rows': [line for line in apps.splitlines() if line.split(',')[0].strip() == uuid]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    start = time.time()
    budget = int(plan['cap_seconds']); assert budget == 2100 and plan['term_grace_seconds'] == 30
    deadline = begin + budget
    write(output / 'driver-start.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'driver_pid': os.getpid(),
          'deadline_TERM_unix': start + budget, 'deadline_KILL_unix': start + budget + 30,
          'budget_seconds': budget, 'grace_seconds': 30, 'plan_sha256': sha(args.plan)})
    child = None
    rc = None
    error = None
    timed_out = False
    try:
        for path, expected in plan['files'].items():
            assert sha(path) == expected, path
        free_bytes = shutil.disk_usage(output).free
        assert free_bytes >= 128 * 1024 ** 2, 'Insufficient output disk space'
        fresh = gpu_state(plan['gpu_uuid'])
        fresh['output_disk_free_bytes'] = free_bytes
        write(output / 'fresh-gpu.json', fresh)
        assert not fresh['target_compute_rows'] and int(fresh['gpu'].split(',')[2].strip()) <= 100
        assert int(fresh['gpu'].split(',')[3].strip()) >= 23500 and int(fresh['gpu'].split(',')[4].strip()) == 24576
        env = os.environ.copy()
        env.update(plan['env'])
        env['CUDA_VISIBLE_DEVICES'] = plan['gpu_uuid']
        command = [plan['python'], '-u', str(Path(plan['source']) / 'infer.py'), '--plan', args.plan, '--output', str(output / 'native')]
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            write(output / 'child-start.json', {'pid': child.pid, 'command': command, 'at_utc': datetime.now(timezone.utc).isoformat()})
            try:
                rc = child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait()
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                rc = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                rc = child.wait()
    finally:
        write(output / 'driver-result.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
              'return_code': rc, 'error': error, 'timed_out': timed_out,
              'elapsed_seconds': time.monotonic() - begin, 'plan_sha256': sha(args.plan)})
        write(output / 'released-gpu.json', gpu_state(plan['gpu_uuid']))
    if error or timed_out or rc != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
