"""One independent 900+30 native process, no automatic restart."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
from common import ref, save, sha, stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for name, digest in plan['source_files'].items():
        assert sha(Path(__file__).parent / name) == digest
    root = Path(plan['run_root'])
    root.mkdir(exist_ok=False)
    now = datetime.now(timezone.utc)
    tick = time.monotonic()
    deadline = tick + 900
    env = dict(os.environ)
    env.update(plan['environment'])
    env['CUDA_VISIBLE_DEVICES'] = plan['gpu_uuid']
    command = [plan['python'], str(Path(__file__).parent / 'f0_infer.py'), '--plan', str(Path(args.plan).resolve())]
    with (root / 'stdout.log').open('xb') as out, (root / 'stderr.log').open('xb') as err:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 env=env, start_new_session=True)
        start = {'at_utc': now.isoformat(), 'driver_pid': os.getpid(), 'child_pid': child.pid,
                 'TERM_at_utc': (now + timedelta(seconds=900)).isoformat(),
                 'KILL_at_utc': (now + timedelta(seconds=930)).isoformat(), 'plan': ref(args.plan)}
        save(root / 'child-start.json', start)
        timed_out = False
        try:
            child.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=max(.001, deadline + 30 - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    release = {'at_utc': stamp(), 'target_gpu': [x for x in gpu.splitlines() if plan['gpu_uuid'] in x],
               'target_apps': [x for x in apps.splitlines() if plan['gpu_uuid'] in x]}
    result = {'status': 'COMPLETE_FIXED_F0' if child.returncode == 0 and not timed_out and (root / 'native/complete.json').exists() else 'FAILED_OR_TIMEOUT_NO_RESTART',
              'return_code': child.returncode, 'timed_out': timed_out, 'seconds': time.monotonic() - tick,
              'release': release, 'at_utc': stamp()}
    save(root / 'driver-result.json', result)
    files = {}
    for directory, prefix in [(root, 'run'), (Path(__file__).parent, 'source')]:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.jsonl', '.log'} and path.stat().st_size < 1048576:
                files[prefix + '/' + str(path.relative_to(directory))] = path
    for name, entry in plan['terminal_inputs'].items():
        path = Path(entry['path'])
        assert ref(path) == entry
        files['inputs/' + name] = path
    files['plan.json'] = Path(args.plan)
    inventory = {n: ref(p) for n, p in files.items()}
    save(root / 'inventory.json', inventory)
    with zipfile.ZipFile(root / 'terminal.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for n, p in files.items():
            z.write(p, n)
        z.write(root / 'inventory.json', 'inventory.json')
    save(root / 'terminal-receipt.json', {'archive': ref(root / 'terminal.zip'), 'members': len(files) + 1})


if __name__ == '__main__':
    main()
