"""One fixed common deadline for probe, formal training, inference and linking."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import write_json, ref


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text()); out = Path(plan['output']); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'driver-started.json').exists(), 'Already started; recover original process'
    started = time.monotonic(); wall = time.time(); deadline = started + 600
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True)
    line = [x for x in gpu.splitlines() if plan['gpu_uuid'] in x]; assert len(line) == 1
    used = int(line[0].split(',')[2].strip()); assert used <= 64
    assert not any(plan['gpu_uuid'] in x for x in apps.splitlines())
    disk = shutil.disk_usage(out); assert disk.free >= 5*1024**3
    write_json(out/'driver-started.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'driver_pid': os.getpid(),
              'plan': ref(a.plan), 'deadline_TERM_unix': wall+600, 'deadline_KILL_unix': wall+630,
              'budget_seconds': 600, 'includes_probe': True, 'fresh_gpu': gpu, 'fresh_apps': apps,
              'free_bytes': disk.free, 'no_automatic_retry': True})
    results = []; status = 'COMPLETE'; child = None
    for name, command in plan['commands']:
        before = time.monotonic()
        with (out/(name+'.stdout')).open('wb') as stdout, (out/(name+'.stderr')).open('wb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            write_json(out/'child.json', {'name': name, 'pid': child.pid, 'command': command})
            timed_out = False
            try:
                rc = child.wait(timeout=max(0.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(child.pid, signal.SIGTERM)
                try: rc = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); rc = child.wait()
        results.append({'name': name, 'returncode': rc, 'timed_out': timed_out, 'seconds': time.monotonic()-before,
                        'stdout': ref(out/(name+'.stdout')), 'stderr': ref(out/(name+'.stderr'))})
        if rc or timed_out: status = 'FAILED'; break
    write_json(out/'terminal.json', {'status': status, 'seconds': time.monotonic()-started,
                                   'steps': results, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'status': status, 'seconds': time.monotonic()-started}), flush=True)
    if status != 'COMPLETE': sys.exit(1)


if __name__ == '__main__': main()
