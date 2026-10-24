"""One 240+TERM30 probe, fresh resource gate and durable actual terminal."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now(): return datetime.now(timezone.utc).isoformat()


def save(name, value):
    p = Path(name)
    with p.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    parent = p.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent


def resources():
    def query(argument):
        return subprocess.check_output(['nvidia-smi', argument, '--format=csv,noheader,nounits'], text=True, timeout=10)
    return dict(at_utc=now(), gpu=query('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),
                processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'),
                disk_free_bytes=os.statvfs('.').f_bavail * os.statvfs('.').f_frsize)


def main():
    os.umask(0o077)
    plan = json.loads(Path('plan.json').read_text())
    assert plan['budget_seconds'] == 240 and plan['grace_seconds'] == 30
    assert plan['fresh_state_discarded'] and plan['diagnostic_rows'] == 13
    assert all(not Path(x).exists() for x in ['driver-started.json', 'child-started.json', 'wrapper-complete.json', 'started.json'])
    for name, item in plan['remote_files'].items():
        raw = Path(name).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    gate = resources(); save('fresh-resource-gate.json', gate)
    target = [line.split(',') for line in gate['gpu'].splitlines() if plan['gpu_uuid'] in line]
    assert len(target) == 1 and int(target[0][2].strip()) <= 10 and int(target[0][3].strip()) >= 79000
    assert plan['gpu_uuid'] not in gate['processes'] and gate['disk_free_bytes'] >= 512 * 1024**2
    shm=Path(plan['probe_state_directory']).parent
    assert shm==Path('/dev/shm/patientphex-quote-set-completion-probe-v1') and not shm.exists()
    assert os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize>=4*1024**3
    shm.mkdir(); command = [plan['python'], '-u', 'probe.py']
    start = time.monotonic(); utc = time.time()
    save('driver-started.json', dict(at_utc=now(), driver_pid=os.getpid(), command=command,
        plan_sha256=hashlib.sha256(Path('plan.json').read_bytes()).hexdigest(),
        deadline_TERM_unix=utc + 240, deadline_KILL_unix=utc + 270,
        budget_seconds=240, term_grace_seconds=30, no_automatic_retry=True))
    timed_out = False
    with Path('stdout.log').open('xb') as out, Path('stderr.log').open('xb') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        save('child-started.json', dict(at_utc=now(), driver_pid=os.getpid(), child_pid=child.pid))
        try: rc = child.wait(timeout=max(.001, start + 240 - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(child.pid, signal.SIGTERM)
            try: rc = child.wait(timeout=max(.001, start + 270 - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); rc = child.wait()
        for stream in [out, err]: stream.flush(); os.fsync(stream.fileno())
    terminal = dict(at_utc=now(), returncode=rc, timed_out=timed_out, elapsed_seconds=time.monotonic() - start,
        cap_seconds=240, grace_seconds=30, child_pid=child.pid,
        complete_exists=Path('complete.json').exists(), failure_exists=Path('failure.json').exists())
    if rc == 0 and not timed_out:
        try:
            complete=json.loads(Path('complete.json').read_text()); step=json.loads(Path('step.json').read_text())
            assert complete['actual_steps']==1 and complete['actual_backwards']==13 and complete['fresh_state_discarded']
            assert complete['step_sha256']==hashlib.sha256(Path('step.json').read_bytes()).hexdigest()
            assert complete['manifest_sha256']==hashlib.sha256(Path('run-manifest.json').read_bytes()).hexdigest()
            assert step['actual_forwards']==13 and step['actual_backwards']==13 and step['all_two_Pareto_completed']
            assert step['state_save_roundtrip']['all_SHA_fsync_reopen'] and step['state_save_roundtrip']['logical_state_exact'] and step['state_save_roundtrip']['discarded_after_check']
            assert step['final_inventory']['all_grad_finite'] and step['final_inventory']['frozen_base_no_gradient']
            terminal['all13_backwards_and_one_update_verified']=True
        except Exception as error:terminal['endpoint_error']=dict(type=type(error).__name__,message=str(error))
    try:
        actual = resources(); terminal['gpu_release'] = actual
        terminal['target_has_no_process'] = plan['gpu_uuid'] not in actual['processes']
    except Exception as error: terminal['release_unknown'] = type(error).__name__
    save('wrapper-complete.json', terminal)
    print(json.dumps(terminal), flush=True)
    if rc or timed_out or 'endpoint_error' in terminal: raise SystemExit(1)


if __name__ == '__main__': main()
