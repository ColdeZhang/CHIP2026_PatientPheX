"""One explicit CPU child with timeout120+TERM5; no remote or model entry."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time


def save(p, value):
    with Path(p).open('x') as h:
        json.dump(value, h, indent=2); h.write('\n'); h.flush(); os.fsync(h.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args(); p = json.loads(a.plan.read_text())
    assert p['CPU_only'] and p['cap_seconds'] == 120 and p['grace_seconds'] == 5
    assert not p['GPU_authorized'] and not p['training_authorized']
    assert os.path.realpath(sys.executable) == os.path.realpath(p['python'])
    for name, item in p['files'].items():
        b = (a.plan.parent/name).read_bytes()
        assert len(b) == item['bytes'] and hashlib.sha256(b).hexdigest() == item['sha256']
    env = os.environ.copy(); env.update(p['environment']); env['CUDA_VISIBLE_DEVICES'] = ''
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '120s', p['python'], '-u',
        str(a.plan.parent/'native_cpu.py'), '--plan', str(a.plan.resolve()),
        '--base', p['base'], '--output', str(a.output.resolve())]
    save(a.output/'driver-started.json', {'at_epoch': started, 'deadline_unix': started+120,
        'pid': os.getpid(), 'command': command, 'cap_seconds': 120, 'grace_seconds': 5,
        'CUDA_VISIBLE_DEVICES': '', 'plan_sha256': hashlib.sha256(a.plan.read_bytes()).hexdigest()})
    with (a.output/'stdout.log').open('xb') as out, (a.output/'stderr.log').open('xb') as err:
        child = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=True)
        save(a.output/'child-started.json', {'pid': child.pid, 'pgid': child.pid})
        rc = child.wait()
    save(a.output/'wrapper-complete.json', {'returncode': rc, 'seconds': time.time()-started,
        'at_epoch': time.time(), 'cap_seconds': 120, 'grace_seconds': 5,
        'native_complete': (a.output/'native-result.json').is_file()})
    raise SystemExit(rc)


if __name__ == '__main__': main()
