"""One owned fresh-base indexed-head local/pressure probe with a fixed timeout and fresh GPU guard."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open('.', os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


started = time.monotonic()
plan = json.loads(Path('plan.json').read_text())
assert os.getuid() == 1045
assert plan['cap_seconds'] == 420 and plan['term_grace_seconds'] == 30
assert not Path('driver-started.json').exists()
for name, record in plan['files'].items():
    assert Path(name).stat().st_size == record['bytes'] and sha(name) == record['sha256']
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,memory.total',
                              '--format=csv,noheader,nounits'], text=True)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                               '--format=csv,noheader,nounits'], text=True)
target = [line for line in gpu.splitlines() if line.split(',')[1].strip() == plan['gpu_uuid']]
assert len(target) == 1
index, uuid, used, free, total = [part.strip() for part in target[0].split(',')]
assert index == '2' and int(used) <= 100 and int(free) >= 78000 and int(total) == 81920
assert not any(line.split(',')[0].strip() == plan['gpu_uuid'] for line in apps.splitlines() if line.strip())
assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
assert os.statvfs('.').f_bavail * os.statvfs('.').f_frsize >= 30_000_000
save('fresh-gpu.json', {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'target_csv': target[0], 'target_compute_processes': [],
                      'plan_sha256': sha('plan.json')})
begin = datetime.datetime.now(datetime.timezone.utc)
save('driver-started.json', {'at_utc': begin.isoformat(), 'pid': os.getpid(),
    'cap_seconds': 420, 'term_grace_seconds': 30, 'plan_sha256': sha('plan.json'),
    'source_sha256': sha(__file__), 'deadline_utc': (begin + datetime.timedelta(seconds=420)).isoformat()})
with Path('stdout.log').open('x') as out, Path('stderr.log').open('x') as err:
    child = subprocess.Popen(['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', '420s',
                              plan['python'], '-u', 'probe.py'], stdout=out, stderr=err,
                             start_new_session=True)
    save('child-started.json', {'pid': child.pid, 'pgid': child.pid, 'argv_role': 'timeout_then_runner'})
    rc = child.wait()
save('wrapper-complete.json', {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'returncode': rc, 'cap_seconds': 420, 'term_grace_seconds': 30, 'wrapper_seconds': time.monotonic() - started,
    'plan_sha256': sha('plan.json'), 'source_sha256': sha(__file__)})
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,memory.total',
                              '--format=csv,noheader,nounits'], text=True)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                               '--format=csv,noheader,nounits'], text=True)
save('gpu-release.json', {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'target_csv': [line for line in gpu.splitlines() if line.split(',')[1].strip() == plan['gpu_uuid']],
    'target_compute_rows': [line for line in apps.splitlines() if line.split(',')[0].strip() == plan['gpu_uuid']]})
raise SystemExit(rc)
