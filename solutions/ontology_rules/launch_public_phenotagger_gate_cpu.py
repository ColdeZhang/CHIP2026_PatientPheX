"""Run the frozen two-point census once with a CPU-only wall-clock cap."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
assert not (root/'run-state.json').exists()
plan = json.loads((root/'plan.json').read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert sha(root/'plan.json') == '1b8ab8ffe47112ec35562954e477dbb96ca4723f98060292c8ede8a1af1511b9'
assert sha(root/'measure_public_phenotagger_gate.py') == plan['source_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
command = ['timeout', '--signal=TERM', '--kill-after=10s', '180s', sys.executable,
           'measure_public_phenotagger_gate.py', '--plan', 'plan.json']
state = {'started_at': now(), 'wrapper_sha256': sha(__file__),
         'plan_sha256': sha(root/'plan.json'), 'source_sha256': plan['source_sha256'],
         'command': command, 'environment': plan['runtime']['environment'],
         'CPU_cap_seconds': 180, 'TERM_grace_seconds': 10, 'status': 'running'}
def save():
    (root/'run-state.json').write_text(json.dumps(state, indent=2)+'\n')
save()
started = time.monotonic()
try:
    env = dict(os.environ)
    env.update(plan['runtime']['environment'])
    with (root/'stdout.txt').open('x') as out, (root/'stderr.txt').open('x') as err:
        child = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=195)
    state['returncode'] = child.returncode
    assert child.returncode == 0, child.returncode
    state['summary_sha256'] = sha(root/'output/summary.json')
    state['status'] = 'complete_CPU_only_no_quality'
except BaseException as exc:
    state['status'] = 'failed_preserved'
    state['failure'] = repr(exc)
    raise
finally:
    state['elapsed_seconds'] = time.monotonic()-started
    state['finished_at'] = now()
    save()
    print(json.dumps(state), flush=True)
