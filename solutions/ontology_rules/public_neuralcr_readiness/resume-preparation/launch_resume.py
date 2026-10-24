"""Own-process timeout wrapper with actual exit status; no model execution."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
plan = json.loads((R / 'resume-plan.json').read_text())
assert plan['status'] == 'authorized_frozen_resume_after_attempt1_exit'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not (R / 'resume-job-state.json').exists()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['wrapper_sha256']
command = ['timeout', '--signal=TERM', '--kill-after=30s', str(plan['budget_seconds']) + 's',
           sys.executable, str(R / 'resume_assets.py')]
state = {'status': 'running', 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'plan_sha256': hashlib.sha256((R / 'resume-plan.json').read_bytes()).hexdigest(),
         'wrapper_sha256': plan['wrapper_sha256'], 'command': command}
path = R / 'resume-job-state.json'
path.write_text(json.dumps(state, indent=2) + '\n')
started = time.monotonic()
with (R / 'stdout.log').open('xb') as stdout, (R / 'stderr.log').open('xb') as stderr:
    child = subprocess.Popen(command, cwd=R, stdout=stdout, stderr=stderr)
    state['child_pid'] = child.pid
    path.write_text(json.dumps(state, indent=2) + '\n')
    result = child.wait()
state.update(status='complete' if result == 0 else 'failed_preserved', returncode=result,
             elapsed_seconds=time.monotonic() - started,
             finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state), flush=True)
