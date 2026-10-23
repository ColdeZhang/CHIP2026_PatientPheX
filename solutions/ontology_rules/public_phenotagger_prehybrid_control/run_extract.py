"""One fixed CPU census with an explicit wall-clock cap and no overwrite."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


p = json.loads(Path('plan.json').read_text())
assert sha(__file__) == p['runner_sha256'] and sha('extract_raw95.py') == p['source_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not Path('run-state.json').exists() and not Path('output').exists()
assert p['runtime']['CPU_hardcap_seconds'] == 120 and p['runtime']['TERM_grace_seconds'] == 5
command = ['timeout', '--signal=TERM', '--kill-after=5s', '120s', sys.executable, 'extract_raw95.py']
state = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
             plan_sha256=sha('plan.json'), source_sha256=sha('extract_raw95.py'), CPU_cap_seconds=120, TERM_grace_seconds=5)
Path('run-state.json').write_text(json.dumps(state, indent=2) + '\n')
started = time.monotonic()
try:
    with open('stdout.log', 'x') as out, open('stderr.log', 'x') as err:
        result = subprocess.run(command, stdout=out, stderr=err, timeout=130)
    state['returncode'] = result.returncode
    assert result.returncode == 0, result.returncode
    state['summary_sha256'] = sha('output/manifest.json')
    state['status'] = 'complete'
finally:
    state['seconds'] = time.monotonic() - started
    state['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
    if state['status'] != 'complete':
        state['status'] = 'failed_preserved'
    Path('run-state.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)
