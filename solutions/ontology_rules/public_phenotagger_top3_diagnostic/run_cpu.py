"""One bounded CPU diagnostic, no model import/GPU/fit."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

r=Path.cwd();assert not (r/'run-state.json').exists()
p=json.loads((r/'plan.json').read_text())
assert sha(__file__)==p['runner_sha256']
assert sha(r/'analyze_cached_top3.py')==p['source_sha256']
assert os.environ['CUDA_VISIBLE_DEVICES']==''
command=['timeout','--signal=TERM','--kill-after=5s','120s',sys.executable,'analyze_cached_top3.py','--plan','plan.json']
state=dict(status='running',started_at_utc=now(),command=command,source_sha256=p['source_sha256'],
    runner_sha256=sha(__file__),plan_sha256=sha(r/'plan.json'),CPU_cap_seconds=120,TERM_grace_seconds=5)
(r/'run-state.json').write_text(json.dumps(state,indent=2)+'\n');started=time.monotonic()
try:
    with (r/'stdout.log').open('x') as out,(r/'stderr.log').open('x') as err:
        q=subprocess.run(command,stdout=out,stderr=err,timeout=130)
    state['returncode']=q.returncode;assert q.returncode==0,q.returncode
    state['status']='complete';state['summary_sha256']=sha(r/'output/summary.json')
except BaseException as exc:
    state['status']='failed_preserved';state['failure']=repr(exc);raise
finally:
    state['ended_at_utc']=now();state['seconds']=time.monotonic()-started
    (r/'run-state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
