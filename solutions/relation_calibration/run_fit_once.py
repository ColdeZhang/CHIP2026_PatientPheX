"""One CPU child with an absolute 30s deadline and 5s TERM grace."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import checked, fsync_dir, mkdir_new, now, ref, write_json

p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt-dir',required=True);a=p.parse_args()
plan=json.loads(Path(a.plan).read_text());checked(plan['sources']['fit_once.py'])
assert not Path(a.output).exists()
out=mkdir_new(a.receipt_dir)
command=[sys.executable,plan['sources']['fit_once.py']['path'],'--plan',a.plan,'--output',a.output]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
started=time.monotonic();write_json(out/'intent.json',{'utc':now(),'command':command,'plan':ref(a.plan),'source':ref(__file__),'cap_seconds':30,'TERM_grace_seconds':5,'unique_fit_output':a.output,'environment_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}})
timed_out=False
with (out/'stdout').open('xb') as stdout,(out/'stderr').open('xb') as stderr:
    child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    write_json(out/'child.json',{'utc':now(),'pid':child.pid})
    try:rc=child.wait(timeout=max(.001,started+30-time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(child.pid,signal.SIGTERM)
        try:rc=child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    for f in (stdout,stderr):f.flush();os.fsync(f.fileno())
fsync_dir(out)
result=Path(a.output)/'result.json'
write_json(out/'terminal.json',{'utc':now(),'returncode':rc,'timed_out':timed_out,'seconds':time.monotonic()-started,'stdout':ref(out/'stdout'),'stderr':ref(out/'stderr'),'result':ref(result) if result.is_file() else None,'status':'COMPLETE' if rc==0 and not timed_out and result.is_file() else 'FAILED'})
sys.exit(0 if rc==0 and not timed_out and result.is_file() else 1)
