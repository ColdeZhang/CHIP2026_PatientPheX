"""Deploy only the constrained small persistent fetcher; no background process."""
from pathlib import Path
import sys,json,hashlib,shlex,datetime,os
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
T=Path(__file__).resolve().parent;assert not (T/'fetcher-deployment-intent.json').exists()
path='research/medgemma-relation-control/fullcontext-formal-preparation-v1/fetch_checkpoint.py';digest=hashlib.sha256((T/'fetch_checkpoint.py').read_bytes()).hexdigest();io.M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def save(n,v):
 with (T/n).open('x') as h:h.write(json.dumps(v,indent=2)+'\n');h.flush();os.fsync(h.fileno())
save('fetcher-deployment-intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mount=io.M,path=path,sha256=digest,no_job=True))
pre=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(path)+').exists()'),'cwd':'.','timeout_seconds':15});assert pre['exit_code']==0
io.upload(str(T/'fetch_checkpoint.py'),path)
r=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;p=Path('+repr(path)+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(digest)+';print(json.dumps(dict(path=str(p),sha256='+repr(digest)+',bytes=p.stat().st_size)))'),'cwd':'.','timeout_seconds':15});assert r['exit_code']==0
save('fetcher-deployment-complete.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual=r,no_jobs_started=True));print('FULLCONTEXT_PERSISTENT_FETCHER_DEPLOYED')
