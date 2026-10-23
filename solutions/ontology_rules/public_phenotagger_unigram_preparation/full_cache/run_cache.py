"""Single isolated480s CPU probe; preserve failure and never overwrite prior output."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(Path('plan.json').read_text());assert sha('cache.py')==p['source_sha256'];assert sha(__file__)==p['runner_sha256']
assert p['CPU_cap_seconds']==480 and p['TERM_grace_seconds']==5
for k,v in p['runtime']['environment'].items():assert os.environ.get(k)==v,k
assert not Path('run-state.json').exists() and not Path('output').exists()
argv=['timeout','--signal=TERM','--kill-after=5s','480s',sys.executable,'cache.py']
s={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'command':argv,'plan_sha256':sha('plan.json'),'source_sha256':sha('cache.py'),'runner_sha256':sha(__file__)}
Path('run-state.json').write_text(json.dumps(s,indent=2)+'\n');t=time.monotonic()
try:
 with Path('stdout.log').open('x') as out,Path('stderr.log').open('x') as err:r=subprocess.run(argv,stdout=out,stderr=err,timeout=490)
 s['returncode']=r.returncode;assert r.returncode==0,r.returncode
 s['status']='complete';s['summary_sha256']=sha('output/summary.json')
except BaseException as e:s.update(status='failed_preserved',exception=repr(e));raise
finally:
 s['ended_at_utc']=datetime.now(timezone.utc).isoformat();s['seconds']=time.monotonic()-t;Path('run-state.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)
