"""Single local60s fixed fourfold append/evaluation endpoint."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--plan',required=True);q=a.parse_args();p=json.loads(Path(q.plan).read_text());assert p['status']=='frozen_after_complete_input_only_cache_before_quality'
assert sha(__file__)==p['runner_sha256'] and p['CPU_cap_seconds']==60
for k,v in p['environment'].items():assert os.environ.get(k)==v,k
j=Path(p['job_path']);assert not j.exists() and not Path(p['output_dir']).exists()
cmd=['timeout','--signal=TERM','--kill-after=5s','60s',p['python'],str(Path(__file__).with_name('evaluate_control.py')),'--plan',q.plan]
s=dict(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),command=cmd,plan_sha256=sha(q.plan));j.write_text(json.dumps(s,indent=2)+'\n');t=time.monotonic()
try:
 with j.with_name('stdout.log').open('x') as out,j.with_name('stderr.log').open('x') as err:r=subprocess.run(cmd,stdout=out,stderr=err,timeout=70)
 s['returncode']=r.returncode;assert r.returncode==0,r.returncode;s['status']='complete';s['summary_sha256']=sha(Path(p['output_dir'])/'summary.json')
except BaseException as e:s.update(status='failed_preserved',exception=repr(e));raise
finally:s.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-t);j.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
