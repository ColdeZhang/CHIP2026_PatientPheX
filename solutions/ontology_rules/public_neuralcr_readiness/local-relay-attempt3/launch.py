from pathlib import Path
import datetime,json,hashlib,subprocess,sys,time
R=Path(__file__).resolve().parent
assert not (R/'job.json').exists()
p=json.loads((R/'plan.json').read_text());t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc)
x={'status':'running','started_at':start.isoformat(),'deadline':(start+datetime.timedelta(seconds=3600)).isoformat(),'hard_kill_at':(start+datetime.timedelta(seconds=3630)).isoformat(),'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),'command':['timeout','--signal=TERM','--kill-after=30s','3600s',sys.executable,str(R/'relay.py')]}
(R/'job.json').write_text(json.dumps(x,indent=2)+'\n')
with (R/'stdout.log').open('w') as out,(R/'stderr.log').open('w') as err:
 child=subprocess.Popen(x['command'],stdout=out,stderr=err);x['pid']=child.pid;(R/'job.json').write_text(json.dumps(x,indent=2)+'\n');rc=child.wait()
x.update(status='completed' if rc==0 else 'incomplete_preserved',returncode=rc,elapsed_seconds=time.monotonic()-t,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(R/'job.json').write_text(json.dumps(x,indent=2)+'\n')
