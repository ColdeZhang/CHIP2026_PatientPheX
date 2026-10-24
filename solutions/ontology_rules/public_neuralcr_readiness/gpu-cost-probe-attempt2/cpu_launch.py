from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';assert not(R/'CPU-policy-job.json').exists();start=time.monotonic()
job={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256((R/'check_policy_cpu.py').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),'cap_seconds':30,'TERM_grace_seconds':5,'command':['timeout','--signal=TERM','--kill-after=5s','30s',sys.executable,'check_policy_cpu.py']};(R/'CPU-policy-job.json').write_text(json.dumps(job)+'\n')
with (R/'CPU-policy.stdout.log').open('xb') as out,(R/'CPU-policy.stderr.log').open('xb') as err:p=subprocess.run(job['command'],stdout=out,stderr=err,timeout=40)
job.update(returncode=p.returncode,elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(R/'CPU-policy-job.json').write_text(json.dumps(job,indent=2)+'\n');print(json.dumps(job));raise SystemExit(p.returncode)
