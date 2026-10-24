"""One 894s pure-input cache process with actual terminal and physical GPU release."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';plan=json.loads((R/'plan.json').read_text());assert not(R/'run-state.json').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def query(args):return subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,timeout=15,check=True).stdout.strip()
def gpu():return {'at_utc':now(),'gpu':query(['-i','2','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits']),'compute_apps':query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])}
s={'status':'preparing','started_at':now(),'source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'bound_plan_sha256':sha(R/'bound-plan.json'),'cap_seconds':894,'TERM_grace_seconds':30}
def save():(R/'run-state.json').write_text(json.dumps(s,indent=2)+'\n')
save()
try:
 for n,digest in plan['deployed_sources'].items():assert sha(R/n)==digest,n
 bound=json.loads((R/'bound-plan.json').read_text());assert bound['plan_sha256']==sha(R/'plan.json') and bound['preparation_sha256']==sha(R/'preparation.json')
 prep=json.loads((R/'preparation.json').read_text());assert prep['status']=='complete_input_only_preparation'
 for item in [*plan['model_files'].values(),plan['input'],*prep['artifacts'].values()]:assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
 fresh=gpu();fields=[x.strip() for x in fresh['gpu'].split(',')];assert fields[0]=='2' and fields[1]==plan['gpu_uuid'] and int(fields[2])<1024
 assert not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in fresh['compute_apps'].splitlines()),fresh
 s.update(status='running',fresh_gpu_gate=fresh,gpu_started_at=now());env={**os.environ,**plan['GPU_environment']};command=['timeout','--signal=TERM','--kill-after=30s','894s',sys.executable,'run_cache.py'];s.update(command=command,GPU_environment=plan['GPU_environment']);save();begin=time.monotonic()
 with (R/'cache.stdout.log').open('xb') as out,(R/'cache.stderr.log').open('xb') as err:p=subprocess.run(command,cwd=R,env=env,stdout=out,stderr=err,timeout=945)
 s.update(returncode=p.returncode,gpu_elapsed_seconds=time.monotonic()-begin,gpu_finished_at=now(),release=gpu());save()
 assert not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in s['release']['compute_apps'].splitlines()),s['release']
 assert p.returncode==0,p.returncode;result=json.loads((R/'cache-result.json').read_text());assert result['status']=='complete_all_keys_top1_and_native_raw_span_cache_no_quality'
 s.update(status='complete_cache_GPU_released',result_sha256=sha(R/'cache-result.json'),finished_at=now());save()
except BaseException as e:
 s.update(status='failed_preserved',error=repr(e),finished_at=now())
 try:s['failure_release_observation']=gpu()
 except Exception as q:s['failure_release_observation_error']=repr(q)
 save();raise
