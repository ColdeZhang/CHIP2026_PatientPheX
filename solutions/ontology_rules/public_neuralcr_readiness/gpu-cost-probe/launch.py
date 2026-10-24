"""Fresh GPU2 gate, one bounded child, actual terminal and release evidence."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'run-state.json').exists()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def query(args):return subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,timeout=15,check=True).stdout.strip()
def gpu_state():return {'at_utc':now(),'gpu':query(['-i','2','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv,noheader,nounits']),'compute_apps':query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])}
state={'status':'preparing','started_at':now(),'source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'cap_seconds':180,'TERM_grace_seconds':30}
def save():(R/'run-state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for n,digest in plan['deployed_sources'].items():assert sha(R/n)==digest,n
 for key,item in plan['required_files'].items():assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],key
 fresh=gpu_state();f=[x.strip() for x in fresh['gpu'].split(',')];assert f[0]=='2' and f[1]==plan['gpu_uuid'] and int(f[3])<1024,fresh
 assert not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in fresh['compute_apps'].splitlines()),fresh
 state.update(fresh_gpu_gate=fresh,status='running',gpu_started_at=now());save()
 env={**os.environ,**plan['GPU_environment']};command=['timeout','--signal=TERM','--kill-after=30s','180s',sys.executable,'probe.py'];state['command']=command;state['GPU_environment']=plan['GPU_environment'];save();begin=time.monotonic()
 with (R/'probe.stdout.log').open('xb') as out,(R/'probe.stderr.log').open('xb') as err:
  result=subprocess.run(command,cwd=R,env=env,stdout=out,stderr=err,timeout=225)
 state.update(returncode=result.returncode,gpu_elapsed_seconds=time.monotonic()-begin,gpu_finished_at=now());state['release']=gpu_state();save()
 assert not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in state['release']['compute_apps'].splitlines()),state['release']
 assert result.returncode==0,result.returncode
 state.update(status='complete_gpu_released',result_sha256=sha(R/'result.json'),finished_at=now());save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),finished_at=now());save();raise
