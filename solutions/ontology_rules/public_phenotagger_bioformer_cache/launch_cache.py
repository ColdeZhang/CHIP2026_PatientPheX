"""Bounded public Bioformer train64 forward and CPU provenance; no labels or quality."""
import os,sys,json,time,hashlib,subprocess,datetime
from pathlib import Path
R=Path.cwd();SHARED=R.parent/'public-phenotagger-reuse';D=R/'train64-cache';U='GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def query(args):
 p=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,timeout=20,check=True);return p.stdout.strip()
def gpu_state():
 return {'gpu':query(['-i',U,'--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits']),'compute_apps':query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']),'at':now()}
state={'started_at':now(),'scope':'independent public Bioformer train64 all-new forward then CPU provenance','plan_sha256':sha(D/'plan.json'),'wrapper_sha256':sha(__file__),'GPU_cap_seconds':600,'TERM_grace_seconds':30,'CPU_union_cap_seconds':120}
def save():(D/'run-state.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 assert state['plan_sha256']=='73f7fadc504b08949e8a345ca564fa6c704d626fbda6a6a2adf580dfd215c485'
 plan=json.loads((D/'plan.json').read_text())
 for n,h in plan['remote_source_sha256'].items():assert sha(R/n)==h,n
 assert sha(D/'train64.input.jsonl')==plan['input64_sha256']
 assert sha(D/'train64.input.jsonl')==plan['input64_sha256']
 prep=json.loads((R/'native-train64-cache-cpu-preparation/preparation.json').read_text());assert prep['documents']==64 and prep['input_sha256']==plan['input64_sha256']
 assert prep['reused_forward_keys']==0
 for k,v in plan['expected_input_only_counts'].items():assert prep[k]==v,(k,prep[k],v)
 state['preparation']=prep;state['preparation_sha256']=sha(R/'native-train64-cache-cpu-preparation/preparation.json')
 # Actual previous throughput plus CPU preparation and conservative fixed setup/postprocess allowance.
 projected=prep['new_forward_keys']/1425.5+prep['seconds']+60
 assert projected<600,projected
 state['estimated_seconds']=projected;state['estimate_basis']='conservative prior PubMed NN1425.5/s plus actual64 CPU prep+60s; Bioformer GPU speed not yet measured; hard600s and first real batch gate';save()
 fresh=gpu_state();fields=[x.strip() for x in fresh['gpu'].split(',')]
 assert len(fields)==4 and fields[0]=='2' and fields[1]==U and int(fields[2])<1024,fresh
 assert not any(line.split(',')[0].strip()==U for line in fresh['compute_apps'].splitlines()),fresh
 state['fresh_gpu_gate']=fresh;state['gpu_started_at']=now();save()
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=U,PYTHONPATH=str(SHARED/'tf-gpu-overlay'),LD_LIBRARY_PATH=str(SHARED/'tf-gpu-overlay/nvidia/cudnn/lib')+':/usr/local/cuda-11.8/lib64',FORCE_TF_AVAILABLE='1',USE_TF='1',USE_TORCH='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
 command=['timeout','--signal=TERM','--kill-after=30s','600s',sys.executable,str(D/'run_train64.py')]
 state['gpu_command']=command;state['gpu_environment']=env={k:v for k,v in env.items() if k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','LD_LIBRARY_PATH','FORCE_TF_AVAILABLE','USE_TF','USE_TORCH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','TF_CPP_MIN_LOG_LEVEL']};save()
 childenv=dict(os.environ);childenv.update(env);s=time.monotonic()
 with (D/'native.stdout').open('w') as out,(D/'native.stderr').open('w') as err:
  p=subprocess.run(command,stdout=out,stderr=err,env=childenv,timeout=645)
 state['gpu_returncode']=p.returncode;state['gpu_elapsed_seconds']=time.monotonic()-s;state['gpu_finished_at']=now();state['release_after_gpu']=gpu_state();save();assert p.returncode==0,p.returncode
 state['native_manifest_sha256']=sha(R/'native-train64-cache/manifest.json');save()
 cpuenv=dict(os.environ);cpuenv.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH='');s=time.monotonic();state['union_started_at']=now();save()
 with (D/'provenance.stdout').open('w') as out,(D/'provenance.stderr').open('w') as err:
  p=subprocess.run(['timeout','--signal=TERM','--kill-after=10s','120s',sys.executable,str(D/'make_provenance.py')],stdout=out,stderr=err,env=cpuenv,timeout=135)
 state['union_returncode']=p.returncode;state['union_elapsed_seconds']=time.monotonic()-s;state['union_finished_at']=now();save();assert p.returncode==0,p.returncode
 state['final_manifest_sha256']=sha(R/'native-train64-final-cache/manifest.json');state['status']='complete_cache_without_quality_evaluation';state['finished_at']=now();save();print(json.dumps(state),flush=True)
except BaseException as e:
 state['status']='failed_preserved';state['failure']=repr(e);state['finished_at']=now();save();raise
