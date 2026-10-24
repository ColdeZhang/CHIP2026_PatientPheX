"""One absolute-budget sequential fixed-public B run. Never fits or scores."""
from pathlib import Path
import os,sys,time,json,hashlib,subprocess,datetime,shutil,traceback
D=Path(__file__).resolve().parent;P=D.parent;SHARED=P/'public-phenotagger-reuse';PY=SHARED/'.venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def gpu():
 q=lambda args:subprocess.check_output(['nvidia-smi']+args,text=True,timeout=15).strip()
 return dict(at=now(),gpu=q(['-i',U,'--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits']),apps=q(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']))
def empty(g):
 row=[s.strip() for s in g['gpu'].split(',')];assert row[0]=='2' and row[1]==U and int(row[2])<1024 and int(row[3])>35000,g
 assert not any(x.split(',')[0].strip()==U for x in g['apps'].splitlines()),g
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
plan=json.loads((D/'batch-plan.json').read_text());U=plan['gpu_uuid'];assert sha(__file__)==plan['driver_sha256'];assert plan['total_cap_seconds']==2400
with (D/'driver-started.json').open('x') as f:json.dump(dict(at=now(),pid=os.getpid(),batch_plan_sha256=sha(D/'batch-plan.json')),f)
start=time.monotonic();deadline=start+plan['total_cap_seconds'];state=dict(started_at=now(),pid=os.getpid(),batch_plan_sha256=sha(D/'batch-plan.json'),total_cap_seconds=2400,TERM_grace_seconds=30,phases=[])
write(D/'run-state.json',state)
try:
 assert shutil.disk_usage(P).free>10*1024**3
 for item in plan['models']:
  root=P/item['remote_root'];sub=root/'B901-cache';assert sha(sub/'plan.json')==item['plan_sha256'];mp=json.loads((sub/'plan.json').read_text())
  for name,digest in mp['remote_source_sha256'].items():assert sha(root/name)==digest,(item['model'],name)
  assert sha(sub/'B.input.jsonl')==plan['input_sha256'];assert sha(root/item['weight_file'])==mp['model_weight_sha256']
 state['asset_validation_complete_at']=now();state['fresh_gpu']=gpu();empty(state['fresh_gpu']);write(D/'run-state.json',state)
 for item in plan['models']:
  root=P/item['remote_root'];sub=root/'B901-cache';assert not (root/'native-B901-cache').exists();assert not (root/'native-B901-final-cache').exists()
  g=gpu();empty(g)
  for mode,filename in [('native','run_B.py'),('provenance','make_provenance.py')]:
   remaining=deadline-time.monotonic();assert remaining>0
   env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=U if mode=='native' else '',PYTHONPATH=str(SHARED/'tf-gpu-overlay') if mode=='native' else '',LD_LIBRARY_PATH=str(SHARED/'tf-gpu-overlay/nvidia/cudnn/lib')+':/usr/local/cuda-11.8/lib64',FORCE_TF_AVAILABLE='1',USE_TF='1',USE_TORCH='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
   cmd=['timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s',str(PY),str(sub/filename)]
   phase=dict(model=item['model'],mode=mode,started_at=now(),remaining_at_start=remaining,command=cmd);state['phases'].append(phase);write(D/'run-state.json',state);t=time.monotonic()
   with (sub/(mode+'.stdout')).open('w') as out,(sub/(mode+'.stderr')).open('w') as err:p=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err,timeout=remaining+35)
   phase.update(finished_at=now(),elapsed_seconds=time.monotonic()-t,returncode=p.returncode);write(D/'run-state.json',state);assert p.returncode==0,phase
  native=root/'native-B901-cache';final=root/'native-B901-final-cache'
  for folder in [native,final]:
   mf=json.loads((folder/'manifest.json').read_text())
   for n,ref in mf['outputs'].items():assert sha(folder/n)==ref['sha256'] and (folder/n).stat().st_size==ref['bytes'],(folder,n)
  state.setdefault('completed_models',[]).append(dict(model=item['model'],native_manifest_sha256=sha(native/'manifest.json'),final_manifest_sha256=sha(final/'manifest.json'),counts=json.loads((final/'manifest.json').read_text())['counts']));write(D/'run-state.json',state)
 state['status']='complete_three_public_fixed_B_no_quality';state['returncode']=0
except BaseException as e:
 state['status']='failed_preserved';state['failure']=repr(e);state['traceback']=traceback.format_exc();state['returncode']=1
finally:
 state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);state['release']=gpu();write(D/'run-state.json',state);print(json.dumps(state),flush=True)
sys.exit(state['returncode'])
