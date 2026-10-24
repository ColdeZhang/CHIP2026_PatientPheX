from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time,zipfile,sys
D=Path.cwd();P=D.parent;plan=json.loads((D/'recovery-plan.json').read_text());old=json.loads((P/'state.json').read_text());deadline=old['deadline_epoch'];assert deadline==plan['original_deadline_epoch']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def gpu():
 q=lambda a:subprocess.check_output(['nvidia-smi']+a,text=True).strip()
 return dict(at=now(),gpu=q(['-i',plan['gpu_uuid'],'--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits']),apps=q(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']))
assert old['status']=='failed_preserved' and old['stages'][0]['returncode']==0 and old['stages'][1]['returncode']==1
assert not (P/'held-native/unique-raw-predictions.jsonl').exists()
assert sha(P/'held-native.stderr')==plan['original_error_sha256']
for n,r in plan['files'].items():assert sha(D/n)==r['sha256'] and (D/n).stat().st_size==r['bytes']
save(D/'driver-started.json',dict(at=now(),pid=os.getpid(),original_deadline_epoch=deadline,no_training=True,no_prior_held_forward=True))
state=dict(status='running',started_at=now(),deadline_epoch=deadline,original_failed_state=old,stages=[],original_run_preserved=True);started=time.monotonic();child=None
try:
 fresh=gpu();f=[x.strip() for x in fresh['gpu'].split(',')];assert f[1]==plan['gpu_uuid'] and int(f[2])<=256 and int(f[3])>=35000 and plan['gpu_uuid'] not in fresh['apps'];state['fresh_gpu']=fresh
 native=Path(plan['native_root']);env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],PYTHONPATH=str(native/'tf-gpu-overlay'),LD_LIBRARY_PATH=str(native/'tf-gpu-overlay/nvidia/cudnn/lib')+':/usr/local/cuda-11.8/lib64',FORCE_TF_AVAILABLE='1',USE_TF='1',USE_TORCH='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',CAMPAIGN_DEADLINE_EPOCH=str(deadline))
 for stage,script,visible in [('held-native','infer_held.py',plan['gpu_uuid']),('held-provenance','provenance.py','')]:
  remaining=deadline-time.time()-15;assert remaining>0;env['CUDA_VISIBLE_DEVICES']=visible;t=time.monotonic()
  with (D/(stage+'.stdout')).open('x') as out,(D/(stage+'.stderr')).open('x') as err:
   child=subprocess.Popen([str(native/'.venv/bin/python'),script],cwd=D,env=env,stdout=out,stderr=err,start_new_session=True)
   try:rc=child.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM);rc=child.wait(timeout=30);state['timed_out']=True
  state['stages'].append(dict(stage=stage,returncode=rc,seconds=time.monotonic()-t));assert rc==0 and not state.get('timed_out')
 state.update(status='GPU_AND_NATIVE_COMPLETE_CPU_ENDPOINTS_PENDING',returncode=0)
except BaseException as e:state.update(status='failed_preserved',returncode=1,error=repr(e))
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 state.update(finished_at=now(),seconds=time.monotonic()-started,release=gpu(),owned_child_absent=child is None or child.poll() is not None);save(D/'state.json',state)
 paths=[p for p in D.iterdir() if p.is_file() and p.name!='held.input.jsonl']
 for name in ['actual','held-final']:paths.extend(p for p in (D/name).glob('*') if p.is_file())
 members={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
 with zipfile.ZipFile(D/'terminal.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,str(p.relative_to(D)))
  z.writestr('members.json',json.dumps(members,indent=2)+'\n')
 with (D/'terminal.zip').open('rb') as f:os.fsync(f.fileno())
 save(D/'terminal-manifest.json',dict(state=state,members=members,archive=dict(path=str(D/'terminal.zip'),bytes=(D/'terminal.zip').stat().st_size,sha256=sha(D/'terminal.zip'))));print(json.dumps(state),flush=True)
raise SystemExit(state['returncode'])
