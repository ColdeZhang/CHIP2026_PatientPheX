"""Independent formal600+TERM30 training and fresh held inference campaign."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time,shutil,traceback,zipfile
D=Path.cwd();plan=json.loads((D/'plan.json').read_text())
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
 return dict(at=now(),gpu=q(['-i',plan['gpu_uuid'],'--query-gpu=index,uuid,name,memory.used,memory.free','--format=csv,noheader,nounits']),apps=q(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']))
activation=json.loads((D/'activation.json').read_text());assert activation['official_submission_accepted'] is True and activation['execute_this_probe'] is True and activation['plan_sha256']==sha(D/'plan.json')
assert plan['execution_authorized_at_preparation'] is True and plan['cap_seconds']==600 and plan['TERM_grace_seconds']==30
assert sha(__file__)==plan['sources']['bounded_driver.py']['sha256']
with (D/'driver-started.json').open('x') as f:json.dump(dict(at=now(),plan_sha256=sha(D/'plan.json'),activation=activation),f)
start=time.monotonic();deadline=start+600;deadline_epoch=time.time()+600;child=None;state=dict(status='running',started_at=now(),plan_sha256=sha(D/'plan.json'),CPU_preparation_is_not_GPU_evidence=True);write(D/'state.json',state)
try:
 assert shutil.disk_usage(D).free>=plan['minimum_output_free_bytes']
 for n,ref in plan['sources'].items():assert sha(D/n)==ref['sha256'] and (D/n).stat().st_size==ref['bytes']
 for name in ['training-queries.jsonl.gz','sampling.json','variable-coverage.json','held.input.jsonl']:
  item=plan[{'training-queries.jsonl.gz':'dataset','sampling.json':'sampling','variable-coverage.json':'prior_variable_coverage','held.input.jsonl':'held_input'}[name]];assert sha(D/name)==item['sha256'] and (D/name).stat().st_size==item['bytes']
 for item in plan['assets'].values():assert sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes']
 fresh=gpu();fields=[x.strip() for x in fresh['gpu'].split(',')];assert fields[0]=='2' and fields[1]==plan['gpu_uuid'] and 'A100' in fields[2] and int(fields[3])<=256 and int(fields[4])>=35000
 assert not any(x.split(',')[0].strip()==plan['gpu_uuid'] for x in fresh['apps'].splitlines());state['fresh_gpu']=fresh;state['fresh_disk_free_bytes']=shutil.disk_usage(D).free;write(D/'state.json',state)
 native=Path(plan['native_root']);env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],PYTHONPATH=str(native/'tf-gpu-overlay'),LD_LIBRARY_PATH=str(native/'tf-gpu-overlay/nvidia/cudnn/lib')+':/usr/local/cuda-11.8/lib64',FORCE_TF_AVAILABLE='1',USE_TF='1',USE_TORCH='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
 env['CAMPAIGN_DEADLINE_EPOCH']=str(deadline_epoch);state['deadline_epoch']=deadline_epoch;state['stages']=[]
 for stage,script,extra,visible in [('training','trainer.py',['--plan','plan.json'],plan['gpu_uuid']),('held-native','infer_held.py',[],plan['gpu_uuid']),('held-provenance','provenance.py',[],'')]:
  remaining=deadline-time.monotonic()-plan['control_reserve_seconds'];assert remaining>0;env['CUDA_VISIBLE_DEVICES']=visible;state['stage']=stage;state['remaining_compute_budget_seconds']=remaining;write(D/'state.json',state)
  stage_start=time.monotonic()
  with (D/(stage+'.stdout')).open('x') as out,(D/(stage+'.stderr')).open('x') as err:
   child=subprocess.Popen([str(native/'.venv/bin/python'),script]+extra,cwd=D,env=env,stdout=out,stderr=err,start_new_session=True);state['child_pid']=child.pid;write(D/'state.json',state)
   try:rc=child.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    state['timed_out']=True;os.killpg(child.pid,signal.SIGTERM)
    try:rc=child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
  state['stages'].append(dict(stage=stage,returncode=rc,seconds=time.monotonic()-stage_start));write(D/'state.json',state);assert rc==0 and not state.get('timed_out')
 assert (D/'actual/complete.json').exists() and (D/'held-final/manifest.json').exists();state.update(status='GPU_AND_NATIVE_COMPLETE_CPU_ENDPOINTS_PENDING',returncode=0)
except BaseException as e:
 state.update(status='failed_preserved',returncode=1,error=repr(e),traceback=traceback.format_exc())
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 try:
  state['release']=gpu();state['owned_child_absent']=child is None or not any(x.split(',')[-1].strip()==str(child.pid) for x in state['release']['apps'].splitlines())
 except BaseException as e:
  state['release']={'error':repr(e)};state['owned_child_absent']=False;state['returncode']=1;state['status']='release_unverified_preserved'
 state['finished_at']=now();state['seconds']=time.monotonic()-start;write(D/'state.json',state)
 files=[p for p in D.iterdir() if p.is_file() and p.name not in ['terminal.zip','terminal-manifest.json','training-queries.jsonl.gz','held.input.jsonl']]
 for directory in ['actual','held-final']:
  if (D/directory).exists():files += [p for p in (D/directory).glob('*') if p.is_file() and p.suffix!='.npz']
 members={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files if p.is_file()}
 # Large full model/optimizer and raw inference remain persistent; exact references recorded.
 artifacts={}
 for directory in ['trained-state','held-native']:
  if (D/directory).exists():
   for p in sorted((D/directory).iterdir()):
    if p.is_file():
     with p.open('rb') as f:os.fsync(f.fileno())
     artifacts[str(p.relative_to(D))]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 write(D/'persistent-artifacts.json',dict(files=artifacts,scope='Actual persistent A100 disk; complete state/raw retained, no large Git files'))
 members['persistent-artifacts.json']=dict(bytes=(D/'persistent-artifacts.json').stat().st_size,sha256=sha(D/'persistent-artifacts.json'))
 with zipfile.ZipFile(D/'terminal.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in members:z.write(D/n,n)
  z.writestr('members.json',json.dumps(members,indent=2)+'\n')
 with (D/'terminal.zip').open('rb') as f:os.fsync(f.fileno())
 write(D/'terminal-manifest.json',dict(state=state,members=members,archive=dict(path=str(D/'terminal.zip'),bytes=(D/'terminal.zip').stat().st_size,sha256=sha(D/'terminal.zip'))));print(json.dumps(state),flush=True)
raise SystemExit(state['returncode'])
