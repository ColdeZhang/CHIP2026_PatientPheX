"""Bounded execution of the already reviewed two Sap argv; no gold access."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
root=Path.cwd()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(): (root/'postlink-state.json').write_text(json.dumps(state,indent=2)+'\n')
assert not (root/'postlink-state.json').exists(), 'No re-entry/overwrite'
assert sha(root/'postlink-commands.json')=='6f1c9044f4383810353b23752be8171416b9f7d450a4c3433aaf98d5ea5dcf71'
commands=json.loads((root/'postlink-commands.json').read_text())
assert [c['stage'] for c in commands]==['baseline-runtime','augmented-runtime']
uuid='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
assert os.environ['CUDA_VISIBLE_DEVICES']==uuid
formal=json.loads((root/'formal-run/state.json').read_text())
assert formal['complete'] and len(formal['stages'])==3
assert all(s['return_code']==0 for s in formal['stages'])
state={'started_at_utc':now(),'source_sha256':sha(__file__),'commands_sha256':sha(root/'postlink-commands.json'),'formal_state_sha256':sha(root/'formal-run/state.json'),'cuda_visible_devices':uuid,'gold_read':False,'stages':[],'complete':False}
write()
for item in commands:
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True)
 assert uuid not in apps, ('Assigned GPU is occupied',apps)
 name=item['stage']; command=item['command']
 entry={'name':name,'command':command,'started_at_utc':now(),'timeout_seconds':300,'gpu_preflight':gpu,'compute_apps_preflight':apps}
 state['stages'].append(entry);write();started=time.monotonic()
 with open(root/(name+'-sap.stdout.log'),'w') as out,open(root/(name+'-sap.stderr.log'),'w') as err:
  process=subprocess.Popen(command,cwd=item['cwd'],stdout=out,stderr=err,start_new_session=True)
  entry['pid']=process.pid;write()
  try: rc=process.wait(timeout=300)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL);process.wait();rc=124
 entry.update(return_code=rc,finished_at_utc=now(),seconds=time.monotonic()-started)
 if rc==0:
  entry['outputs']={f:sha(root/(name+'-sap')/f) for f in ['manifest.json','scored-candidates.jsonl']}
 write()
 if rc: raise SystemExit(rc)
state.update(complete=True,finished_at_utc=now());write();print(json.dumps(state),flush=True)
