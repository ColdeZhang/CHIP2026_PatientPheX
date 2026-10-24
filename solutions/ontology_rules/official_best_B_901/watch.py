"""Read-only original batch monitor and fixed result recovery; no inference start."""
from pathlib import Path
import json,sys,time,subprocess,hashlib,datetime,shlex
D=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';deadline=time.monotonic()+2700;last=None;milestones=D/'milestones';milestones.mkdir(exist_ok=True)
with (D/'watch-started.json').open('x') as f:json.dump(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_job=json.loads((D/'job.json').read_text())['result']['id'],read_only_no_start=True),f)
def call(name,a):
 r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=M,**a)));assert not r.get('isError'),r;return r['structuredContent']
while time.monotonic()<deadline:
 code="from pathlib import Path;p=Path('official-B901-three-public/run-state.json');print(p.read_text())"
 r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=20));assert r['exit_code']==0,r;state=json.loads(r['stdout']);signature=[[(p['model'],p['mode'],p.get('returncode')) for p in state['phases']],state.get('status')]
 if signature!=last:
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S');(milestones/(stamp+'.json')).write_text(json.dumps(state,indent=2)+'\n');last=signature;print(json.dumps(dict(at=stamp,phases=signature,completed_models=state.get('completed_models',[]))),flush=True)
 for item in state.get('completed_models',[]):
  model=item['model']
  if not (D/'results'/model/'collection.json').exists():
   p=subprocess.run([sys.executable,str(D/'operations.py'),'collect',model],capture_output=True,text=True);(D/('collect-'+model+'.stdout')).write_text(p.stdout);(D/('collect-'+model+'.stderr')).write_text(p.stderr);assert p.returncode==0,dict(model=model,returncode=p.returncode,stderr_path=str(D/('collect-'+model+'.stderr')));print(p.stdout.strip(),flush=True)
 if state.get('status'):
  j=json.loads((D/'job.json').read_text())['result'];terminal={s:call('read_process_output',dict(process_id=j['id'],stream=s,cursor=0,limit=32768)) for s in ['stdout','stderr']}
  if all(not x['running'] for x in terminal.values()):
   (D/'managed-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');(D/'terminal-state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(status=state['status'],returncode=state['returncode'],elapsed_seconds=state['elapsed_seconds'],release=state['release'])),flush=True);sys.exit(state['returncode'])
 time.sleep(60)
raise TimeoutError('read-only watcher budget; original job must be inspected, never restarted')
