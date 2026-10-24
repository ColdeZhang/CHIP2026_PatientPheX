"""Inspect an ambiguous original CPU start; only verified absence permits reentry.

The original driver creates driver-started.json exclusively before running the
CPU command. Reentry reuses its exact driver/command/env, so any delayed original
start loses that claim before loading a tokenizer. No GPU/model command exists.
"""
from pathlib import Path
import argparse,ast,json,os,shlex,sys,time,hashlib
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=argparse.ArgumentParser();P.add_argument('action',choices=['inspect','start']);P.add_argument('--directory',type=Path,required=True);a=P.parse_args();B=a.directory;p=json.loads((B/'plan.json').read_text());root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+p['remote'];stamp=str(time.time_ns())
def save(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
cl=Client();cl.initialize()
def call(name,params,label):
 r=cl.request('tools/call',dict(name=name,arguments=dict(mount_id=p['mount_id'],**params)));save(B/label,r)
 if r.get('isError'):raise RuntimeError('Ambiguous/offline preserved: inspect original state, no blind replay')
 return r.get('structuredContent') or json.loads(r['content'][0]['text'])
if a.action=='inspect':
 boundary=json.loads((B/'recovery-offline-boundary.json').read_text());assert time.time()>=boundary['next_inventory_not_before_epoch']
 code='''from pathlib import Path
import os,json,time
r=Path(ROOT);out={'at_unix':time.time(),'namespace':str(r),'processes':[],'files':{}}
for name in ['driver-started.json','terminal.json','native-intent.json','native-result.json']:
 p=r/name
 if p.is_file():out['files'][name]=json.loads(p.read_text())
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  if (q/'cwd').resolve()==r:out['processes'].append({'pid':int(q.name),'argv0':(q/'cmdline').read_bytes().split(bytes([0]))[0].decode(errors='replace')})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(out))
'''
 r=call('run_command',dict(command='python3 -c '+shlex.quote('ROOT='+repr(root)+'\n'+code),timeout_seconds=20),'recovery-inventory-'+stamp+'-tool.json');assert r['exit_code']==0;x=json.loads(r['stdout']);save(B/('recovery-inventory-'+stamp+'.json'),x);print(json.dumps(x))
else:
 assert not (B/'recovery-start-intent.json').exists();candidates=sorted(x for x in B.glob('recovery-inventory-*.json') if not x.name.endswith('-tool.json'));inv=candidates[-1];x=json.loads(inv.read_text());assert time.time()-x['at_unix']<90 and not x['files'] and not x['processes']
 original=json.loads((B/'start-intent.json').read_text());source=Path(__file__).with_name('start_collect.py');tree=ast.parse(source.read_text());driver=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and any(isinstance(t,ast.Name) and t.id=='driver' for t in n.targets));assert "with p.open('x')" in driver and "c=subprocess.Popen(COMMAND" in driver
 save(B/'recovery-start-intent.json',dict(at_unix=time.time(),observed_original_namespace_empty=inv.name,original_intent_sha256=sha(B/'start-intent.json'),source_sha256=sha(Path(__file__)),same_original_CPU_command_environment=True,budget_seconds=120,grace_seconds=5,exclusive_remote_driver_prevents_duplicate=True))
 cmd=shlex.join([p['python'],'-u','-c','COMMAND='+repr(original['command'])+'\n'+driver]);r=call('start_process',dict(command=cmd,cwd=p['remote'],env=original['environment']),'recovery-start-tool.json');save(B/'recovery-job.json',r);print(json.dumps(r))
