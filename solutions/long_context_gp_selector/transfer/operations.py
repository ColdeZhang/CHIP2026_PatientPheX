"""Recorded single-intent transfers, no implicit replay after UNKNOWN."""
import argparse,hashlib,json,shlex,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S.parent));from rpc import call
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/asset-copy-v1');D.mkdir(exist_ok=True)
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy-source','prepare','serve','observe-source','deploy-target','fetch','observe-target']);a=ap.parse_args();p=json.load(open(S/'copy-plan.json'));rel=p['relative'];remote=P+'/'+rel;source=a.action in ('deploy-source','prepare','serve','observe-source');mount=p['source_mount' if source else 'target_mount']
 if a.action.startswith('deploy'):
  code="import hashlib,json;from pathlib import Path;r=Path("+repr(remote)+");print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['copy_assets.py','copy-plan.json'] if (r/n).exists()}))"
  inventory=call(D,'run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=30),'deploy-recovery-inventory');assert inventory['exit_code']==0;existing=json.loads(inventory['stdout'])
  for n in ('copy_assets.py','copy-plan.json'):
   if existing.get(n)==hashlib.sha256((S/n).read_bytes()).hexdigest():continue
   v=call(D,'write_file',dict(mount_id=mount,path=rel+'/'+n,content=(S/n).read_text(),encoding='utf-8',create_parents=True),'deploy-'+a.action+'-'+n)
  save(D/(a.action+'-'+str(time.time_ns())+'.json'),dict(completed=time.time(),source_sha=hashlib.sha256((S/'copy_assets.py').read_bytes()).hexdigest()))
 elif a.action=='prepare':
  admit=time.time()+180;save(D/('prepare-v2-'+str(time.time_ns())+'.intent.json'),dict(at=time.time(),admit_before=admit,no_retry=True));v=call(D,'run_command',dict(mount_id=mount,cwd=rel,command=shlex.join([p['source_python'],'copy_assets.py','prepare','--plan','copy-plan.json','--admit-before',str(admit)]),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=120),'prepare-assets');save(D/'prepared.json',v);assert v['exit_code']==0;print(v['stdout'])
 elif a.action in ('serve','fetch'):
  admit=time.time()+180;argv=['timeout','--signal=TERM','--kill-after=10',str(450 if source else 300),'python3','-u','copy_assets.py',a.action,'--plan','copy-plan.json','--admit-before',str(admit)]
  if not source:
   r=json.load(open(D/'source-ready.json'))['serve-ready.json'];argv+=['--port',str(r['port']),'--manifest-sha',r['manifest_sha256']]
  cmd=shlex.join(argv)+' > '+a.action+'.stdout 2> '+a.action+'.stderr';save(D/(a.action+'.intent.json'),dict(command=cmd,admit_before=admit,no_retry=True));v=call(D,'start_process',dict(mount_id=mount,cwd=rel,command=cmd,env={'CUDA_VISIBLE_DEVICES':''}),a.action);save(D/(a.action+'-started.json'),v);print(json.dumps(v))
 else:
  names=['serve-ready.json','serve-closed.json','source-manifest.json'] if source else ['copy-complete.json']
  code="import json;from pathlib import Path;r=Path("+repr(remote)+");o={n:json.loads((r/n).read_text()) for n in "+repr(names)+" if (r/n).exists()};o['stderr']=(r/"+repr('serve.stderr' if source else 'fetch.stderr')+").read_text() if (r/"+repr('serve.stderr' if source else 'fetch.stderr')+").exists() else '';\nif 'source-manifest.json' in o:o['source-manifest.json'].pop('runtime_members',None)\nprint(json.dumps(o))"
  v=call(D,'run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=30),a.action);assert v['exit_code']==0;o=json.loads(v['stdout']);save(D/(a.action+'-'+str(time.time_ns())+'.json'),o)
  if source and 'serve-ready.json' in o and not (D/'source-ready.json').exists():save(D/'source-ready.json',o)
  if not source and 'copy-complete.json' in o:save(D/'copy-complete.json',o)
  print(json.dumps(o))
if __name__=='__main__':main()
