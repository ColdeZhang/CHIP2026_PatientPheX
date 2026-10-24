"""Collect only original completed-training metadata; never load or mutate models."""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,shlex,sys,zipfile
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);a=p.parse_args();C=a.controller;D=C/'training-terminal';D.mkdir(exist_ok=False)
c=json.loads((C/'controller-plan.json').read_text());assert (C/'campaign-complete.json').exists()
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def js(p,j):save(p,(json.dumps(j,indent=2)+'\n').encode())
cl=Client();cl.initialize()
def remote(code,params,label):
 result=cl.request('tools/call',dict(name='run_command',arguments=dict(mount_id=c['mount_id'],command='python3 -c '+shlex.quote('PARAM='+repr(params)+'\n'+code),timeout_seconds=30)))
 js(D/(label+'-tool.json'),result)
 if result.get('isError'):raise RuntimeError('Read-only collection failed; preserve receipt, never retrain')
 r=result.get('structuredContent') or json.loads(result['content'][0]['text']);assert r['exit_code']==0
 return json.loads(r['stdout'])
code='''from pathlib import Path
import json,hashlib,base64,io,zipfile
p=Path(PARAM['root']);names=['run_manifest.json','training_complete.json','progress.json','base-loading-info.json','model-coverage.json','token-mask-audit.json']
for ep in [1,2]:names.extend([f'epoch-{ep:03d}/checkpoint.json',f'local-checkpoint-epoch-{ep:03d}.json'])
names.extend(['first-nonzero-update.json','resume-state-verified.json','resume-export-ack-verified.json'])
names+=sorted(q.relative_to(p).as_posix() for q in (p/'run_controls').glob('*.json'))
buf=io.BytesIO();refs={}
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names:
  q=p/n;assert q.is_file() and not q.is_symlink();b=q.read_bytes();assert len(b)<1000000
  refs[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());z.writestr(n,b)
b=buf.getvalue();assert len(b)<65536
print(json.dumps(dict(root=str(p),files=refs,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),b64=base64.b64encode(b).decode())))
'''
r=remote(code,dict(root=c['output_dir']),'collection');payload=base64.b64decode(r.pop('b64'));assert len(payload)==r['bytes'] and sha(payload)==r['sha256'];save(D/'terminal.zip',payload)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 assert set(z.namelist())==set(r['files'])
 for n,f in r['files'].items():
  assert not Path(n).is_absolute() and '..' not in Path(n).parts;b=z.read(n);assert len(b)==f['bytes'] and sha(b)==f['sha256'];q=D/'collected'/n;q.parent.mkdir(parents=True,exist_ok=True);save(q,b)
js(D/'manifest.json',r)
restores={}
for ep in [1]:
 q=D/'collected'/'resume-state-verified.json';j=json.loads(q.read_text());assert j['status']=='PASS_EXACT_LOGICAL_FULL_STATE_RESTORE' and j['completed_epochs']==ep and j['global_step']==16*ep and j['restored_state_digest']==j['serialized_state_digest'] and j['all_RNG_restored']
 restores[str(ep)]=j
js(D/'checks.json',dict(all_original_members_SHA=True,restores=restores,model_calls=0,source_sha256=sha(Path(__file__).read_bytes()),manifest_sha256=sha((D/'manifest.json').read_bytes())))
print(json.dumps(dict(collected=len(r['files']),zip_bytes=len(payload),zip_sha256=sha(payload),one_actual_complete_state_restore=True)))
