"""Explicit post-four-ACK inference stages; no implicit training or quality."""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
p=argparse.ArgumentParser();p.add_argument('action',choices=['start','inspect','collect']);p.add_argument('--controller',type=Path,required=True);a=p.parse_args();B=a.controller;config=json.loads((B/'controller-plan.json').read_text());root=config['remote_absolute'];client=Client();client.initialize()
def sha(b):return hashlib.sha256(b).hexdigest()
def save(n,x):
 with (B/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def call(n,p,label):
 r=client.request('tools/call',dict(name=n,arguments=dict(mount_id=config['mount_id'],**p)));save(label,r)
 if r.get('isError'):raise RuntimeError('Unknown/error; preserve intent and inspect, never repeat model start')
 return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def remote(code,args,label):
 r=call('run_command',dict(command='python3 -c '+shlex.quote('PARAM='+repr(args)+'\n'+code),timeout_seconds=30),label);assert r['exit_code']==0;return json.loads(r['stdout'])
if a.action=='start':
 assert (B/'campaign-complete.json').exists() and not (B/'inference-start-intent.json').exists()
 auth=json.loads((B/'GPU-AUTHORIZATION.json').read_text());assert auth.get('F1_inference_and_quality_authorized') is True
 plan=json.loads((B/'inference-plan.json').read_text());assert plan['controller_plan']['sha256']==sha((B/'controller-plan.json').read_bytes())
 save('inference-start-intent.json',dict(at_unix=time.time(),plan_sha256=sha((B/'inference-plan.json').read_bytes()),cap=1200,grace=30,no_retry=True))
 code='''from pathlib import Path
import os,json,hashlib
root=Path(PARAM['root']);assert not (root/'inference-control').exists()
for n,content in PARAM['files'].items():
 p=root/n;b=content.encode()
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(str(root),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(published=True)))
'''
 remote(code,dict(root=root,files={n:(B/n).read_text() for n in ['inference-plan.json','inference-proof.json']}),'inference-publish-tool.json')
 r=call('start_process',dict(command=shlex.join([config['python'],'-u',root+'/inference_driver.py','--plan',root+'/inference-plan.json']),cwd=config['remote_relative'],env=config['env']),'inference-start-tool.json');save('inference-job.json',r);print(json.dumps(r))
elif a.action=='inspect':
 code='''from pathlib import Path
import json
p=Path(PARAM['root']);d={}
for n in ['driver-started.json','child-started.json','wrapper-complete.json','gpu-release.json']:
 q=p/'inference-control'/n
 if q.exists():d[n]=json.loads(q.read_text())
f=p/'inference-output/responses.jsonl';d['responses']=sum(1 for _ in f.open()) if f.exists() else 0
print(json.dumps(d))
'''
 r=remote(code,dict(root=root),'inference-inspect-%d-tool.json'%time.time_ns());print(json.dumps(r))
else:
 assert not (B/'inference-collect-intent.json').exists();save('inference-collect-intent.json',dict(at_unix=time.time(),no_model=True))
 code='''from pathlib import Path
import json,zipfile,hashlib,os
p=Path(PARAM['root']);assert (p/'inference-control/wrapper-complete.json').exists() and (p/'inference-control/gpu-release.json').exists()
paths=sorted(q for d in ['inference-control','inference-output'] for q in (p/d).rglob('*') if q.is_file());refs=[]
for q in paths:
 assert not q.is_symlink();b=q.read_bytes();assert len(b)<5000000;refs.append(dict(name=q.relative_to(p).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
zp=p/'inference-terminal.zip'
if not zp.exists():
 with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
  for q,r in zip(paths,refs):z.writestr(r['name'],q.read_bytes())
 with zp.open('rb') as f:os.fsync(f.fileno())
with zipfile.ZipFile(zp) as z:
 assert set(z.namelist())=={r['name'] for r in refs}
 for r in refs:assert len(z.read(r['name']))==r['bytes'] and hashlib.sha256(z.read(r['name'])).hexdigest()==r['sha256']
b=zp.read_bytes();print(json.dumps(dict(remote=str(zp),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),files=refs)))
'''
 m=remote(code,dict(root=root),'inference-collect-manifest-tool.json');save('inference-terminal-manifest.json',m);pieces=[]
 for offset in range(0,m['bytes'],32768):
  code='''import json,base64,hashlib
with open(PARAM['path'],'rb') as f:f.seek(PARAM['offset']);b=f.read(PARAM['size'])
print(json.dumps(dict(b64=base64.b64encode(b).decode(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))
'''
  n=min(32768,m['bytes']-offset);r=remote(code,dict(path=m['remote'],offset=offset,size=n),'inference-collect-%09d-tool.json'%offset);b=base64.b64decode(r['b64']);assert len(b)==n==r['bytes'] and sha(b)==r['sha256'];(B/('inference-part-%09d.bin'%offset)).write_bytes(b);pieces.append(b)
 payload=b''.join(pieces);assert len(payload)==m['bytes'] and sha(payload)==m['sha256'];(B/'inference-terminal.zip').write_bytes(payload)
 dest=B/'inference-terminal';dest.mkdir(exist_ok=False)
 with zipfile.ZipFile(io.BytesIO(payload)) as z:
  for r in m['files']:
   n=r['name'];assert not Path(n).is_absolute() and '..' not in Path(n).parts;b=z.read(n);assert len(b)==r['bytes'] and sha(b)==r['sha256'];q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 save('inference-collected.json',dict(manifest_sha256=sha((B/'inference-terminal-manifest.json').read_bytes()),zip_sha256=m['sha256'],members=len(m['files']),all_local_SHA=True));print('collected',len(m['files']))
