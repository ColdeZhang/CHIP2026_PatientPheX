"""Bounded hidden-CPU extraction and small immutable result transport."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import zipfile

P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE=P+'/research/canonical-prefix-agreement-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','collect']);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s=Path(__file__).resolve().parent
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 client=Client();client.initialize()
 def call(name,args,label):
  tag=out/(str(time.time_ns())+'-'+label);safe={k:v for k,v in args.items() if k!='content'};save(str(tag)+'.intent.json',dict(tool=name,args=safe))
  try:r=client.request('tools/call',dict(name=name,arguments=args))
  except Exception as exc:save(str(tag)+'.UNKNOWN.json',dict(error=type(exc).__name__));raise
  if r.get('isError'):save(str(tag)+'.ERROR.json',r);raise RuntimeError('Original request failure preserved; no automatic replay')
  value=r.get('structuredContent') or json.loads(next(c['text'] for c in r['content'] if c['type']=='text'))
  save(str(tag)+'.response.json',{k:v for k,v in value.items() if name!='read_file' or k!='content'});return value
 def command(code,label,env=None,seconds=30):
  r=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),env=env or {},timeout_seconds=seconds),label)
  assert r['exit_code']==0,r.get('stderr');return json.loads(r['stdout'])
 if a.stage=='prepare':
  assert not (out/'prepared.json').exists();archive=out/'deployment.zip'
  files={f'source/{n}':s/n for n in ['prefix.py','contracts.py','native_extract.py']};files['native-plan.json']=s/'evidence/native-plan.json'
  with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
   for n,p in files.items():z.write(p,n)
  identity=ref(archive);relative='research/canonical-prefix-agreement-v1/deployment.zip'
  call('write_file',dict(mount_id=MOUNT,path=relative,content=base64.b64encode(archive.read_bytes()).decode(),encoding='base64',create_parents=True),'upload')
  code='ROOT='+repr(REMOTE)+'\nEXPECTED='+repr(identity['sha256'])+'\n'+PREPARE
  result=command(code,'prepare');save(out/'prepared.json',dict(local=identity,remote=result));print(json.dumps(ref(out/'prepared.json')))
 elif a.stage=='run':
  assert (out/'prepared.json').exists() and not (out/'start-intent.json').exists();save(out/'start-intent.json',dict(no_NN=True,budget=60,TERM_grace=5,source=ref(__file__)))
  code='ROOT='+repr(REMOTE)+'\n'+RUN;result=command(code,'run-once',seconds=75);save(out/'run-result.json',result);print(json.dumps(ref(out/'run-result.json')))
 else:
  assert not (out/'collection.json').exists();ready=command('ROOT='+repr(REMOTE)+'\n'+COLLECT,'freeze-results');save(out/'ready.json',ready);b=bytearray()
  for i in range(0,ready['bytes'],131072):
   r=call('read_file',dict(mount_id=MOUNT,path=ready['path'][len(P)+1:],offset=i,limit=min(131072,ready['bytes']-i)),f'read-{i}')
   b.extend(base64.b64decode(r['content']) if r.get('encoding')=='base64' else r['content'].encode())
  assert len(b)==ready['bytes'] and hashlib.sha256(b).hexdigest()==ready['sha256'];archive=out/'result.zip'
  with archive.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  actual=out/'actual';actual.mkdir()
  with zipfile.ZipFile(archive) as z:
   idx=json.loads(z.read('index.json'));assert set(z.namelist())==set(idx)|{'index.json'}
   for name,item in idx.items():
    assert '/' not in name and name not in ['.','..'];data=z.read(name);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    with (actual/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    assert ref(actual/name)['sha256']==item['sha256']
  save(out/'collection.json',dict(status='PREFIX_CPU_ALL_MEMBERS_SHA_FSYNC_REOPEN',archive=ref(archive),remote=ready,members=len(idx),no_NN=True));print(json.dumps(ref(out/'collection.json')))

PREPARE=r'''
import json,hashlib,zipfile
from pathlib import Path
root=Path(ROOT);archive=root/'deployment.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED
assert not (root/'source').exists()
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():
  assert '..' not in Path(n).parts and not n.startswith('/');p=root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(z.read(n))
plan=json.loads((root/'native-plan.json').read_text())
for r in plan['sources'].values():
 b=Path(r['path']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
print(json.dumps(dict(status='PREPARED_PREFIX_CPU',sources=plan['sources'])))
'''
RUN=r'''
from pathlib import Path
import os,json,subprocess,datetime,time
root=Path(ROOT);plan=json.loads((root/'native-plan.json').read_text());env=dict(os.environ);env.update(plan['environment']);env['CUDA_VISIBLE_DEVICES']=''
assert not (root/'actual').exists();started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
with (root/'stdout.log').open('xb') as out,(root/'stderr.log').open('xb') as err:
 r=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','60s',plan['python'],str(root/'source/native_extract.py'),'--plan',str(root/'native-plan.json')],stdout=out,stderr=err,env=env)
result=dict(return_code=r.returncode,seconds=time.monotonic()-tick,started_at_utc=started,no_NN=True)
with (root/'run-result.json').open('x') as f:json.dump(result,f);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''
COLLECT=r'''
from pathlib import Path
import json,hashlib,zipfile,os
root=Path(ROOT);run=json.loads((root/'run-result.json').read_text());assert run['return_code']==0
actual=root/'actual';assert (actual/'complete.json').exists();archive=root/'result.zip'
def identity(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
if not archive.exists():
 files={p.name:p for p in actual.iterdir() if p.is_file()};files['native-plan.json']=root/'native-plan.json';files['run-result.json']=root/'run-result.json'
 idx={n:identity(p) for n,p in files.items()}
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.write(p,n)
  z.writestr('index.json',json.dumps(idx)+'\n')
 with archive.open('rb') as f:os.fsync(f.fileno())
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(path=str(archive),**identity(archive))))
'''

if __name__=='__main__':main()
