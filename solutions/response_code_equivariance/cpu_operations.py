"""Single .100 CPU120+5 deploy/start/collect, no GPU or model operations."""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
B=Path('/workspace/patientphex-runs/round2-20261023/research/response-code-equivariance-v1/native-preparation-v1')
D=B.parent/'native100-v1'
M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REL='patient-response-code-equivariance/native100-v1'
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REL
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,x):
 with (D/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','start','collect']);a=ap.parse_args();D.mkdir(exist_ok=True)
 manifest=json.loads((B/'deployment-manifest.json').read_text());plan=json.loads((B/'native-plan.json').read_text());c=Client();c.initialize()
 def call(name,args,label):
  r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=M,**args)));save(label,r);assert not r.get('isError'),'Unknown/offline operation preserved; inspect original state, do not blind-repeat mutation'
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 def remote(code,param,label):
  x=call('run_command',dict(command='python3 -c '+shlex.quote('P='+repr(param)+'\n'+code),timeout_seconds=30),label);assert x['exit_code']==0,x.get('stderr');return json.loads(x['stdout'])
 if a.action=='deploy':
  assert not (D/'start-intent.json').exists();raw=Path(manifest['archive']['path']).read_bytes();assert sha(raw)==manifest['archive']['sha256']
  inv=remote("from pathlib import Path\nimport json,hashlib,os\nr=Path(P['root']);assert os.getuid()==1006;r.mkdir(parents=True,exist_ok=True,mode=0o700);assert r.stat().st_uid==1006 and not r.is_symlink();assert not (r/'actual').exists();print(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in r.glob('part-*')}))",dict(root=ROOT),'inventory-'+str(time.time_ns())+'-tool.json')
  parts=[]
  for offset in range(0,len(raw),32768):
   b=raw[offset:offset+32768];name='part-%09d'%offset;item=dict(bytes=len(b),sha256=sha(b));parts.append(dict(name=name,**item))
   if name in inv:assert inv[name]==item;continue
   value=call('write_file',dict(path=REL+'/'+name,content=base64.b64encode(b).decode(),encoding='base64',create_parents=False),name+'-'+str(time.time_ns())+'-tool.json');assert value['size']==len(b) and value['sha256']==sha(b)
  code='''from pathlib import Path
import json,hashlib,os,tarfile,io
r=Path(P['root']);m=P['manifest'];assert not (r/'actual').exists()
chunks=[]
for p in P['parts']:
 b=(r/p['name']).read_bytes();assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256'];chunks.append(b)
raw=b''.join(chunks);assert len(raw)==m['archive']['bytes'] and hashlib.sha256(raw).hexdigest()==m['archive']['sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
 assert set(t.getnames())==set(m['files'])
 for name,ref in m['files'].items():
  assert name==Path(name).name;b=t.extractfile(name).read();assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256'];p=r/name
  if p.exists():assert not p.is_symlink() and p.read_bytes()==b
  else:
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  if name.endswith('.py'):compile(b,name,'exec')
fd=os.open(str(r),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ALL9_SOURCE_INPUT_PLAN_SHA_FSYNC_REOPEN',files=len(m['files']),plan_sha256=hashlib.sha256((r/'native-plan.json').read_bytes()).hexdigest(),no_model=True)))
'''
  value=remote(code,dict(root=ROOT,manifest=manifest,parts=parts),'deployment-tool.json');save('deployment.json',value);print(json.dumps(value))
 elif a.action=='start':
  assert json.loads((D/'deployment.json').read_text())['plan_sha256']==sha((B/'native-plan.json').read_bytes());assert not (D/'start-intent.json').exists()
  command=shlex.join([plan['python'],'-u','native_driver.py','--plan',ROOT+'/native-plan.json','--output',ROOT+'/actual'])
  save('start-intent.json',dict(at_unix=time.time(),command=command,environment=plan['environment'],plan_sha256=sha((B/'native-plan.json').read_bytes()),source_sha256=sha(Path(__file__).read_bytes()),CPU_seconds=120,grace_seconds=5,no_repeat_unknown=True))
  value=call('start_process',dict(command=command,cwd=REL,env=plan['environment']),'start-tool.json');save('job.json',value);print(json.dumps({k:value[k] for k in ['id','pid','started_at','running']}))
 else:
  code='''from pathlib import Path
import json,io,zipfile,base64,hashlib
r=Path(P['root'])/'actual'
if not (r/'wrapper-complete.json').is_file():print(json.dumps(dict(complete=False)));raise SystemExit(0)
files={p.name:p.read_bytes() for p in r.iterdir() if p.is_file()};buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,b in files.items():z.writestr(name,b)
b=buf.getvalue();assert len(b)<100000
print(json.dumps(dict(complete=True,files={n:dict(bytes=len(x),sha256=hashlib.sha256(x).hexdigest()) for n,x in files.items()},bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),b64=base64.b64encode(b).decode(),wrapper=json.loads(files['wrapper-complete.json']))))
'''
  value=remote(code,dict(root=ROOT),'collect-'+str(time.time_ns())+'-tool.json')
  if not value['complete']:print(json.dumps(value));return
  b=base64.b64decode(value.pop('b64'));assert len(b)==value['bytes'] and sha(b)==value['sha256'];(D/'terminal.zip').write_bytes(b);out=D/'terminal';out.mkdir(exist_ok=False)
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   assert set(z.namelist())==set(value['files'])
   for n,ref in value['files'].items():assert n==Path(n).name;raw=z.read(n);assert len(raw)==ref['bytes'] and sha(raw)==ref['sha256'];(out/n).write_bytes(raw)
  save('terminal-manifest.json',value);print(json.dumps(dict(complete=True,sha256=value['sha256'],files=len(value['files']),wrapper=value['wrapper'])))
if __name__=='__main__':main()
