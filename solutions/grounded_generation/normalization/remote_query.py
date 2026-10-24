"""Chunked transport for the frozen one-shot Sap query; recover original intent."""
import argparse,base64,fcntl,hashlib,json,os,shlex,sys,time
from pathlib import Path

D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1/normalization-v1/deployment')
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 parent=p.resolve().parent
 while True:
  fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if parent.parent==parent:break
  parent=parent.parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['deploy','start','observe']);a=ap.parse_args()
 os.umask(0o077);cfg=json.loads((D/'package.json').read_text());root=cfg['remote_root'];rel=str(Path(root).relative_to(PROJECT));mount=cfg['mount_id'];plan=json.loads(Path(cfg['plan']['path']).read_text())
 assert sha(Path(cfg['archive']['path']).read_bytes())==cfg['archive']['sha256'];assert sha(Path(cfg['plan']['path']).read_bytes())==cfg['plan']['sha256']
 t=D/'transport';t.mkdir(exist_ok=True);lock=(t/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p in t.glob('backoff-*.json'):assert time.time()>=json.loads(p.read_text())['not_before']
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize()
 def call(tool,args,tag):
  f=t/(tag+'-'+str(time.time_ns()));safe=dict(args)
  if 'content' in safe:safe['content']='[frozen bytes omitted]'
  save(f.with_suffix('.intent.json'),dict(tool=tool,arguments=safe))
  try:r=c.request('tools/call',dict(name=tool,arguments=args))
  except Exception as e:
   save(f.with_suffix('.error.json'),dict(UNKNOWN=True,type=type(e).__name__));raise RuntimeError('Unknown outcome; recover original markers before any action') from None
  text=json.dumps(r)
  for k in ['MCP_TOKEN','MCP_URL']:
   if os.environ.get(k):text=text.replace(os.environ[k],'[REDACTED]')
  r=json.loads(text)
  if r.get('isError'):
   save(f.with_suffix('.error.json'),r)
   if 'LUNA_OFFLINE' in text:save(t/('backoff-'+str(time.time_ns())+'.json'),dict(not_before=time.time()+185))
   raise RuntimeError('Tool error preserved, no automatic replay')
  v=r.get('structuredContent')
  if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
  save(f.with_suffix('.response.json'),v);return v
 common='''from pathlib import Path
import json,hashlib,os,zipfile
root=Path(CFG['remote_root'])
def sha(b):return hashlib.sha256(b).hexdigest()
def sync(p):
 while True:
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if p.parent==p:break
  p=p.parent
'''
 def command(body,tag):
  v=call('run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote('CFG='+repr(cfg)+'\n'+common+body),timeout_seconds=40),tag)
  assert v['exit_code']==0,v.get('stderr','')[:300];return json.loads(v['stdout'])
 inventory="""parts={}
if root.exists():
 for p in root.glob('chunk-*.bin'):
  b=p.read_bytes();parts[p.name]=dict(bytes=len(b),sha256=sha(b))
markers=[n for n in ['start-intent.json','driver-started.json','child-started.json','wrapper-complete.json'] if (root/n).exists()]
print(json.dumps(dict(parts=parts,markers=markers,deployed=(root/'deployment-complete.json').exists(),actual_exists=(root/'actual').exists())))
"""
 if a.stage=='deploy':
  v=command(inventory,'inventory');assert not v['markers'] and not v['actual_exists']
  data=Path(cfg['archive']['path']).read_bytes();chunks=[]
  for offset in range(0,len(data),32768):
   b=data[offset:offset+32768];name='chunk-%06d.bin'%offset;want=dict(bytes=len(b),sha256=sha(b));chunks.append(name)
   if name in v['parts']:assert v['parts'][name]==want;continue
   got=call('write_file',dict(mount_id=mount,path=rel+'/'+name,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True),'chunk-'+str(offset));assert got['size']==len(b) and got['sha256']==sha(b)
  body="CHUNKS="+repr(chunks)+'''\nassert not any((root/n).exists() for n in ['driver-started.json','start-intent.json','actual'])
b=b''.join((root/n).read_bytes() for n in CHUNKS);assert len(b)==CFG['archive']['bytes'] and sha(b)==CFG['archive']['sha256']
p=root/'deployment.zip'
if p.exists():assert p.read_bytes()==b
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with zipfile.ZipFile(p) as z:
 assert set(z.namelist())==set(CFG['members']) and len(z.namelist())==len(CFG['members'])
 for name,item in CFG['members'].items():
  assert '/' not in name and name not in ['.','..'];value=z.read(name);assert len(value)==item['bytes'] and sha(value)==item['sha256']
  q=root/name;assert not q.is_symlink()
  if q.exists():assert q.read_bytes()==value
  else:
   with q.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
  with q.open('rb') as f:os.fsync(f.fileno())
  assert sha(q.read_bytes())==item['sha256']
  if name.endswith('.py'):compile(value,str(q),'exec')
receipt=dict(archive_sha256=CFG['archive']['sha256'],members=len(CFG['members']),all_actual_SHA_fsync_reopen=True)
p=root/'deployment-complete.json';encoded=(json.dumps(receipt,indent=2)+'\\n').encode()
if p.exists():assert p.read_bytes()==encoded
else:
 with p.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
sync(root);print(json.dumps(receipt))
'''
  got=command(body,'verify-deploy');save(D/'deployment-actual.json',got)
 elif a.stage=='start':
  assert not (D/'start-intent.json').exists();v=command(inventory,'inventory-before-start');assert v['deployed'] and not v['markers'] and not v['actual_exists']
  cmd=(D/'run-command.sh').read_text().strip();intent=dict(command=cmd,source_sha256=sha(Path(__file__).read_bytes()),plan=cfg['plan'],no_retry=True,prior_DocRE_release='2026-10-24T09:54:09.379711Z')
  save(D/'start-intent.json',intent)
  command("value="+repr(intent)+"\np=root/'start-intent.json'\nwith p.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())\nsync(root)\nprint(json.dumps(dict(claimed=True)))",'claim')
  got=call('start_process',dict(mount_id=mount,cwd=rel,command=cmd),'start');save(D/'started.json',got);print(json.dumps(got))
 else:
  body="""result={}
for name in ['driver-started.json','resource-check.json','child-started.json','wrapper-complete.json','actual/complete.json','actual/failure.json']:
 p=root/name
 if p.exists():result[name]=dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()),value=json.loads(p.read_text()))
for name in ['stdout.log','stderr.log']:
 p=root/name
 if p.exists():result[name]=p.read_text()[-3000:]
print(json.dumps(result))
"""
  got=command(body,'observe');save(D/('observation-'+str(time.time_ns())+'.json'),got);print(json.dumps(got))

if __name__=='__main__':main()
