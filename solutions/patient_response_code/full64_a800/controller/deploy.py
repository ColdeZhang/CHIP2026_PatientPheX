"""Resume exact private deployment pieces after UNKNOWN; never starts model work."""
from pathlib import Path
import argparse,base64,fcntl,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();B=a.directory;plan=json.loads((B/'controller-plan.json').read_text());m=json.loads((B/'deployment-manifest.json').read_text());payload=Path(m['archive']['path']).read_bytes();assert sha(payload)==m['archive']['sha256']
 lock=(B/'deploy-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (B/'campaign.json').exists() and not (B/'epoch-001-start-intent.json').exists()
 T=B/'deployment-transport';T.mkdir(exist_ok=True)
 for p in T.glob('*.backoff.json'):assert time.time()>=json.loads(p.read_text())['not_before_unix']
 if not (B/'deployment-intent.json').exists():save(B/'deployment-intent.json',dict(manifest=m,at_epoch=time.time(),no_model=True))
 else:assert json.loads((B/'deployment-intent.json').read_text())['manifest']==m
 cl=Client();cl.initialize()
 def call(mount,n,args,label):
  p=T/(label+'-'+str(time.time_ns()));safe={k:v for k,v in args.items() if k!='content'};save(p.with_suffix('.intent.json'),dict(tool=n,arguments=safe))
  r=cl.request('tools/call',dict(name=n,arguments=dict(mount_id=mount,**args)));save(p.with_suffix('.response.json'),r)
  if r.get('isError'):
   if 'LUNA_OFFLINE' in json.dumps(r):save(T/(str(time.time_ns())+'.backoff.json'),dict(not_before_unix=time.time()+185))
   raise RuntimeError('Original deployment outcome preserved; inventory exact pieces before continuation')
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 def remote(mount,code,param,label):
  r=call(mount,'run_command',dict(command='python3 -c '+shlex.quote('P='+repr(param)+'\n'+code),timeout_seconds=60),label);assert r['exit_code']==0;return json.loads(r['stdout'])
 def put(mount,path,raw,label):
  r=call(mount,'write_file',dict(path=path,content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True),label);assert r['size']==len(raw) and r['sha256']==sha(raw)
 inventory='''from pathlib import Path
import os,pwd,json,hashlib
p=Path(P['root']);assert pwd.getpwuid(os.getuid()).pw_name=='zhangyuheng'
if p.exists():assert not p.is_symlink() and p.stat().st_uid==os.getuid()
assert not (p/'campaign.json').exists() and not (p/'control').exists()
parts={q.name:dict(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in p.glob('part-*') if q.is_file()}
print(json.dumps(dict(uid=os.getuid(),parts=parts)))
'''
  # Namespace identity checked before every continuation; existing different bytes fail.
 inv=remote(plan['mount_id'],inventory,dict(root=plan['remote_absolute']),'source-inventory');parts=[]
 for i,start in enumerate(range(0,len(payload),32768)):
  b=payload[start:start+32768];r=dict(name='part-%03d'%i,bytes=len(b),sha256=sha(b));parts.append(r)
  if r['name'] in inv['parts']:assert inv['parts'][r['name']]=={k:r[k] for k in ['bytes','sha256']}
  else:put(plan['mount_id'],plan['remote_relative']+'/'+r['name'],b,r['name'])
 assert set(inv['parts'])<={r['name'] for r in parts}
 verify='''from pathlib import Path
import os,pwd,json,hashlib,tarfile,gzip
p=Path(P['root']);assert pwd.getpwuid(os.getuid()).pw_name=='zhangyuheng' and p.stat().st_uid==os.getuid();os.chmod(p,0o700)
assert os.statvfs(p).f_bavail*os.statvfs(p).f_frsize>=512*1024**2
assert not (p/'campaign.json').exists() and not (p/'control').exists()
def put(path,b):
 if path.exists():assert not path.is_symlink() and path.read_bytes()==b
 else:
  with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 with path.open('rb') as f:os.fsync(f.fileno())
 for parent in [path.parent,p]:
  fd=os.open(str(parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for r in P['parts']:
 b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
b=b''.join((p/r['name']).read_bytes() for r in P['parts']);assert len(b)==P['manifest']['archive']['bytes'] and hashlib.sha256(b).hexdigest()==P['manifest']['archive']['sha256'];put(p/'deployment.tar.gz',b)
with tarfile.open(p/'deployment.tar.gz') as t:
 assert set(t.getnames())==set(P['manifest']['files']) and len(t.getnames())==len(P['manifest']['files'])
 for m in t.getmembers():
  assert m.isfile() and m.name==Path(m.name).name;b=t.extractfile(m).read();r=P['manifest']['files'][m.name];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];put(p/m.name,b)
  if m.name.endswith('.py'):compile(b,m.name,'exec')
for gz,r in P['decompress'].items():
 b=gzip.decompress((p/gz).read_bytes());assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];put(p/r['name'],b)
shm=Path(P['shm']);assert not Path(P['output']).exists()
if shm.exists():assert shm.is_dir() and not shm.is_symlink() and shm.stat().st_uid==os.getuid()
else:shm.mkdir(mode=0o700)
print(json.dumps(dict(status='ACTUAL_SOURCE_DATA_SHA_COMPILE_FSYNC_DEPLOYED',files=len(P['manifest']['files']),uid=os.getuid(),plan_sha256=hashlib.sha256((p/'controller-plan.json').read_bytes()).hexdigest(),private_shm=str(shm))))
'''
 actual=remote(plan['mount_id'],verify,dict(root=plan['remote_absolute'],manifest=m,parts=parts,decompress=plan['decompress'],shm=str(Path(plan['output_dir']).parent),output=plan['output_dir']),'source-verify')
 if not (B/'source-deployment.json').exists():save(B/'source-deployment.json',actual)
 target_files={n:(B/n).read_bytes() for n in ['fetch_checkpoint.py','io_driver.py','controller-plan.json']}
 invcode='''from pathlib import Path
import os,json,hashlib
p=Path(P['root']);assert os.getuid()==1006
if p.exists():assert p.stat().st_uid==1006 and not p.is_symlink()
assert not (p/'campaign.json').exists()
print(json.dumps({n:dict(bytes=(p/n).stat().st_size,sha256=hashlib.sha256((p/n).read_bytes()).hexdigest()) for n in P['names'] if (p/n).exists()}))
'''
 inv=remote(plan['persistent_mount_id'],invcode,dict(root=plan['persistent_remote_absolute'],names=list(target_files)),'target-inventory')
 for n,b in target_files.items():
  if n in inv:assert inv[n]==dict(bytes=len(b),sha256=sha(b))
  else:put(plan['persistent_mount_id'],plan['persistent_remote_relative']+'/'+n,b,'target-'+n)
 verifytarget='''from pathlib import Path
import os,hashlib,json,shutil
p=Path(P['root']);assert os.getuid()==1006 and p.stat().st_uid==1006;os.chmod(p,0o700);assert shutil.disk_usage(p).free>=2*1024**3
for n,r in P['files'].items():
 q=p/n;assert not q.is_symlink() and q.stat().st_size==r['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==r['sha256']
 with q.open('rb') as f:os.fsync(f.fileno())
 if n.endswith('.py'):compile(q.read_text(),n,'exec')
fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ACTUAL_PERSISTENT_FETCH_SOURCE_FSYNC',free=shutil.disk_usage(p).free)))
'''
 actual=remote(plan['persistent_mount_id'],verifytarget,dict(root=plan['persistent_remote_absolute'],files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in target_files.items()}),'target-verify')
 if not (B/'target-deployment.json').exists():save(B/'target-deployment.json',actual)
 print(json.dumps(dict(status='PRIVATE_SOURCE_AND_PERSISTENT_CPU_FETCH_DEPLOYED_NO_GPU')))
if __name__=='__main__':main()
