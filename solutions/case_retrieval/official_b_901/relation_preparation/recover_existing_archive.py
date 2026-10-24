"""Resume only an already frozen terminal archive, with durable small chunks.

No server mutation, new archive, model, generation, fit, score or automatic retry.
The existing archive manifest and independent server inventory must agree first.
"""
import argparse,base64,hashlib,json,os,sys,tarfile,time
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def save(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 syncdir(Path(p).parent)
def unpack(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--attempt',required=True);a=ap.parse_args();d=a.directory.resolve()
 dep=json.loads((d/'deployment.json').read_text());manifest=d/'archive-manifest.json';m=json.loads(manifest.read_text())
 inv=unpack(json.loads(a.inventory.read_text()));assert inv['exit_code']==0;inventory=json.loads(inv['stdout']);remote_archive=inventory['files']['terminal-artifacts.tar.gz'];remote_manifest=inventory['files']['terminal-artifacts.manifest.json']
 assert remote_archive['bytes']==m['bytes'] and remote_archive['sha256']==m['sha256']
 assert remote_manifest['bytes']==manifest.stat().st_size and remote_manifest['sha256']==sha(manifest.read_bytes())
 assert m['status']=='FROZEN_TERMINAL_ARCHIVE' and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in m['members'])
 attempts=d/'archive-recovery';attempts.mkdir(exist_ok=True);syncdir(d);run=attempts/a.attempt;run.mkdir(exist_ok=False);syncdir(attempts)
 save(run/'intent.json',dict(source=ref(__file__),manifest=ref(manifest),inventory=ref(a.inventory),chunk_bytes=200000,at_epoch=time.time(),server_read_only=True,retry_count=0))
 cache=attempts/('chunks-'+m['sha256']);cache.mkdir(exist_ok=True);syncdir(attempts)
 sys.path.insert(0,dep['mcp_scripts']);from mcp_client import Client
 c=Client();c.initialize()
 for offset in range(0,m['bytes'],200000):
  length=min(200000,m['bytes']-offset);part=cache/('%09d.bin'%offset);ack=cache/('%09d.json'%offset)
  if part.exists():
   assert ack.exists();r=json.loads(ack.read_text());assert r['offset']==offset and r['bytes']==length and ref(part)==r['part'];continue
  assert not ack.exists()
  try:r=c.request('tools/call',{'name':'read_file','arguments':{'mount_id':dep['mount_id'],'path':dep['remote_relative']+'/terminal-artifacts.tar.gz','offset':offset,'limit':length}})
  except Exception as e:
   save(run/('exception-%09d.json'%offset),dict(exception_type=type(e).__name__,offset=offset,requested_bytes=length));raise
  if r.get('isError'):
   save(run/('tool-error-%09d.json'%offset),r);raise RuntimeError('Read-only transfer error persisted; no retry')
  v=unpack(r);data=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode()
  if len(data)!=length:
   save(run/('length-error-%09d.json'%offset),dict(offset=offset,expected=length,actual=len(data),response=r));raise RuntimeError('Unexpected fragment length; no retry')
  with part.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  syncdir(cache);save(ack,dict(offset=offset,bytes=length,part=ref(part)))
  print(json.dumps(dict(offset=offset,bytes=length,persisted=True)),flush=True)
 data=b''.join((cache/('%09d.bin'%offset)).read_bytes() for offset in range(0,m['bytes'],200000));assert len(data)==m['bytes'] and sha(data)==m['sha256']
 archive=d/'terminal-artifacts.tar.gz'
 if archive.exists():assert ref(archive)['sha256']==m['sha256']
 else:
  with archive.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 syncdir(d);out=d/'recovered';out.mkdir(exist_ok=True);syncdir(d)
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers();assert len(members)==len(set(x.name for x in members))==len(m['members']) and {x.name for x in members}==set(m['members'])
  for member in members:
   assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
   b=tar.extractfile(member).read();expected=m['members'][member.name];assert len(b)==expected['bytes'] and sha(b)==expected['sha256']
   p=out/member.name;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert not p.is_symlink() and p.read_bytes()==b
   else:
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
   syncdir(p.parent)
 for directory in sorted([p for p in out.rglob('*') if p.is_dir()],key=lambda x:len(x.parts),reverse=True):syncdir(directory)
 syncdir(out);syncdir(d)
 report=dict(status='ALL_TERMINAL_MEMBERS_RESTORED_EXACT_SIZE_SHA',source=ref(__file__),archive_sha256=m['sha256'],archive_bytes=m['bytes'],members=len(m['members']),exit_code=m['terminal_exit_code'],existing_remote_archive_reused=True,no_new_archive_or_NN=True,manifest=ref(manifest),inventory=ref(a.inventory),attempt=str(run))
 save(run/'complete.json',report);save(d/'collection-recovered.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
