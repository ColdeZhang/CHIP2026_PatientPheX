"""Bounded private copy of one pinned completed state; never removes its source."""
from datetime import datetime,timezone
from http.server import HTTPServer,BaseHTTPRequestHandler
import hashlib,json,os,shutil,sys,time,urllib.request
from pathlib import Path
PROJECT=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def main():
 spec=json.loads(os.environ['RECOVERY_SPEC']);root=Path(spec['source_directory']);out=Path(spec['target_directory']);files=spec['files']
 assert PROJECT/'case-retrieval/qwen35-preparation' in root.parents
 assert PROJECT/'research/storage-recovery-20261024' in out.parents
 assert sorted(files)==['adapter/adapter_config.json','checkpoint.json','training_state.pt']
 prefix='/'+os.environ['RECOVERY_PREFIX']+'/'
 if sys.argv[1]=='serve':
  for n,r in files.items():
   p=root/n;s=p.stat();assert not p.is_symlink() and s.st_uid==os.getuid()==1045
   assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'] and digest(p)==r['sha256']
  served=set()
  class Handler(BaseHTTPRequestHandler):
   def do_GET(self):
    if self.client_address[0]!='10.5.2.100' or not self.path.startswith(prefix):self.send_error(403);return
    name=self.path[len(prefix):]
    if name=='ready':self.send_response(200);self.send_header('Content-Length','5');self.end_headers();self.wfile.write(b'READY');return
    if name not in files:self.send_error(404);return
    p=root/name;self.connection.settimeout(30);self.send_response(200);self.send_header('Content-Length',str(files[name]['bytes']));self.end_headers()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(4*1024*1024),b''):self.wfile.write(b)
    self.wfile.flush();served.add(name)
   def log_message(self,*a):pass
  server=HTTPServer(('10.5.4.7',51963),Handler);server.timeout=.25;end=time.monotonic()+50
  try:
   while len(served)<3 and time.monotonic()<end:server.handle_request()
  finally:server.server_close()
  print(json.dumps(dict(status='SERVER_CLOSED',served=sorted(served),source_preserved=True)))
 elif sys.argv[1]=='receive':
  assert not out.exists() and shutil.disk_usage(PROJECT).free>1024**3
  base='http://10.5.4.7:51963'+prefix;ready=False
  for _ in range(20):
   try:
    with urllib.request.urlopen(base+'ready',timeout=1) as f:ready=f.read()==b'READY'
   except OSError:time.sleep(.5)
   if ready:break
  assert ready;out.mkdir(parents=True);start=time.monotonic();received={}
  for name,r in files.items():
   p=out/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part')
   with urllib.request.urlopen(base+name,timeout=25) as response,tmp.open('xb') as f:
    for b in iter(lambda:response.read(4*1024*1024),b''):f.write(b)
    f.flush();os.fsync(f.fileno())
   assert tmp.stat().st_size==r['bytes'] and digest(tmp)==r['sha256'];tmp.rename(p);syncdir(p.parent)
   assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256']
   received[name]=dict(path=str(p),bytes=r['bytes'],sha256=r['sha256'],reopened_sha256=r['sha256'],fsynced=True)
  cursor=out
  while True:
   syncdir(cursor)
   if cursor==PROJECT:break
   cursor=cursor.parent
  receipt=dict(status='THREE_ORIGINAL_FILES_DURABLE_REOPENED',at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,files=received,source_directory=str(root),target_directory=str(out),parent_chain_fsynced_through=str(PROJECT),source_preserved=True,scope='State relocation only; unchanged adapter/base recovery dependencies remain in original checkpoint metadata; not a standalone checkpoint bundle')
  q=out/'migration-receipt.json'
  with q.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  syncdir(out);syncdir(out.parent)
  print(json.dumps(dict(status=receipt['status'],receipt=str(q),bytes=q.stat().st_size,sha256=digest(q),seconds=receipt['seconds'])))
 else:raise ValueError('Only serve or receive; source deletion is a separately verified step')
if __name__=='__main__':main()
