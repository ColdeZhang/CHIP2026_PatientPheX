"""Bounded allowlist-only private copy, streamed SHA/fsync/reopen; no model calls."""
import argparse,hashlib,http.server,json,os,socketserver,tarfile,time,urllib.request
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sync(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 sync(p.parent);sync(p.parent.parent)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','serve','fetch']);ap.add_argument('--plan',required=True);ap.add_argument('--port',type=int);ap.add_argument('--manifest-sha');ap.add_argument('--admit-before',type=float);a=ap.parse_args();p=json.load(open(a.plan));root=Path(a.plan).resolve().parent
 if a.action=='prepare':
  rt=root/'runtime.tar.gz';assert not rt.exists();members=[]
  with tarfile.open(rt,'w:gz',compresslevel=1) as tar:
   for n in p['runtime_names']:
    q=Path(p['runtime_source'])/n;assert q.exists()
    for f in sorted(q.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith('.pyc'):
      name=str(f.relative_to(q.parent));assert not f.is_symlink();tar.add(f,arcname=name,recursive=False);members.append(dict(name=name,bytes=f.stat().st_size,sha256=sha(f)))
  with rt.open('rb') as f:os.fsync(f.fileno())
  files={**p['source_files'],'runtime.tar.gz':str(rt)};manifest={}
  for name,path in files.items():
   q=Path(path);manifest[name]=dict(bytes=q.stat().st_size,sha256=sha(q))
   if name in p['expected']:assert manifest[name]==p['expected'][name]
  save(root/'source-manifest.json',dict(files=manifest,runtime_members=members,source_paths=files,model_calls=0));print(json.dumps({'manifest_sha256':sha(root/'source-manifest.json'),'files':manifest,'runtime_members':len(members)}),flush=True)
 elif a.action=='serve':
  assert time.time()<a.admit_before;save(root/'serve-claim.json',dict(pid=os.getpid(),started=time.time(),admit_before=a.admit_before))
  manifest=json.load(open(root/'source-manifest.json'));files=manifest['source_paths'];deadline=time.monotonic()+450
  class H(http.server.BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    n=self.path.lstrip('/')
    if n=='manifest.json':f=root/'source-manifest.json'
    elif n in files:f=Path(files[n])
    else:self.send_error(404);return
    self.send_response(200);self.send_header('Content-Length',str(f.stat().st_size));self.end_headers()
    with f.open('rb') as h:
     for b in iter(lambda:h.read(1024*1024),b''):self.wfile.write(b)
  class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):daemon_threads=True
  server=Server((p['source_ip'],0),H);server.timeout=1
  save(root/'serve-ready.json',dict(pid=os.getpid(),port=server.server_port,manifest_sha256=sha(root/'source-manifest.json'),deadline_unix=time.time()+450))
  try:
   while time.monotonic()<deadline:server.handle_request()
  finally:server.server_close();save(root/'serve-closed.json',dict(pid=os.getpid(),closed=time.time()))
 else:
  assert time.time()<a.admit_before;save(root/'fetch-claim.json',dict(pid=os.getpid(),started=time.time(),admit_before=a.admit_before));start=time.monotonic()
  url='http://'+p['source_ip']+':'+str(a.port)+'/';raw=urllib.request.urlopen(url+'manifest.json',timeout=30).read();assert hashlib.sha256(raw).hexdigest()==a.manifest_sha;m=json.loads(raw)
  target=root/'assets';target.mkdir(exist_ok=False);sync(root)
  for n,r in m['files'].items():
   assert '/' not in n
   path=target/n;h=hashlib.sha256();size=0
   with urllib.request.urlopen(url+n,timeout=120) as src,path.open('xb') as dst:
    while True:
     b=src.read(1024*1024)
     if not b:break
     dst.write(b);h.update(b);size+=len(b)
    dst.flush();os.fsync(dst.fileno())
   assert size==r['bytes'] and h.hexdigest()==r['sha256'] and sha(path)==r['sha256']
  sync(target);runtime=root/'runtime';runtime.mkdir();sync(root)
  with tarfile.open(target/'runtime.tar.gz') as tar:
   assert set(tar.getnames())=={x['name'] for x in m['runtime_members']}
   for r in m['runtime_members']:
    n=r['name'];assert not n.startswith('/') and '..' not in Path(n).parts
    b=tar.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];q=runtime/n;q.parent.mkdir(parents=True,exist_ok=True)
    with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert sha(q)==r['sha256']
  for d,ds,fs in os.walk(runtime,topdown=False):sync(d)
  sync(root);save(root/'copy-complete.json',dict(status='ALL_ASSETS_SOURCE_DESTINATION_SHA_FSYNC_REOPEN',manifest_sha256=a.manifest_sha,files=m['files'],runtime_members=len(m['runtime_members']),seconds=time.monotonic()-start,model_calls=0));print(json.dumps({'status':'COMPLETE','seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
