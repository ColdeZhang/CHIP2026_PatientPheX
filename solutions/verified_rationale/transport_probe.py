"""Single authenticated 32MiB private-network read; no model transfer or GPU."""
import argparse,hashlib,hmac,json,os,socket,time,urllib.request
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path

def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def serve(args):
 p=Path(args.path);st=p.stat();n=32*1024**2
 with p.open('rb') as f:data=f.read(n)
 assert len(data)==n;sha=hashlib.sha256(data).hexdigest();secret=os.environ['RATIONALE_TRANSFER_AUTH'];assert len(secret)>=32
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*_):pass
  def do_GET(self):
   if self.path!='/probe' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+secret):self.send_error(403);return
   self.connection.settimeout(15);self.send_response(200);self.send_header('Content-Length',str(n));self.send_header('X-SHA256',sha);self.end_headers();self.wfile.write(data)
 server=HTTPServer(('0.0.0.0',int(os.environ['PORT'])),Handler);server.timeout=120
 put('ready.json',dict(pid=os.getpid(),bytes=n,sha256=sha,path=str(p),source_stat={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino},port=int(os.environ['PORT']),max_requests=1,timeout_seconds=120))
 server.handle_request();server.server_close();put('closed.json',dict(pid=os.getpid(),closed=True));print('closed')
def receive(args):
 start=time.monotonic();op=urllib.request.build_opener(urllib.request.ProxyHandler({}));req=urllib.request.Request(os.environ['RATIONALE_TRANSFER_URL'],headers={'Authorization':'Bearer '+os.environ['RATIONALE_TRANSFER_AUTH']})
 with op.open(req,timeout=30) as r:
  assert int(r.headers['Content-Length'])==32*1024**2;expected=r.headers['X-SHA256'];b=r.read(32*1024**2+1)
 assert len(b)==32*1024**2 and hashlib.sha256(b).hexdigest()==expected
 dt=time.monotonic()-start;v=dict(status='PASS_32MiB_READ_ONLY_NETWORK_PROBE',bytes=len(b),sha256=expected,seconds=dt,bytes_per_second=len(b)/dt,received_in_memory_only=True,full_model_transfer=False,GPU=False);put('receive.json',v);print(json.dumps(v))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('serve','receive'));p.add_argument('--path');a=p.parse_args();globals()[a.mode](a)
