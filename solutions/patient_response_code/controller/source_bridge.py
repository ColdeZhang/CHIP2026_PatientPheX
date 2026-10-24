"""Same immutable deployment archive via bounded private .100 HTTP; CPU only."""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
M100='5a9955da-5240-41cc-9e52-05d3d8da8a3c';MA='476f81cf-09e7-445b-a7f0-d0e12b12e50f';REL='patient-response-code/source-bridge-v1';ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REL
SERVER='''from pathlib import Path
import hashlib,json,os,time,subprocess
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=Path(__file__).parent;m=json.loads((p/'identity.json').read_text());raw=(p/'deployment.tar.gz').read_bytes();assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256'];assert not (p/'started.json').exists();start=time.time()
addresses=subprocess.check_output(['hostname','-I'],text=True).split();address=next(x for x in addresses if x.startswith('10.5.'))
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  if self.path!='/deployment.tar.gz':self.send_error(404);return
  self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
s=ThreadingHTTPServer((address,0),H);s.timeout=.5
v=dict(pid=os.getpid(),address=address,port=s.server_address[1],started_unix=start,deadline_unix=start+600,archive=m,no_model=True)
with (p/'started.json').open('x') as f:json.dump(v,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
print(json.dumps(v),flush=True)
while time.time()<start+600:s.handle_request()
s.server_close()
with (p/'complete.json').open('x') as f:json.dump(dict(ended_unix=time.time(),no_model=True),f);f.flush();os.fsync(f.fileno())
'''

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','serve','ready','fetch']);ap.add_argument('--controller',type=Path,required=True);a=ap.parse_args();C=a.controller;B=C/'source-bridge';B.mkdir(exist_ok=True);m=json.loads((C/'deployment-manifest.json').read_text());data=Path(m['archive']['path']).read_bytes();assert sha(data)==m['archive']['sha256'];p=json.loads((C/'controller-plan.json').read_text());cl=Client();cl.initialize()
 def call(mount,n,args,label):
  pref=B/(label+'-'+str(time.time_ns()));save(pref.with_suffix('.intent.json'),dict(tool=n,arguments={k:v for k,v in args.items() if k!='content'}));r=cl.request('tools/call',dict(name=n,arguments=dict(mount_id=mount,**args)));save(pref.with_suffix('.tool.json'),r);assert not r.get('isError'),str(r);return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 def remote(mount,code,param,label):
  r=call(mount,'run_command',dict(command='python3 -c '+shlex.quote('P='+repr(param)+'\n'+code),timeout_seconds=30),label);assert r['exit_code']==0;return json.loads(r['stdout'])
 if a.action=='stage':
  code='''from pathlib import Path
import os,json,hashlib
p=Path(P['root']);assert os.getuid()==1006;p.mkdir(parents=True,exist_ok=True,mode=0o700);assert p.stat().st_uid==1006;os.chmod(p,0o700);assert not (p/'started.json').exists()
print(json.dumps({q.name:dict(bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in p.glob('part-*')}))
'''
  inv=remote(M100,code,dict(root=ROOT),'inventory');parts=[]
  for i,offset in enumerate(range(0,len(data),32768)):
   raw=data[offset:offset+32768];name='part-%03d'%i;item=dict(bytes=len(raw),sha256=sha(raw));parts.append(dict(name=name,**item))
   if name in inv:assert inv[name]==item
   else:
    r=call(M100,'write_file',dict(path=REL+'/'+name,content=base64.b64encode(raw).decode(),encoding='base64',create_parents=False),name);assert r['size']==len(raw) and r['sha256']==sha(raw)
  code='''from pathlib import Path
import json,hashlib,os
p=Path(P['root'])
def put(n,b):
 q=p/n
 if q.exists():assert q.read_bytes()==b
 else:
  with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for r in P['parts']:
 b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
b=b''.join((p/r['name']).read_bytes() for r in P['parts']);assert len(b)==P['identity']['bytes'] and hashlib.sha256(b).hexdigest()==P['identity']['sha256'];put('deployment.tar.gz',b);put('identity.json',(json.dumps(P['identity'],indent=2)+chr(10)).encode());put('server.py',P['server'].encode());compile(P['server'],'server.py','exec');fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd);print(json.dumps(dict(status='READY_EXACT_ARCHIVE_FSYNC',archive=P['identity'],server_sha256=hashlib.sha256(P['server'].encode()).hexdigest())))
'''
  r=remote(M100,code,dict(root=ROOT,parts=parts,identity=dict(bytes=len(data),sha256=sha(data)),server=SERVER),'verify');save(B/'staged.json',r);print(json.dumps(r))
 elif a.action=='serve':
  assert (B/'staged.json').is_file() and not (B/'start-intent.json').exists();save(B/'start-intent.json',dict(at_unix=time.time(),archive=m['archive'],cap_seconds=600,grace_seconds=10,no_model=True));r=call(M100,'start_process',dict(command='timeout --signal=TERM --kill-after=10s 610s python3 -u server.py',cwd=REL,env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}),'start');save(B/'job.json',r);print(json.dumps(r))
 elif a.action=='ready':
  r=remote(M100,"from pathlib import Path\nimport json\np=Path(P['root']);print((p/'started.json').read_text() if (p/'started.json').exists() else '{}')",dict(root=ROOT),'ready');assert r['archive']==dict(bytes=len(data),sha256=sha(data)) and time.time()<r['deadline_unix'];save(B/'ready.json',r);print(json.dumps(r))
 else:
  ready=json.loads((B/'ready.json').read_text());assert time.time()<ready['deadline_unix'];assert not (C/'campaign.json').exists();url='http://'+ready['address']+':'+str(ready['port'])+'/deployment.tar.gz'
  code='''from pathlib import Path
import hashlib,json,os,time,urllib.request
p=Path(P['root']);assert not (p/'campaign.json').exists() and not (p/'control').exists();assert time.time()<P['deadline']
with urllib.request.urlopen(P['url'],timeout=min(20,P['deadline']-time.time())) as r:b=r.read(P['bytes']+1)
assert len(b)==P['bytes'] and hashlib.sha256(b).hexdigest()==P['sha256'];reused=[];written=[]
for i,start in enumerate(range(0,len(b),32768)):
 chunk=b[start:start+32768];q=p/('part-%03d'%i)
 if q.exists():assert not q.is_symlink() and q.read_bytes()==chunk;reused.append(q.name)
 else:
  with q.open('xb') as f:f.write(chunk);f.flush();os.fsync(f.fileno())
  written.append(q.name)
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd);print(json.dumps(dict(status='EXACT_ORIGINAL_DEPLOYMENT_PARTS_FILLED',archive_sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),reused=reused,written=written,no_model=True)))
'''
  r=remote(MA,code,dict(root=p['remote_absolute'],url=url,deadline=ready['deadline_unix'],bytes=len(data),sha256=sha(data)),'fetch-original-pieces');save(B/'fetch-complete.json',r);print(json.dumps(r))
if __name__=='__main__':main()
