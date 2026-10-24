"""No-network fixture: immutable prefix, partial resume, fragment retry and gate."""
from pathlib import Path
import importlib.util,tempfile,json,hashlib,io,time,base64
R=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('relay',R/'relay.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
start=time.monotonic();source=(R/'relay.py').read_bytes();payload=bytes(range(256))*5000;old=123;requests=[];stored={};attempts={};injected=[False]
class Response:
 status=206
 def __init__(self,a,b): self.a=a;self.b=b;self.pos=a;self.headers={'ETag':'fixture','Content-Range':f'bytes {a}-{b}/{len(payload)}'}
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read(self,n):
  if self.pos>self.a and not injected[0]:injected[0]=True;raise TimeoutError('fixture interrupted stream')
  b=payload[self.pos:min(self.pos+n,self.b+1)];self.pos+=len(b);return b
class Client:
 def initialize(self):pass
 def request(self,method,p):
  name=p['name'];a=p['arguments']
  if name=='write_file':
   d=base64.b64decode(a['content']);assert len(d)<=m.PIECE;key=a['path'];attempts[key]=attempts.get(key,0)+1
   if '/pieces/' in key and attempts[key]==1:raise TimeoutError('fixture one fragment failure')
   stored[key]=d;z={'sha256':hashlib.sha256(d).hexdigest(),'size':len(d)}
  elif name=='start_process':z={'id':'fixture'}
  elif name=='read_process_output':z={'running':False}
  elif name=='read_file':
   manifest=json.loads(stored['remote/HP.tar.gz.segments.json']);actual=payload[:old]
   for x in sorted(manifest['segments'],key=lambda x:x['start'])[1:]:
    assert x['start']==len(actual);b=stored['remote/'+x['path']];assert hashlib.sha256(b).hexdigest()==x['sha256'];actual+=b
   assert actual==payload;z={'encoding':'utf8','content':json.dumps({'status':'complete_assembled_and_verified','continue_fastText':False})}
  return {'structuredContent':z}
def urlopen(req,timeout):
 a,b=map(int,req.headers['Range'].removeprefix('bytes=').split('-'));assert a>=old;requests.append([a,b]);return Response(a,b)
with tempfile.TemporaryDirectory() as tmp:
 m.R=Path(tmp);m.Client=Client;m.urllib.request.urlopen=urlopen
 for name in ['relay.py','remote_finalize.py','remote_launch.py']: (m.R/name).write_bytes((R/name).read_bytes())
 plan={'sources':{},'budget_seconds':3600,'mount_id':'fixture','remote_root':'remote','remote_project':'p','python':'python','assets':[{'filename':'HP.tar.gz','url':'https://fixture.invalid','etag':'fixture','asset_id':1,'expected_bytes':len(payload),'existing_segments':[{'path':'old','start':0,'bytes':old,'sha256':hashlib.sha256(payload[:old]).hexdigest()}],'missing_ranges_half_open':[[old,len(payload)]]},{'filename':'pmc_model_new.bin'}]}
 (m.R/'plan.json').write_text(json.dumps(plan));m.main();state=json.loads((m.R/'state.json').read_text());assert state['status']=='parameter_gate_rejects_even_BioBERT_replacement';assert injected[0] and len(requests)==2 and requests[1][0]==old+65536
 report={'status':'pass','fixture':'midstream resume, per-fragment write retry, immutable original prefix excluded, max512KiB, HP gate blocks fastText','network_calls':0,'elapsed_seconds':time.monotonic()-start,'relay_sha256':hashlib.sha256(source).hexdigest(),'range_requests':requests,'new_bytes':state['remote_verified_new_bytes']}
(R/'transport-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
