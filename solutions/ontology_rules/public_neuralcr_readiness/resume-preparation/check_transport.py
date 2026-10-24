"""Local synthetic transport fixtures. No external requests or model/data access."""
from pathlib import Path
import hashlib,json,os,runpy,shutil,tempfile,time,urllib.request,re,threading
R=Path(__file__).resolve().parent
source=(R/'resume_assets.py').read_bytes(); digest=hashlib.sha256(source).hexdigest()
start=time.monotonic(); original=urllib.request.urlopen
payload={n:(bytes(range(256))*((20*1024*1024)//256)+b'end') for n in ['HP.tar.gz','pmc_model_new.bin']}
results=[]
for invalid_etag in [False,True]:
 with tempfile.TemporaryDirectory(prefix='ncr-resume-fixture-') as temporary:
  root=Path(temporary);(root/'resume_assets.py').write_bytes(source)
  assets=[]
  for i,(name,data) in enumerate(payload.items()):
   prefix=root/(name+'.prefix');prefix.write_bytes(data[:12345+i])
   assets.append({'filename':name,'url':'https://github.com/data-team-uhn/NeuralCR/releases/download/1.1/'+name,'asset_id':i+1,'etag':'"fixed"','bytes':len(data),'prefix':{'path':str(prefix),'bytes':prefix.stat().st_size,'sha256':hashlib.sha256(prefix.read_bytes()).hexdigest()}})
  (root/'resume-plan.json').write_text(json.dumps({'status':'authorized_frozen_resume_after_attempt1_exit','workers':4,'chunk_bytes':8*1024*1024,'source_sha256':digest,'prior_job_terminal':True,'budget_seconds':10,'assets':assets}))
  requests=[];lock=threading.Lock();fail=[True]
  class Response:
   status=206
   def __init__(self,name,begin,end):
    self.name=name;self.pos=begin;self.begin=begin;self.end=end
    self.headers={'Content-Range':f'bytes {begin}-{end}/{len(payload[name])}','ETag':'"wrong"' if invalid_etag else '"fixed"'}
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self,n):
    with lock:
     if not invalid_etag and self.pos>self.begin and fail[0]:fail[0]=False;raise OSError('synthetic interrupted range')
    data=payload[self.name][self.pos:min(self.pos+n,self.end+1)];self.pos+=len(data);return data
  def fake_open(request,timeout):
   name=request.full_url.rsplit('/',1)[1];begin,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
   assert request.get_header('If-range')=='"fixed"';assert 0<timeout<=20
   with lock:requests.append((name,begin,end))
   return Response(name,begin,end)
  urllib.request.urlopen=fake_open;os.environ['CUDA_VISIBLE_DEVICES']=''
  error=None
  try:runpy.run_path(str(root/'resume_assets.py'),run_name='__main__')
  except Exception as exc:error=type(exc).__name__
  finally:urllib.request.urlopen=original
  state=json.loads((root/'resume-state.json').read_text())
  for asset in assets:
   assert hashlib.sha256(Path(asset['prefix']['path']).read_bytes()).hexdigest()==asset['prefix']['sha256']
  if invalid_etag:
   assert error and state['status']=='failed_preserved' and not(root/'downloads/complete.json').exists()
  else:
   assert error is None and state['status']=='complete_official_assets_not_loaded' and not fail[0]
   for item in state['outputs']:
    data=(root/'downloads'/item['filename']).read_bytes();assert data==payload[item['filename']]
    assert item['sha256']==hashlib.sha256(data).hexdigest()
  results.append({'case':'reject_changed_ETag' if invalid_etag else 'exact_join_with_midrange_retry','passed':True,'requests':len(requests),'old_prefixes_unchanged':True,'complete':not invalid_etag})
report={'source_sha256':digest,'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'pass_local_synthetic_transport_only','cases':results,'elapsed_seconds':time.monotonic()-start,'no_external_requests':True,'no_model_or_competition_data':True}
(R/'transport-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
