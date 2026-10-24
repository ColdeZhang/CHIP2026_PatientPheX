"""One bounded official NeuralCR release1.1 download; no extraction, model or corpus access."""
from pathlib import Path
import os,json,time,hashlib,urllib.request,concurrent.futures,threading,datetime
R=Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';D=R/'downloads';D.mkdir(exist_ok=False)
plan=json.loads((R/'download-plan.json').read_text());assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['source_sha256']['download_assets.py']
start=time.monotonic();deadline=start+600;lock=threading.Lock();state={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'budget_seconds':600,'sources':plan['assets'],'files':{},'no_model_or_NN_loaded':True}
def save():
 with lock:(D/'download-state.json').write_text(json.dumps(state,indent=2)+'\n')
def fetch(a):
 name=a['filename'];p=D/(name+'.part');h=hashlib.sha256();offset=0;etag=None;fails=0;started=time.monotonic()
 with p.open('xb') as out:
  while offset<a['bytes']:
   if time.monotonic()>=deadline:raise TimeoutError('download600 budget exhausted')
   last=min(a['bytes']-1,offset+16*1024*1024-1);request=urllib.request.Request(a['url'],headers={'User-Agent':'PatientPheX-public-NeuralCR-readiness/1.0','Range':f'bytes={offset}-{last}'})
   try:
    with urllib.request.urlopen(request,timeout=min(30,max(1,deadline-time.monotonic()))) as q:
     assert q.status==206 and q.headers.get('Content-Range')==f'bytes {offset}-{last}/{a["bytes"]}',(name,q.status,q.headers.get('Content-Range'))
     current=q.headers.get('ETag');assert etag is None or current==etag,(name,'ETag changed');etag=current
     remaining=last-offset+1
     while remaining:
      if time.monotonic()>=deadline:raise TimeoutError('download600 budget exhausted')
      b=q.read(min(1024*1024,remaining));assert b,(name,'short range response');out.write(b);h.update(b);offset+=len(b);remaining-=len(b)
    out.flush();state['files'][name]={'downloaded_bytes':offset,'expected_bytes':a['bytes'],'etag':etag,'seconds':time.monotonic()-started};save()
   except Exception:
    fails+=1;out.flush();state['files'][name]={'downloaded_bytes':offset,'expected_bytes':a['bytes'],'network_failures':fails,'seconds':time.monotonic()-started};save()
    if fails>=3 or time.monotonic()>=deadline:raise
    time.sleep(1)
 assert p.stat().st_size==a['bytes'];final=D/name;p.rename(final);result={'filename':name,'bytes':offset,'sha256':h.hexdigest(),'etag':etag,'network_failures':fails,'seconds':time.monotonic()-started,'official_asset_id':a['asset_id'],'source_url':a['url']}
 state['files'][name]=result;save();return result
save()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(fetch,plan['assets']))
 state.update(status='complete_official_assets_not_loaded',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,outputs=results);assert state['elapsed_seconds']<=600;save();(D/'complete.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),elapsed_seconds=time.monotonic()-start);save();raise
