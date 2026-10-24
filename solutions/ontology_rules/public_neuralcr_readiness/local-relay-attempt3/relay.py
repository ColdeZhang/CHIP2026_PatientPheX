"""One bounded local official Range -> MCP relay. No signed URL is logged."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,os,shlex,sys,threading,time,urllib.request,datetime
R=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
PIECE=524288; BLOCK=8388608

def sha(b):return hashlib.sha256(b).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def chunks(gaps):
 return [(x,min(x+BLOCK,end)) for start,end in gaps for x in range(start,end,BLOCK)]
def main():
 plan=json.loads((R/'plan.json').read_text()); assert not (R/'state.json').exists()
 for name,h in plan['sources'].items():assert sha((R/name).read_bytes())==h
 started=time.monotonic(); deadline=started+plan['budget_seconds'];local=threading.local();lock=threading.Lock();receipts=[];failures=[]
 state={'status':'running','started_at':stamp(),'plan_sha256':sha((R/'plan.json').read_bytes()),'assets':[],'max_workers':8}
 def save():
  with lock:
   state.update(elapsed_seconds=time.monotonic()-started,remote_verified_new_bytes=sum(x['bytes'] for x in receipts),receipt_count=len(receipts),failed_fragments=len(failures));p=R/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(R/'state.json')
 def call(name,args):
  if time.monotonic()+65>=deadline:raise TimeoutError('budget_reserve')
  if not hasattr(local,'client'):local.client=Client();local.client.initialize()
  result=local.client.request('tools/call',{'name':name,'arguments':{'mount_id':plan['mount_id'],**args}})
  if result.get('isError'):raise RuntimeError('MCP_tool_error')
  return result['structuredContent']
 def record(kind,value):
  with lock:
   with (R/'receipts.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,**value},separators=(',',':'))+'\n');f.flush()
   (receipts if kind=='piece' else failures).append(value)
 def put(path,data):
  assert 0<len(data)<=PIECE
  r=call('write_file',{'path':path,'content':base64.b64encode(data).decode(),'encoding':'base64','create_parents':True})
  assert r['sha256']==sha(data) and r['size']==len(data)
  return r
 def worker(asset,bounds):
  start,end=bounds;name=asset['filename'];p=R/'local-chunks'/f'{name}.{start}-{end}.part';p.parent.mkdir(exist_ok=True);assert not p.exists()
  errors=[]
  # Each retry requests only the exact verified prefix remainder, never earlier bytes.
  for attempt in range(3):
   have=p.stat().st_size if p.exists() else 0
   if have==end-start:break
   if time.monotonic()+65>=deadline:break
   try:
    a=start+have; req=urllib.request.Request(asset['url'],headers={'Range':f'bytes={a}-{end-1}','If-Range':asset['etag'],'User-Agent':'NCR-readiness'})
    with urllib.request.urlopen(req,timeout=20) as response:
     assert response.status==206 and response.headers.get('ETag')==asset['etag']
     assert response.headers.get('Content-Range')==f'bytes {a}-{end-1}/{asset["expected_bytes"]}'
     with p.open('ab') as out:
      while have<end-start:
       if time.monotonic()+65>=deadline:raise TimeoutError('budget_reserve')
       b=response.read(min(65536,end-start-have))
       if not b:raise EOFError('range_short_read')
       out.write(b);have+=len(b)
   except Exception as e:errors.append(type(e).__name__)
  have=p.stat().st_size if p.exists() else 0;uploaded=0
  if have:
   with p.open('rb') as f:
    while True:
     data=f.read(PIECE)
     if not data:break
     offset=start+uploaded;rp=plan['remote_root']+f'/pieces/{name}.{offset}-{offset+len(data)}.bin';success=False;pe=[]
     for attempt in range(3):
      try:
       result=put(rp,data);success=True;break
      except Exception as e:pe.append(type(e).__name__)
     if success:record('piece',{'filename':name,'start':offset,'bytes':len(data),'sha256':sha(data),'path':rp.removeprefix(plan['remote_root']+'/'),'local_path':str(p),'local_offset':uploaded})
     else:record('failure',{'filename':name,'start':offset,'bytes':len(data),'errors':pe,'local_retained':str(p)})
     uploaded+=len(data)
  if have<end-start:record('failure',{'filename':name,'start':start+have,'bytes':end-start-have,'errors':errors,'local_retained':str(p)})
  save()
 def finalize(asset):
  new=[x for x in receipts if x['filename']==asset['filename']]
  segments=asset['existing_segments']+[{k:x[k] for k in ['path','start','bytes','sha256']} for x in new]
  pos=0
  for s in sorted(segments,key=lambda x:x['start']):
   if s['start']!=pos:return None
   pos+=s['bytes']
  if pos!=asset['expected_bytes']:return None
  manifest={'filename':asset['filename'],'expected_bytes':pos,'asset_id':asset['asset_id'],'etag':asset['etag'],'segments':segments}
  fname=asset['filename']+'.segments.json';payload=json.dumps(manifest,separators=(',',':')).encode();assert len(payload)<=PIECE
  (R/fname).write_bytes(payload);put(plan['remote_root']+'/'+fname,payload)
  command=shlex.quote(plan['python'])+' remote_launch.py '+shlex.quote(fname)
  job=call('start_process',{'command':command,'cwd':plan['remote_root'],'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}})
  (R/(asset['filename']+'.assembly-job.json')).write_text(json.dumps(job,indent=2)+'\n')
  while True:
   out=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':2048})
   if not out['running']:break
   time.sleep(2)
  rr=call('read_file',{'path':plan['remote_root']+'/'+asset['filename']+'.assembly.json','offset':0,'limit':524288})
  content=rr['content'];content=base64.b64decode(content) if rr.get('encoding')=='base64' else content.encode()
  (R/(asset['filename']+'.assembly.json')).write_bytes(content);result=json.loads(content);assert result['status']=='complete_assembled_and_verified';return result
 try:
  # Shared executor is the only worker pool. CPU assembly runs after workers drain.
  for source in ['remote_finalize.py','remote_launch.py','plan.json']:
   put(plan['remote_root']+'/'+source,(R/source).read_bytes())
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
   for asset in plan['assets']:
    state['phase']=asset['filename'];save()
    futures=[executor.submit(worker,asset,x) for x in chunks(asset['missing_ranges_half_open'])]
    for f in concurrent.futures.as_completed(futures):
     try:f.result()
     except Exception as e:record('failure',{'phase':asset['filename'],'worker_error_type':type(e).__name__})
    if time.monotonic()+190>=deadline:state['status']='incomplete_budget_preserved';break
    result=finalize(asset)
    if result is None:state['status']='incomplete_fragments_preserved';break
    state['assets'].append(result);save()
    if asset['filename']=='HP.tar.gz' and not result['continue_fastText']:state['status']='parameter_gate_rejects_even_BioBERT_replacement';break
   else:state['status']='complete_two_official_assets_verified_native_not_run'
 except Exception as e:state.update(status='failed_preserved',error_type=type(e).__name__)
 finally:state['finished_at']=stamp();save()
 if state['status'] not in ['complete_two_official_assets_verified_native_not_run','parameter_gate_rejects_even_BioBERT_replacement']:raise SystemExit(2)
if __name__=='__main__':main()
