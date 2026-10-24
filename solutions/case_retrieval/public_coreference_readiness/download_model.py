"""Pinned public F-coref download, resumable within one hard 600-second budget."""
import hashlib,json,os,pathlib,time,urllib.request,datetime
P=pathlib.Path.cwd();plan=json.loads((P/'download-plan.json').read_text());start=time.monotonic();deadline=start+plan['download_cap_seconds'];assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
D=P/'model';D.mkdir(exist_ok=True);state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=plan['download_cap_seconds'])).isoformat(),'files':{}}
def write(): (P/'download-state.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify(p,a):
 if p.stat().st_size!=a['bytes']:return False
 if a.get('sha256'):return sha(p)==a['sha256']
 h=hashlib.sha1();h.update(b'blob '+str(a['bytes']).encode()+b'\0');h.update(p.read_bytes());return h.hexdigest()==a['git_blob_sha1']
write()
try:
 for a in plan['files']:
  name=a['name'];dest=D/name;partial=D/(name+'.partial')
  if dest.exists():
   if not verify(dest,a):raise RuntimeError('Existing target failed identity: '+name)
   state['files'][name]={'status':'reused_verified_target','sha256':sha(dest),'bytes':dest.stat().st_size};write();continue
  # Only explicit author/revision cache locations; never infer identity from filename.
  reused=False
  for root in plan['cache_snapshot_roots']:
   cached=pathlib.Path(root)/name
   if cached.is_file() and verify(cached,a):dest.symlink_to(cached.resolve());reused=True;break
  if not reused:
   attempts=0
   while not (partial.exists() and partial.stat().st_size==a['bytes']):
    if time.monotonic()>=deadline:raise TimeoutError('600s budget exhausted')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>a['bytes']:raise RuntimeError('Oversized partial')
    headers={'Range':f'bytes={offset}-'} if offset else {}
    try:
     url=a['url']+'?download=true&request_id='+str(time.time_ns())
     with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=min(25,max(1,deadline-time.monotonic()))) as r:
      if offset and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Invalid resumed Content-Range')
      if not offset and r.status not in (200,206):raise RuntimeError('Unexpected HTTP status')
      with partial.open('ab') as f:
       while True:
        if time.monotonic()>=deadline:raise TimeoutError('600s budget exhausted')
        chunk=r.read(1024*1024)
        if not chunk:break
        if f.tell()+len(chunk)>a['bytes']:raise RuntimeError('Response longer than pinned file')
        f.write(chunk)
      state['files'][name]={'status':'partial','bytes':partial.stat().st_size};write()
     if partial.stat().st_size!=a['bytes']:raise RuntimeError('Short response')
    except Exception as e:
     attempts+=1;state['files'][name]={'status':'partial_failure','error_type':type(e).__name__,'attempt':attempts,'bytes':partial.stat().st_size if partial.exists() else 0};write()
     if attempts>=3:raise
   if not verify(partial,a):raise RuntimeError('Pinned checksum mismatch: '+name)
   partial.rename(dest)
  state['files'][name]={'status':'reused_verified_cache' if reused else 'complete','bytes':dest.stat().st_size,'sha256':sha(dest),'source_url':a['url']};write()
 state['status']='complete_verified_all_files';state['seconds']=time.monotonic()-start;write();(P/'download-complete.json').write_text(json.dumps(state,indent=2)+'\n')
except Exception as e:
 state['status']='failed';state['error_type']=type(e).__name__;state['seconds']=time.monotonic()-start;write();raise
