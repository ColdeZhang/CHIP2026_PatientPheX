"""Bounded resumable fixed-revision asset download; stdlib only, no model execution."""
import argparse,concurrent.futures,fcntl,hashlib,json,math,os,signal,struct,sys,threading,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
STOP=threading.Event()
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');b=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(str(tmp),str(path))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
def header(path):
 path=Path(path);size=path.stat().st_size
 with path.open('rb') as f:
  first=f.read(8)
  if len(first)!=8:raise ValueError('short safetensors length')
  n=struct.unpack('<Q',first)[0]
  if not 2<=n<=8*1024*1024 or 8+n>size:raise ValueError('invalid header length')
  d=json.loads(f.read(n).decode('utf-8'),object_pairs_hook=unique)
 spans=[];entries={}
 for k,v in d.items():
  if k=='__metadata__':continue
  shape=v['shape'];dtype=v['dtype'];lo,hi=v['data_offsets']
  if dtype!='BF16' or not isinstance(shape,list) or any(type(x)!=int or x<=0 for x in shape):raise ValueError('unexpected tensor type')
  count=math.prod(shape)
  if type(lo)!=int or type(hi)!=int or lo<0 or hi-lo!=2*count:raise ValueError('offset byte count mismatch')
  spans.append((lo,hi,k));entries[k]={'shape':shape,'dtype':dtype,'numel':count}
 end=0
 for lo,hi,k in sorted(spans):
  if lo!=end:raise ValueError('overlap or unaccounted payload gap')
  end=hi
 if 8+n+end!=size:raise ValueError('payload EOF mismatch')
 return entries,{'file':path.name,'header_bytes':n,'payload_bytes':end,'tensor_keys':len(entries),'dtype':'BF16','contiguous_payload_and_exact_EOF':True}
def verify_headers(root,expected_shapes):
 index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];got={};files=[]
 for name in sorted(set(index.values())):
  entries,summary=header(root/name);files.append(summary)
  for key,entry in entries.items():
   if key in got or index.get(key)!=name:raise ValueError('duplicate or wrong shard key')
   got[key]=entry
 if set(got)!=set(index) or set(got)!=set(expected_shapes):raise ValueError('complete key set mismatch')
 counts={}
 for key,e in expected_shapes.items():
  if got[key]['shape']!=e['shape_derived_from_config_and_native_source'] or got[key]['numel']!=e['numel']:raise ValueError('actual shape differs from frozen formula')
  counts[e['group']]=counts.get(e['group'],0)+got[key]['numel']
 if counts!={'text':8190735360}:raise ValueError('parameter count mismatch')
 if len(got)!=399 or 'lm_head.weight' not in got:raise ValueError('untied head/key count mismatch')
 cfg=json.loads((root/'config.json').read_text())
 if cfg['model_type']!='qwen3' or cfg['architectures']!=['Qwen3ForCausalLM'] or cfg['tie_word_embeddings'] is not False:raise ValueError('wrong native architecture')
 actual_bytes=sum(x['payload_bytes'] for x in files)
 if actual_bytes!=16381470720 or json.loads((root/'model.safetensors.index.json').read_text())['metadata']['total_size']!=actual_bytes:raise ValueError('index payload mismatch')
 return {'status':'ACTUAL_SAFETENSORS_HEADERS_MATCH_FROZEN_399_KEY_QWEN3_FORMULA','tensor_keys':len(got),'dtype':'BF16','counts':counts,'full_bundle_parameters':sum(counts.values()),'text_embedding_and_output_tied_counted_once':False,'separate_lm_head_weight_stored':True,'files':files,'tensor_values_not_decoded':True,'model_not_loaded':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--intent',required=True);a=ap.parse_args();intentpath=Path(a.intent).resolve();base=intentpath.parent;intent=json.loads(intentpath.read_text());deadline=intent['absolute_deadline_epoch'];root=Path(intent['model_dir']);root.mkdir(parents=True,exist_ok=True)
 if os.getuid()!=intent['expected_uid'] or root.stat().st_uid!=os.getuid():raise ValueError('owner mismatch')
 lock=(base/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (base/'started.json').exists():raise ValueError('one-attempt start marker exists; inspect original job, do not relaunch')
 started=time.time();atomic(base/'started.json',{'pid':os.getpid(),'ppid':os.getppid(),'started_epoch':started,'absolute_deadline_epoch':deadline,'remaining_budget_seconds':deadline-started,'uid':os.getuid(),'python':sys.version,'source_sha256':digest(__file__)})
 signal.signal(signal.SIGTERM,lambda s,f:STOP.set());signal.signal(signal.SIGINT,lambda s,f:STOP.set())
 if deadline-started<=0:raise ValueError('deadline already expired')
 assert digest(base/'expected-files.json')==intent['expected_files_sha256'] and digest(__file__)==intent['download_script_sha256'] and digest(base/'derived-tensor-shapes.json')==intent['derived_shapes_sha256'], 'frozen asset source/plan changed'
 plan=json.loads((base/'expected-files.json').read_text());records={};recordlock=threading.Lock()
 def save(name,data):
  with recordlock:records[name]=data
 def abort():
  if STOP.is_set() or time.time()>=deadline-25:raise TimeoutError('absolute budget ending')
 def one(e):
  name=e['name'];p=root/name;part=root/(name+'.part');expect=e['sha256'];size=e['bytes'];st=time.time();attempts=[]
  try:
   abort()
   if p.exists():
    actual=digest(p)
    if p.stat().st_size==size and actual==expect:
     out=dict(name=name,path=str(p),bytes=size,sha256=actual,status='REUSED_FULL_SHA_MATCH',source_url=e['source_url'],attempts=attempts);save(name,out);return out
    raise ValueError('existing final asset differs; do not overwrite')
   if e['kind']=='local_seed':
    raise ValueError('seed absent; deployment must transfer original license bytes first')
   for retry in range(5):
    abort();offset=part.stat().st_size if part.exists() else 0
    if offset>size:raise ValueError('partial larger than expected; preserve for investigation')
    if offset==size:
     actual=digest(part)
     if actual==expect:os.replace(str(part),str(p));break
     raise ValueError('complete partial SHA mismatch; preserve for investigation')
    req=urllib.request.Request(e['source_url'],headers={'User-Agent':'PatientPheX-fixed-assets/1.0','Accept-Encoding':'identity',**({'Range':'bytes=%d-'%offset} if offset else {})})
    attempt={'attempt':retry+1,'started_epoch':time.time(),'resume_offset':offset};attempts.append(attempt)
    try:
     with urllib.request.urlopen(req,timeout=20) as response:
      attempt.update(http_status=response.status,final_host=urllib.parse.urlsplit(response.url).hostname)
      # Never persist redirect query strings or temporary signed URLs.
      if offset:
       if response.status!=206 or not response.headers.get('Content-Range','').startswith('bytes %d-'%offset):raise ValueError('range resume unsupported; old partial retained')
      elif response.status not in (200,206):raise ValueError('unexpected HTTP status')
      with part.open('ab' if offset else 'wb') as f:
       while True:
        abort();b=response.read(4*1024*1024)
        if not b:break
        f.write(b)
        if f.tell()>size:raise ValueError('received more than expected')
        save(name,{'name':name,'status':'DOWNLOADING','bytes_received':f.tell(),'expected_bytes':size,'attempts':list(attempts)})
       f.flush();os.fsync(f.fileno())
     attempt['ended_epoch']=time.time()
     if part.stat().st_size!=size:raise IOError('short response preserved')
     actual=digest(part)
     if actual!=expect:raise ValueError('completed download SHA mismatch')
     os.replace(str(part),str(p));break
    except (urllib.error.URLError,urllib.error.HTTPError,TimeoutError,OSError) as ex:
     attempt.update(error_type=type(ex).__name__,http_code=getattr(ex,'code',None),ended_epoch=time.time())
     if STOP.is_set() or time.time()>=deadline-25 or retry==4:raise
     time.sleep(min(2*(retry+1),8))
   if not p.exists():raise ValueError('no completed asset')
   # digest above is a full post-download local file read, not a transfer-byte counter.
   out={'name':name,'path':str(p),'bytes':p.stat().st_size,'sha256':actual,'status':'DOWNLOADED_FULL_SHA_MATCH','source_url':e['source_url'],'attempts':attempts,'elapsed_seconds':time.time()-st};save(name,out);return out
  except Exception as ex:
   out={'name':name,'status':'PARTIAL_OR_FAILED','error_type':type(ex).__name__,'partial_bytes':part.stat().st_size if part.exists() else 0,'expected_bytes':size,'attempts':attempts};save(name,out);return out
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(one,e) for e in plan['files']]
  last=0
  while not all(f.done() for f in futures):
   if time.time()-last>=20:
    with recordlock:snapshot=dict(records)
    atomic(base/'progress.json',{'pid':os.getpid(),'observed_epoch':time.time(),'absolute_deadline_epoch':deadline,'files':snapshot});last=time.time()
   time.sleep(1)
  outputs=[f.result() for f in futures]
 ok=all(x['status'] in ('REUSED_FULL_SHA_MATCH','DOWNLOADED_FULL_SHA_MATCH') for x in outputs);result={'status':'ASSET_DOWNLOAD_COMPLETE' if ok else 'ASSET_DOWNLOAD_PARTIAL','started_epoch':started,'ended_epoch':time.time(),'elapsed_seconds':time.time()-started,'absolute_deadline_epoch':deadline,'model_dir':str(root),'model_id':intent['model_id'],'MS_revision':intent['MS_revision'],'expected_files_sha256':digest(base/'expected-files.json'),'files':outputs,'no_model_load_no_GPU_no_install':True}
 if ok:
  try:
   abort();shapes=json.loads((base/'derived-tensor-shapes.json').read_text());h=verify_headers(root,shapes);atomic(base/'actual-header-inventory.json',h);result['header_inventory_sha256']=digest(base/'actual-header-inventory.json');result['status']='ASSET_FILES_AND_ACTUAL_HEADERS_VERIFIED'
  except Exception as ex:
   result['status']='FILES_VERIFIED_HEADER_CHECK_FAILED';result['header_error_type']=type(ex).__name__;result['header_error']=str(ex);ok=False
 atomic(base/'complete.json',result);print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'files':len(outputs),'completed_files':sum(x['status']!='PARTIAL_OR_FAILED' for x in outputs)},sort_keys=True),flush=True)
 return 0 if ok else 2
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as ex:
  print(json.dumps({'status':'ASSET_CHECK_FAILED','error_type':type(ex).__name__}),flush=True);sys.exit(3)
