"""Fetch exact published assets; standard library only and no GPU/model execution."""
import argparse,datetime,hashlib,json,math,os,struct,tarfile,time,urllib.request
from pathlib import Path,PurePosixPath

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
def write(p,value):
 tmp=p.with_name(p.name+'.pending')
 with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def run(plan_path):
 plan=json.loads(plan_path.read_bytes());root=Path(plan['output']);root.mkdir(parents=True,exist_ok=True)
 assert root.is_absolute() and root.name=='prepare-v1'
 start=time.monotonic();results=[]
 for item in plan['assets']:
  relative=PurePosixPath(item['path']);assert not relative.is_absolute() and '..' not in relative.parts
  dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.stat().st_size==item['bytes'] and digest(dest)==item['sha256'];action='reused_exact'
  else:
   part=dest.with_name(dest.name+'.part');n=part.stat().st_size if part.exists() else 0
   headers={'User-Agent':'PatientPheX-schema-readiness'}
   if n:headers['Range']=f'bytes={n}-'
   with urllib.request.urlopen(urllib.request.Request(item['url'],headers=headers),timeout=40) as response:
    if n and response.status==206:assert response.headers['Content-Range'].startswith(f'bytes {n}-');mode='ab'
    else:
     if n:part.rename(part.with_name(part.name+'.preserved-'+str(time.time_ns())))
     mode='wb'
    with part.open(mode) as f:
     while True:
      data=response.read(1<<20)
      if not data:break
      f.write(data)
     f.flush();os.fsync(f.fileno())
   assert part.stat().st_size==item['bytes'] and digest(part)==item['sha256'],item['path']
   os.replace(part,dest);action='downloaded_verified'
  results.append(dict(**ref(dest),action=action,url=item['url']))
  write(root/'progress.json',dict(complete_assets=results,elapsed_seconds=time.monotonic()-start))
  print(json.dumps(dict(path=item['path'],status=action,bytes=item['bytes'])),flush=True)
 package=root/'source.tar.gz'
 with tarfile.open(package,'r:gz') as tf:
  for member in tf.getmembers():
   parts=PurePosixPath(member.name).parts
   if not member.isfile() or len(parts)<2:continue
   rel=PurePosixPath(*parts[1:]);assert not rel.is_absolute() and '..' not in rel.parts
   if rel.parts[0]!='gliner2' and str(rel) not in ['pyproject.toml','LICENSE']:continue
   dest=root/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data
   else:
    with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 weights=root/'model/model.safetensors'
 with weights.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert 0<n<16_000_000;header=json.loads(f.read(n))
 count=0;dtypes={};intervals=[]
 for name,value in header.items():
  if name=='__metadata__':continue
  size=math.prod(value['shape']);count+=size;dtypes[value['dtype']]=dtypes.get(value['dtype'],0)+size
  intervals.append(value['data_offsets'])
 intervals.sort();assert intervals[0][0]==0 and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
 assert 8+n+intervals[-1][1]==weights.stat().st_size and count==193581591 and count<10_000_000_000
 config=json.loads((root/'model/config.json').read_bytes());assert config['architecture']=='boundary' and config['boundary_head']['enable_relations'] is True
 write(root/'complete.json',dict(status='COMPLETE_EXACT_ASSETS_AND_SAFE_SOURCE_NO_MODEL_OR_GPU',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),assets=results,source=ref(Path(__file__)),plan=ref(plan_path),parameters=count,dtypes=dtypes,tensors=len(intervals),header_bytes=n,code_files=len(list((root/'code').rglob('*.py'))),elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
 try:run(args.plan)
 except BaseException as exc:
  root=Path(json.loads(args.plan.read_bytes())['output']);root.mkdir(parents=True,exist_ok=True)
  write(root/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),partial_assets_preserved=True,no_model_or_GPU=True))
  raise
