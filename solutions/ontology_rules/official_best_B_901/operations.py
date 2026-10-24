"""MCP controller: fixed public B deployment, one start, read-only monitoring, small result recovery."""
from pathlib import Path
import sys,json,hashlib,datetime,zipfile,io,base64,shlex,shutil
D=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';REMOTE='official-B901-three-public';c=Client();c.initialize()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def call(name,a):
 r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=M,**a)));assert not r.get('isError'),r;return r['structuredContent']
def command(code,timeout=30):
 r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=timeout));assert r['exit_code']==0,r;return json.loads(r['stdout'])
mode=sys.argv[1]
if mode=='deploy':
 assert not (D/'deployment.json').exists();plan=json.loads((D/'batch-plan.json').read_text());files={REMOTE+'/'+n:D/n for n in ['batch_driver.py','batch-plan.json']}
 for m in plan['models']:
  for n in ['run_B.py','make_provenance.py','plan.json','input-contract.json']:files[m['remote_root']+'/B901-cache/'+n]=D/'models'/m['model']/n
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.writestr(n,p.read_bytes())
 b=buf.getvalue();(D/'deployment.zip').write_bytes(b);h=hashlib.sha256(b).hexdigest()
 with (D/'deploy-intent.json').open('x') as f:json.dump(dict(at=now(),archive_sha256=h),f)
 for i,off in enumerate(range(0,len(b),196608)):call('write_file',dict(path=REMOTE+'-upload/part-%03d'%i,content=base64.b64encode(b[off:off+196608]).decode(),encoding='base64',create_parents=True))
 roots=[m['remote_root'] for m in plan['models']]
 code="""from pathlib import Path
import json,io,zipfile,hashlib,shutil
roots=ROOTS
assert not Path('official-B901-three-public').exists()
for root in roots:assert not (Path(root)/'B901-cache').exists()
parts=sorted(Path('official-B901-three-public-upload').glob('part-*'));b=b''.join(p.read_bytes() for p in parts);assert hashlib.sha256(b).hexdigest()==HASH
z=zipfile.ZipFile(io.BytesIO(b));assert all('..' not in n and not n.startswith('/') for n in z.namelist());z.extractall('.')
source=Path('public-phenotagger-biobert/B901-census/B.input.jsonl');assert hashlib.sha256(source.read_bytes()).hexdigest()==INPUTSHA
for root in roots:shutil.copyfile(source,Path(root)/'B901-cache/B.input.jsonl')
result={n:dict(sha256=hashlib.sha256(Path(n).read_bytes()).hexdigest(),bytes=Path(n).stat().st_size) for n in z.namelist()}
for root in roots:
 p=Path(root)/'B901-cache/B.input.jsonl';result[str(p)]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
print(json.dumps(result));[p.unlink() for p in parts]
""".replace('ROOTS',repr(roots)).replace('HASH',repr(h)).replace('INPUTSHA',repr(plan['input_sha256']))
 actual=command(code)
 for n,p in files.items():assert actual[n]['sha256']==sha(p)
 dump('deployment.json',dict(at=now(),mount_id=M,archive_sha256=h,files=actual));print(json.dumps(dict(status='DEPLOYED',archive_bytes=len(b),archive_sha256=h)))
elif mode=='start':
 assert (D/'deployment.json').exists()
 with (D/'start-intent.json').open('x') as f:json.dump(dict(at=now(),batch_plan_sha256=sha(D/'batch-plan.json'),exclusive=True),f)
 cmd='../public-phenotagger-reuse/.venv/bin/python batch_driver.py';job=call('start_process',dict(command=cmd,cwd=REMOTE,env=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')));dump('job.json',dict(at=now(),command=cmd,result=job,batch_plan_sha256=sha(D/'batch-plan.json')));print(json.dumps(job))
elif mode=='inspect':
 j=json.loads((D/'job.json').read_text())['result'];r={s:call('read_process_output',dict(process_id=j['id'],stream=s,cursor=0,limit=32768)) for s in ['stdout','stderr']};state=command("from pathlib import Path;import json;p=Path('official-B901-three-public/run-state.json');print(p.read_text() if p.exists() else '{}')");dump('latest-managed.json',r);dump('latest-state.json',state);print(json.dumps(dict(managed={k:{x:v[x] for x in ['running','next_cursor'] if x in v} for k,v in r.items()},state=state)))
elif mode=='collect':
 model=sys.argv[2];mp=D/'models'/model;plan=json.loads((mp/'plan.json').read_text());root=plan['remote_root'];out=D/'results'/model;out.mkdir(parents=True,exist_ok=True);assert not (out/'collection.json').exists()
 code="""from pathlib import Path
import json,hashlib,zipfile,os
root=Path(ROOT);F=root/'native-B901-final-cache';N=root/'native-B901-cache';D=Path('official-B901-three-public');paths=[]
for folder in [N,F]:
 m=json.loads((folder/'manifest.json').read_text())
 for n,ref in m['outputs'].items():
  p=folder/n;h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  assert h.hexdigest()==ref['sha256'] and p.stat().st_size==ref['bytes'],str(p)
for folder,names in [(F,['manifest.json','canonical-entities.jsonl','source-provenance.jsonl','per-document-reuse.json']),(N,['manifest.json','preparation.json','GPU-first-batch.json'])]:
 for n in names:
  p=folder/n
  if p.exists():paths.append((str(p.relative_to(root)),p))
for n in ['run_B.py','make_provenance.py','plan.json','input-contract.json','native.stdout','native.stderr','provenance.stdout','provenance.stderr']:
 p=root/'B901-cache'/n;paths.append((str(p.relative_to(root)),p))
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in paths}
zpath=D/(MODEL+'-terminal.zip')
if zpath.exists():
 with zipfile.ZipFile(zpath) as z:
  assert json.loads(z.read('members.json'))==manifest
  for n,ref in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==ref['sha256']
else:
 with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in paths:z.write(p,n)
  z.writestr('members.json',json.dumps(manifest,indent=2)+chr(10))
with zpath.open('rb') as f:os.fsync(f.fileno())
print(json.dumps(dict(remote_path=str(zpath),bytes=zpath.stat().st_size,sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),members=manifest,full_native_outputs_reopened_sha_verified=True,raw_full_predictions_remain_persistent_at=str(N))))
""".replace('ROOT',repr(root)).replace('MODEL',repr(model))
 receipt_path=D/('collection-'+model+'-remote.json')
 receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else command(code,120)
 if not receipt_path.exists():dump(receipt_path.name,receipt)
 parts=[]
 for off in range(0,receipt['bytes'],196608):
  v=command("from pathlib import Path;import json,base64;p=Path("+repr(receipt['remote_path'])+");f=p.open('rb');f.seek("+str(off)+");b=f.read(196608);print(json.dumps(dict(content=base64.b64encode(b).decode())))")
  parts.append(base64.b64decode(v['content']))
 b=b''.join(parts);assert len(b)==receipt['bytes'] and hashlib.sha256(b).hexdigest()==receipt['sha256'];(out/'terminal.zip').write_bytes(b)
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert all('..' not in n and not n.startswith('/') for n in z.namelist());z.extractall(out/'terminal')
 for n,ref in receipt['members'].items():assert sha(out/'terminal'/n)==ref['sha256'] and (out/'terminal'/n).stat().st_size==ref['bytes']
 (out/'collection.json').write_text(json.dumps(dict(collected_at=now(),**receipt),indent=2)+'\n');print(json.dumps(dict(model=model,local_directory=str(out),zip_sha256=receipt['sha256'],bytes=len(b),counts=json.loads((out/'terminal/native-B901-final-cache/manifest.json').read_text())['counts'])))
else:raise AssertionError(mode)
