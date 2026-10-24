from pathlib import Path
import sys,json,io,zipfile,base64,hashlib,shlex,datetime
D=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';cwd='public-phenotagger-biobert';remote='B901-census'
def dump(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
c=Client();c.initialize()
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':M,**a}});assert not r.get('isError'),r;return r['structuredContent']
mode=sys.argv[1]
if mode=='deploy':
 assert not (D/'deployment.json').exists();files=['run_census.py','plan.json','input-contract.json','B.input.jsonl'];buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for n in files:z.writestr(remote+'/'+n,(D/n).read_bytes())
 b=buf.getvalue();archive_sha=hashlib.sha256(b).hexdigest();(D/'deployment.zip').write_bytes(b)
 for i,off in enumerate(range(0,len(b),196608)):call('write_file',dict(path=cwd+'/B901-census-upload/part-%03d'%i,content=base64.b64encode(b[off:off+196608]).decode(),encoding='base64',create_parents=True))
 code="from pathlib import Path;import io,zipfile,hashlib,json;parts=sorted(Path('B901-census-upload').glob('part-*'));b=b''.join(p.read_bytes() for p in parts);assert hashlib.sha256(b).hexdigest()=="+repr(archive_sha)+";assert not Path('B901-census').exists();z=zipfile.ZipFile(io.BytesIO(b));assert all(x.startswith('B901-census/') and '..' not in x for x in z.namelist());z.extractall('.');print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in z.namelist()}));[p.unlink() for p in parts]"
 r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=cwd,timeout_seconds=30));assert r['exit_code']==0,r;actual=json.loads(r['stdout'])
 for n in files:assert actual[remote+'/'+n]==sha(D/n)
 dump('deployment.json',dict(at=now(),mount_id=M,cwd=cwd,archive_sha256=archive_sha,files=actual));print('DEPLOYED',archive_sha)
elif mode=='start':
 assert (D/'deployment.json').exists();intent=D/'start-intent.json'
 with intent.open('x') as f:json.dump(dict(at=now(),unique=True,cap=600,term=5,no_GPU=True),f)
 command='timeout --signal=TERM --kill-after=5s 600s ../public-phenotagger-reuse/.venv/bin/python B901-census/run_census.py --prepare-only';env=dict(CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',FORCE_TF_AVAILABLE='1',USE_TF='1',USE_TORCH='0')
 job=call('start_process',dict(command=command,cwd=cwd,env=env));dump('job.json',dict(at=now(),command=command,env=env,result=job));print(json.dumps(job))
elif mode=='inspect':
 j=json.loads((D/'job.json').read_text())['result'];ident=j.get('process_id') or j.get('id');assert ident,j
 r=call('get_process',dict(process_id=ident));dump('latest-process.json',r);print(json.dumps(r))
elif mode=='collect':
 code="from pathlib import Path;import json,hashlib; p=Path('native-B901-census-cpu-preparation/preparation.json');b=p.read_bytes();print(json.dumps(dict(preparation=json.loads(b),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))))"
 r=call('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=cwd,timeout_seconds=20));assert r['exit_code']==0,r;v=json.loads(r['stdout']);dump('collection.json',v);print(json.dumps(v))
else:raise AssertionError(mode)
