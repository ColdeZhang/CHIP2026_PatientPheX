from pathlib import Path
import json,sys,base64,shlex,time,os,hashlib
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
C=Path(__file__).parent;payload=json.loads((C/'payload.json').read_text());M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel=payload['remote_namespace'];root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def save(n,o):
 with (C/n).open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':dict(mount_id=M,**args)})
 if r.get('isError'):save('error-'+str(time.time_ns())+'.json',r);raise RuntimeError(name+' error; reconcile actual state before retry')
 return r
assert not (C/'deployment-intent.json').exists();save('deployment-intent.json',dict(at=time.time(),payload=payload,remote=rel));c=Client();c.initialize()
r=call('write_file',dict(path=rel+'/payload.tar.gz',content=base64.b64encode((C/'payload.tar.gz').read_bytes()).decode(),encoding='base64',create_parents=True));save('payload-write.json',r)
code='''from pathlib import Path
import json,hashlib,os,tarfile
p=Path(PATH);expected=EXPECTED
assert os.getuid()==1006 and p.stat().st_uid==1006;os.chmod(p,0o700)
b=(p/'payload.tar.gz').read_bytes();assert len(b)==expected['bytes'] and hashlib.sha256(b).hexdigest()==expected['sha256']
with tarfile.open(p/'payload.tar.gz','r:gz') as t:
 ms=t.getmembers();assert len(ms)==len(expected['members']) and set(x.name for x in ms)==set(expected['members'])
 for x in ms:
  assert x.isfile() and not x.issym() and Path(x.name).name==x.name
  data=t.extractfile(x).read();e=expected['members'][x.name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];target=p/x.name
  if target.exists():assert target.read_bytes()==data
  else:
   with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 for n in ['cpu_prepare.py','strict_final.py']:compile((p/n).read_text(),n,'exec')
print(json.dumps({'status':'EXACT_PAYLOAD_AND_PYTHON310_SOURCE_READY','uid':os.getuid(),'members':expected['members']}))
'''.replace('PATH',repr(root+rel)).replace('EXPECTED',repr(payload))
r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=30));save('deployment-result.json',r);assert un(r)['exit_code']==0
args=dict(command='timeout --signal=TERM --kill-after=30s 120s /home/zhangyuheng/projects/ScratchVL/.venv/bin/python -u cpu_prepare.py --plan cpu-plan.json > cpu.stdout.log 2> cpu.stderr.log',cwd=rel,env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':root+'case-retrieval/qwen35-preparation/a30-fullcontext/runtime:'+root+'case-retrieval/round2/runtime','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
save('start-intent.json',dict(at=time.time(),arguments=args,no_model_GPU_gold=True))
try:r=call('start_process',args);save('cpu-job.json',r);print(json.dumps(un(r)))
except BaseException as e:save('start-unknown.json',dict(at=time.time(),error=repr(e),action='Do not repeat start; read original actual/result.json and managed PID'));raise
