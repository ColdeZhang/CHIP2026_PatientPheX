"""Unique58 hidden-CUDA CPU contract; read-only verified collection, no GPU/model."""
from pathlib import Path
import json,sys,hashlib,base64,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REL='research/deepseek-r1-0528-qwen3-8b-readiness/f0-preparation-v1/cpu';ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';PY='/home/zhangyuheng/projects/ScratchVL/.venv/bin/python';ENV={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':ROOT+'case-retrieval/qwen35-preparation/a30-fullcontext/runtime:'+ROOT+'case-retrieval/round2/runtime'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (F/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def call(c,name,args):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':M,**args}});assert not r.get('isError'),r;return r

def read_ref(c,record,target):
 data=bytearray()
 for start in range(0,record['bytes'],65536):
  count=min(65536,record['bytes']-start)
  code="from pathlib import Path;import base64,json,hashlib;p=Path("+repr(record['path'])+");assert p.stat().st_size=="+str(record['bytes'])+";f=p.open('rb');f.seek("+str(start)+");b=f.read("+str(count)+");assert len(b)=="+str(count)+";print(json.dumps({'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}))"
  v=un(call(c,'run_command',dict(command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)));assert v['exit_code']==0;x=json.loads(v['stdout']);b=base64.b64decode(x['data']);assert hashlib.sha256(b).hexdigest()==x['sha256'];data.extend(b)
 assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'];target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())

def main():
 save('deployment-intent.json',{'at':time.time(),'source_sha256':sha(F/'prepare_cpu.py'),'plan_sha256':sha(F/'plan.json'),'only_once_no_reexecute_on_UNKNOWN':True});c=Client();c.initialize();files=['prepare_cpu.py','plan.json','prompts.jsonl.gz']
 for n in files:
  b=(F/n).read_bytes();binary=n.endswith('.gz');r=call(c,'write_file',dict(path=REL+'/'+n,content=base64.b64encode(b).decode() if binary else b.decode(),encoding='base64' if binary else 'utf8',create_parents=True));save(n+'.upload.json',r)
 checks={n:sha(F/n) for n in files};code="from pathlib import Path;import hashlib,json,os;p=Path("+repr(ROOT+REL)+");e="+repr(checks)+";assert os.getuid()==p.stat().st_uid==1006;os.chmod(p,0o700);assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in e.items());compile((p/'prepare_cpu.py').read_text(),'prepare_cpu.py','exec');d=json.loads((p/'plan.json').read_text());assert all(Path(r['path']).stat().st_size==r['bytes'] and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in d['refs'].values());assert not (p/'actual').exists();print(json.dumps({'status':'ACTUAL_F0_CPU_SOURCE_PLAN_REFS_READY_NO_RUN','sources':e}))"
 r=call(c,'run_command',dict(command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30));save('actual-prestart-check.json',r);assert un(r)['exit_code']==0
 cmd='timeout --signal=TERM --kill-after=1s 59s '+shlex.quote(PY)+' -u prepare_cpu.py --plan plan.json'
 save('cpu-execution-intent.json',{'at':time.time(),'command':cmd,'env':ENV,'plan_sha256':sha(F/'plan.json'),'on_unknown':'Read existing actual/result; never rerun tokenizer blindly'})
 r=call(c,'run_command',dict(command=cmd,cwd=REL,env=ENV,timeout_seconds=65));save('actual-cpu-tool.json',r);v=un(r);assert v['exit_code']==0,v
 code="from pathlib import Path;import json,hashlib;p=Path("+repr(ROOT+REL+'/actual/result.json')+");print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
 v=un(call(c,'run_command',dict(command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)));assert v['exit_code']==0;rr=json.loads(v['stdout']);read_ref(c,rr,F/'actual-result.json');result=json.loads((F/'actual-result.json').read_text());assert result['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL58_F0_INPUT_CPU_CONTRACT'
 for n,record in result['outputs'].items():read_ref(c,record,F/'recovered'/n)
 save('collection-manifest.json',{'result':rr,'outputs':result['outputs'],'all_local_full_SHA':True,'no_weights_GPU_gold':True});print(json.dumps({k:result[k] for k in ['status','program_seconds','native_lengths','selected_indices','counts']}))
if __name__=='__main__':main()
