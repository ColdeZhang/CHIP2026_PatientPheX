from pathlib import Path
import sys,json,hashlib,gzip,base64,shlex,time,datetime,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
D=Path(__file__).resolve().parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REMOTE='research/medgemma-relation-control/tokenizer-family-audit-v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
 with (D/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert not (D/'intent.json').exists()
prompts=Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/sft-meta1-fusion/meta.prompts.jsonl').read_bytes();assert sha(prompts)=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
files={'audit.py':(D/'audit.py').read_bytes(),'prompts.jsonl.gz':gzip.compress(prompts,mtime=0)}
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/round2/runtime','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
argv=['/usr/bin/timeout','--signal=KILL','60','/home/zhangyuheng/projects/ScratchVL/.venv/bin/python','-u',REMOTE+'/audit.py']
save('plan.json',dict(mount_id=M,namespace=REMOTE,files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in files.items()},argv=argv,env=env,budget_seconds=60,grace_seconds=0,source_sha256=sha(Path(__file__).read_bytes())))
c=Client();c.initialize()
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':M,**args}});assert not r.get('isError'),'MCP error; retain known IDs, no retry';return r.get('structuredContent') or json.loads(r['content'][0]['text'])
r=call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir(parents=True,mode=0o700)'),cwd='.',timeout_seconds=15));assert r['exit_code']==0
for n,b in files.items():
 assert len(b)<512*1024
 call('write_file',dict(path=REMOTE+'/'+n,content=base64.b64encode(b).decode(),encoding='base64',create_parents=False))
refs={REMOTE+'/'+n:sha(b) for n,b in files.items()};r=call('run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;import hashlib;refs='+repr(refs)+';assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==s for n,s in refs.items())'),cwd='.',timeout_seconds=15));assert r['exit_code']==0;save('deployed.json',r)
save('intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=argv,never_duplicate_unknown=True));job=call('start_process',dict(command=shlex.join(argv),cwd='.',env=env));save('job.json',job);print(json.dumps(job),flush=True)
cursors={'stdout':0,'stderr':0};started=time.time()
while True:
 running=False
 for stream in cursors:
  r=call('read_process_output',dict(process_id=job['id'],stream=stream,cursor=cursors[stream],limit=20000));cursors[stream]=r['next_cursor'];running|=r['running']
  if r['content']:
   with (D/(stream+'.log')).open('a') as f:f.write(r['content'])
   print(r['content'],flush=True)
 if not running:break
 assert time.time()-started<150;time.sleep(3)
save('process-terminal.json',dict(job=job,running=False,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
r=call('read_file',dict(path=REMOTE+'/complete.json',offset=0,limit=512*1024));data=base64.b64decode(r['content']) if r.get('encoding')=='base64' else r['content'].encode();report=json.loads(data);assert report['source_sha256']==sha(files['audit.py']) and report['seconds']<=60 and not report['cuda_initialized'];(D/'complete.json').write_bytes(data)
print('FAMILY_AUDIT_COMPLETE '+sha(data),flush=True)
