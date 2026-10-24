"""Deploy/start only the supplemental CPU HTTP guardian; never touch NN/ACKs."""
import argparse,base64,hashlib,json,os,shlex,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/patientphex-tools')
from mcp_client import Client
S=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-full64-v1')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL='grounded-generation-full64-v1/export-guardian-v1';REMOTE=P+'/'+REL
MOUNT='69b991ee-183f-4add-af1a-579af92479c9'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 d=p.parent
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['deploy','start','observe']);a=ap.parse_args()
 d=R/'export-guardian-v1';d.mkdir(exist_ok=True);formal=R/'formal-v1';plan=json.loads((formal/'controller-plan.json').read_text());campaign=json.loads((formal/'campaign.json').read_text())
 assert campaign['plan_sha256']==ref(formal/'controller-plan.json')['sha256'] and campaign['cap_seconds']==6000
 assert time.time()<campaign['absolute_deadline_unix']+30 and plan['mount_id']==MOUNT
 client=Client();client.initialize()
 def call(name,args,tag):
  p=d/(tag+'-'+str(time.time_ns()));save(p.with_suffix('.intent.json'),dict(tool=name,arguments=args))
  result=client.request('tools/call',dict(name=name,arguments=dict(mount_id=MOUNT,**args)))
  raw=json.dumps(result)
  for key in ['MCP_TOKEN','MCP_URL']:
   if os.environ.get(key):raw=raw.replace(os.environ[key],'[REDACTED]')
  result=json.loads(raw);save(p.with_suffix('.tool.json'),result)
  assert not result.get('isError'),'Unknown original operation preserved; no automatic replay'
  return result.get('structuredContent') or json.loads(result['content'][0]['text'])
 def command(code,spec,tag):
  body='SPEC='+repr(spec)+'\n'+code;compile(body,'guardian-remote.py','exec')
  result=call('run_command',dict(command='python3 -c '+shlex.quote(body),timeout_seconds=30),tag)
  assert result['exit_code']==0;return json.loads(result['stdout'])
 if a.stage=='deploy':
  source=ref(S/'export_guardian.py');assert not (d/'start-intent.json').exists()
  code='''import base64,hashlib,json,os
from pathlib import Path
p=Path(SPEC['root']);p.mkdir(parents=True,exist_ok=True);assert os.getuid()==1045 and p.stat().st_uid==1045;os.chmod(p,0o700)
assert not (p/'start-intent.json').exists() and not (p/'actual').exists()
b=base64.b64decode(SPEC['data']);assert hashlib.sha256(b).hexdigest()==SPEC['source']['sha256'];compile(b,'export_guardian.py','exec');q=p/'export_guardian.py'
if q.exists():assert q.read_bytes()==b
else:
 with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with q.open('rb') as f:os.fsync(f.fileno())
assert hashlib.sha256(q.read_bytes()).hexdigest()==SPEC['source']['sha256']
while True:
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if p.parent==p:break
 p=p.parent
print(json.dumps(dict(SHA_fsync_reopen=True,source=SPEC['source'],GPU_model_HTTP_export_calls=0)))
'''
  actual=command(code,dict(root=REMOTE,source=source,data=base64.b64encode(Path(source['path']).read_bytes()).decode()),'deploy')
  save(d/'deployment.json',dict(actual=actual,campaign=ref(formal/'campaign.json'),plan=ref(formal/'controller-plan.json'),source=source,transport=ref(__file__)))
 elif a.stage=='start':
  deployed=json.loads((d/'deployment.json').read_text());assert deployed['source']==ref(S/'export_guardian.py') and deployed['actual']['SHA_fsync_reopen']
  assert not (d/'start-intent.json').exists();remaining=int(campaign['absolute_deadline_unix']+30-time.time());assert remaining>0
  argv=['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(remaining)+'s',plan['python'],'-u',REMOTE+'/export_guardian.py',
   '--plan',plan['remote_absolute']+'/controller-plan.json','--campaign',plan['remote_absolute']+'/campaign.json','--output',REMOTE+'/actual']
  intent=dict(command=shlex.join(argv),source=deployed['source'],original_deadline_unix=campaign['absolute_deadline_unix'],no_NN_or_export_start=True)
  save(d/'start-intent.json',intent)
  code='''from pathlib import Path
import json,os
p=Path(SPEC['root'])/'start-intent.json';assert not p.exists()
with p.open('x') as f:json.dump(SPEC['intent'],f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(claimed=True,start_not_yet_called=True)))
'''
  save(d/'claim.json',command(code,dict(root=REMOTE,intent=intent),'claim'))
  actual=call('start_process',dict(command=intent['command'],cwd=REL,env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'}),'start')
  save(d/'job.json',actual);print(json.dumps(actual))
 else:
  code='''from pathlib import Path
import json
p=Path(SPEC['root'])/'actual';out={}
for n in ['started.json','complete.json','epoch-001-identity.json','epoch-001-closed.json','epoch-002-identity.json','epoch-002-closed.json']:
 if (p/n).exists():out[n]=json.loads((p/n).read_text())
print(json.dumps(out))
'''
  actual=command(code,dict(root=REMOTE),'observe');save(d/('observation-'+str(time.time_ns())+'.json'),actual);print(json.dumps(actual))
if __name__=='__main__':main()
