"""Recover metadata CLAIM only, then the still-unsent first GPU start request.

Original operations never reached start_process. A readable exact claim and an
empty process/driver inventory are required. No model-source/plan bytes change.
"""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();D=a.directory
 package=json.loads((D/'package.json').read_text());plan=json.loads(Path(package['plan']['path']).read_text());intent=json.loads((D/'start-intent.json').read_text())
 assert not (D/'started.json').exists() and not (D/'first-start-after-claim-intent.json').exists()
 # There is deliberately no repeat of an unknown model-start operation.
 original_calls=[json.loads(p.read_text()) for p in (D/'transport').glob('*.intent.json')]
 assert not any(c.get('tool')=='start_process' for c in original_calls)
 observations=sorted(D.glob('observation-*.json'),key=lambda p:p.stat().st_mtime);assert observations
 observed=observations[-1];value=json.loads(observed.read_text());assert time.time()-observed.stat().st_mtime<90
 assert not value.get('matching_pids') and not any(k in value for k in ['driver-started.json','child-started.json','wrapper-complete.json','output/started.json','output/complete.json','output/failure.json'])
 def save(n,x):
  with (D/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 c=Client();c.initialize()
 def call(name,args,label):
  r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=package['mount'],**args)));save(label,r)
  assert not r.get('isError'),'Unknown preserved; inspect exact original state, never repeat unknown model start'
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 existing=value.get('start-intent.json')
 if existing:
  assert existing['value']==intent and existing['sha256']==sha((D/'start-intent.json').read_bytes())
  save('recovered-exact-claim.json',dict(observation=str(observed),claim_sha256=existing['sha256'],no_new_remote_claim=True))
 else:
  # Conditional publication is byte guarded against a delayed original CLAIM.
  code='''from pathlib import Path
import json,hashlib,os
r=Path(P['root']);assert not any((r/n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','started.json'])
for name,ref in P['files'].items():
 b=(r/name).read_bytes();assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
p=r/'start-intent.json';raw=P['claim'].encode()
if p.exists():assert p.read_bytes()==raw
else:
 try:
  with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 except FileExistsError:assert p.read_bytes()==raw
fd=os.open(str(r),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(claim_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),start_not_yet_called=True)))
'''
  args=dict(root=package['remote'],files=package['members'],claim=(D/'start-intent.json').read_text())
  save('recovery-claim-intent.json',dict(at_unix=time.time(),observation=str(observed),claim_sha256=sha((D/'start-intent.json').read_bytes())))
  x=call('run_command',dict(command='python3 -c '+shlex.quote('P='+repr(args)+'\n'+code),timeout_seconds=30),'recovery-claim-tool.json');assert x['exit_code']==0;x=json.loads(x['stdout']);assert x['claim_sha256']==sha((D/'start-intent.json').read_bytes());save('recovered-exact-claim.json',x)
 save('first-start-after-claim-intent.json',dict(at_unix=time.time(),source_sha256=sha(Path(__file__).read_bytes()),original_start_process_never_called=True,plan_sha256=package['plan']['sha256'],command=intent['command'],cwd=intent['cwd'],environment=intent['environment'],no_retry_unknown=True))
 x=call('start_process',dict(command=intent['command'],cwd=intent['cwd'],env=intent['environment']),'first-start-after-claim-tool.json');save('started.json',x)
 print(json.dumps({k:x[k] for k in ['id','pid','started_at','running']}))
if __name__=='__main__':main()
