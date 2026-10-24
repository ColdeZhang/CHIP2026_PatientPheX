"""Single actual target CPU verification; no model and no automatic retry."""
import os,sys,json,hashlib,time,shlex
from pathlib import Path
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1/formal-recovery-v2')
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=json.loads((S/'formal-plan.json').read_text());assert (D/'deployment.json').exists();assert not (D/'cpu-verify-intent.json').exists()
 env={**p['environment'],'CUDA_VISIBLE_DEVICES':''};cmd=shlex.join(['env']+[k+'='+v for k,v in env.items()]+['timeout','--signal=TERM','--kill-after=5','30',p['python'],'-u','verify_cpu.py'])
 args={'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','cwd':'training-seed-ensemble-v1/formal-recovery-v2','command':cmd,'timeout_seconds':40}
 save(D/'cpu-verify-intent.json',{'at_unix':time.time(),'arguments':args,'retry':False})
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':args});save(D/'cpu-verify-tool.json',r)
 assert not r.get('isError');a=r.get('structuredContent')
 if a is None:a=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 assert a['exit_code']==0,a
 x=json.loads(a['stdout']);assert x['status']=='PASS_ACTUAL_TRAINER_VERIFY_INPUTS_NO_MODEL';assert x['plan_sha256']==hashlib.sha256((S/'formal-plan.json').read_bytes()).hexdigest()
 save(D/'cpu-verify.json',x);print(json.dumps(x))
if __name__=='__main__':main()
