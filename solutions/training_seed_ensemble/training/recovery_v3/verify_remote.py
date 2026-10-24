"""Second bounded read-only CPU verifier, distinct receipts; prior UNKNOWN preserved."""
import os,sys,json,hashlib,time,shlex
from pathlib import Path
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1/formal-recovery-v3')
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=json.loads((S/'formal-plan.json').read_text());assert (D/'deployment.json').exists();assert not (D/'cpu-verify-attempt2-intent.json').exists()
 cfg={'python':p['python'],'env':{**p['environment'],'CUDA_VISIBLE_DEVICES':''},'root':p['output_dir'].rsplit('/',1)[0]}
 code='CFG='+repr(cfg)+'\n'+r'''
import os,json,time,subprocess,signal
from pathlib import Path
root=Path(CFG['root']);os.chdir(root)
def save(name,obj):
 with Path(name).open('x') as f:json.dump(obj,f,indent=2);f.write(chr(10));f.flush();os.fsync(f.fileno())
assert not Path('cpu-attempt2.child-started.json').exists()
start=time.monotonic();timed=False
with Path('cpu-attempt2.stdout').open('xb') as out,Path('cpu-attempt2.stderr').open('xb') as err:
 child=subprocess.Popen([CFG['python'],'-u','verify_cpu.py'],env={**os.environ,**CFG['env']},stdout=out,stderr=err,start_new_session=True)
 save('cpu-attempt2.child-started.json',{'pid':child.pid,'start_unix':time.time(),'budget':30,'grace':5,'no_model':True})
 try:rc=child.wait(timeout=max(.001,start+30-time.monotonic()))
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(child.pid,signal.SIGTERM)
  try:rc=child.wait(timeout=max(.001,start+35-time.monotonic()))
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
 for f in [out,err]:f.flush();os.fsync(f.fileno())
result={'returncode':rc,'timed_out':timed,'seconds':time.monotonic()-start,'stdout':Path('cpu-attempt2.stdout').read_text(),'stderr':Path('cpu-attempt2.stderr').read_text()}
save('cpu-attempt2.terminal.json',result);print(json.dumps(result))
'''
 args={'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','cwd':'training-seed-ensemble-v1/formal-recovery-v3','command':'python3 -c '+shlex.quote(code),'timeout_seconds':40}
 save(D/'cpu-verify-attempt2-intent.json',{'at_unix':time.time(),'arguments':args,'retry':False,'prior_UNKNOWN_preserved':True})
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':args});save(D/'cpu-verify-attempt2-tool.json',r)
 assert not r.get('isError');a=r.get('structuredContent')
 if a is None:a=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 assert a['exit_code']==0,a
 terminal=json.loads(a['stdout']);save(D/'cpu-attempt2.terminal.json',terminal);assert terminal['returncode']==0 and not terminal['timed_out'],terminal
 x=json.loads(terminal['stdout']);assert x['status']=='PASS_ACTUAL_TRAINER_VERIFY_INPUTS_NO_MODEL';assert x['plan_sha256']==hashlib.sha256((S/'formal-plan.json').read_bytes()).hexdigest()
 save(D/'cpu-verify-attempt2.json',x);print(json.dumps(x))
if __name__=='__main__':main()
