"""Unique formal start; unknown outcome retains intent, never triggers blind retry."""
from pathlib import Path
import json,sys,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REL='research/deepseek-r1-0528-qwen3-8b-readiness/full80-v1';ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (F/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 assert not (F/'start-intent.json').exists(),'Reconcile original intent/job, never duplicate launch'
 plan=json.loads((F/'plan.json').read_text());cpu=json.loads((F/'actual-input-contract.json').read_text());assert cpu['status']=='ACTUAL_FORMAL_INPUT_CACHE_SCHEMA_PATH_CPU_PASS' and cpu['plan_sha256']==sha(F/'plan.json')
 c=Client();c.initialize();code="from pathlib import Path;import json,hashlib,os;p=Path("+repr(ROOT+REL)+");s=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();assert s(p/'plan.json')=="+repr(sha(F/'plan.json'))+";plan=json.loads((p/'plan.json').read_text());assert all(s(p/n)==h for n,h in plan['sources'].items());cpu=json.loads((p/'actual-input-contract.json').read_text());assert cpu['plan_sha256']==s(p/'plan.json') and cpu['status']=='ACTUAL_FORMAL_INPUT_CACHE_SCHEMA_PATH_CPU_PASS';assert not (p/'wrapper-started.json').exists() and not (p/'terminal').exists();print(json.dumps({'status':'ACTUAL_FORMAL_SOURCE_PLAN_CPU_READY_NO_START','plan_sha256':s(p/'plan.json'),'uid':os.getuid()}))"
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)});save('actual-prestart-check.json',r);assert not r.get('isError') and un(r)['exit_code']==0
 args=dict(mount_id=M,cwd=REL,command=shlex.quote(plan['python'])+' -u launch_full80.py --plan plan.json',env={**plan['GPU_environment'],'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 save('start-intent.json',{'at':time.time(),'arguments':args,'plan_sha256':sha(F/'plan.json'),'unique_formal78fresh':True,'on_unknown':'Read actual original wrapper/managed job, do not restart'})
 try:
  r=c.request('tools/call',{'name':'start_process','arguments':args});save('job.json',r);assert not r.get('isError');print(json.dumps(un(r)))
 except Exception as e:save('start-unknown-outcome.json',{'error':type(e).__name__,'remote_may_be_running':True,'at':time.time()});raise
if __name__=='__main__':main()
