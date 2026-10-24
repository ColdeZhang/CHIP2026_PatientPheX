"""Attempt2 transport, CUDA-hidden CPU gate, unique remaining-budget GPU start and collection."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,sys
R=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/gpu-cost-probe-attempt2';PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';PYTHON=PROJECT+'/public-phenotagger-reuse/.venv/bin/python';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f';plan=json.loads((R/'plan.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,o):(R/n).write_text(json.dumps(o,indent=2)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','cpu','start','collect-cpu','collect']);a=parser.parse_args()
expected={n:sha((R/n).read_bytes()) for n in ['probe.py','launch.py','plan.json','check_policy_cpu.py','cpu_launch.py','probe-inputs.json','cpu-reference.json']};assert all(expected[n]==d for n,d in plan['deployed_sources'].items())
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':MOUNT,**args}})
 if r.get('isError'):raise RuntimeError(str(r.get('content'))[:600])
 return r['structuredContent']
def fetch(n):
 r=call('read_file',{'path':REMOTE+'/'+n,'offset':0,'limit':65536});b=base64.b64decode(r['content']) if r.get('encoding')=='base64' else r['content'].encode();assert len(b)<65536;(R/n).write_bytes(b);return b
if a.action=='deploy':
 assert not(R/'deployment-state.json').exists();s={'status':'running','plan_sha256':expected['plan.json'],'verified_new_files':[]};save('deployment-state.json',s)
 try:
  for n in ['probe.py','launch.py','plan.json','check_policy_cpu.py','cpu_launch.py']:
   b=(R/n).read_bytes();assert len(b)<=65536;s['active_request']={'action':'write_file','name':n,'bytes':len(b),'sha256':expected[n]};save('deployment-state.json',s)
   x=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert x['sha256']==expected[n] and x['size']==len(b);s['verified_new_files'].append(n);save('deployment-state.json',s)
  code="from pathlib import Path;import hashlib,json,os\nexpected="+repr(expected)+"\nfor n in ['probe-inputs.json','cpu-reference.json']:\n source=Path('../gpu-cost-probe')/n;assert hashlib.sha256(source.read_bytes()).hexdigest()==expected[n]\n target=Path(n)\n if not target.exists():os.link(source,target)\nactual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected\nfor n in ['probe.py','launch.py','check_policy_cpu.py','cpu_launch.py']:compile(Path(n).read_text(),n,'exec')\nreceipt={'status':'complete_sources_and_samehost_inputs','source_sha256':actual,'no_NN_or_GPU':True};Path('deployment-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))"
  request={'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30};save('deployment-link-intent.json',request);x=call('run_command',request);save('deployment-tool.json',x);assert x['exit_code']==0
  s.update(status='complete',actual_remote=json.loads(x['stdout']));save('deployment-state.json',s);print(json.dumps({'deployment':'complete','no_NN_GPU':True}));raise SystemExit()
 except Exception as e:s.update(status='failed_or_unknown_outcome_preserved',error=str(e));save('deployment-state.json',s);raise
if a.action=='cpu':
 assert read(R/'deployment-state.json')['status']=='complete' and not(R/'CPU-start-intent.json').exists()
 env={**plan['GPU_environment'],'CUDA_VISIBLE_DEVICES':''};request={'command':shlex.quote(PYTHON)+' cpu_launch.py','cwd':REMOTE,'env':env,'timeout_seconds':45};save('CPU-start-intent.json',{'request':request,'plan_sha256':expected['plan.json']});x=call('run_command',request);save('CPU-tool-receipt.json',x);print(json.dumps(x));raise SystemExit()
if a.action=='collect-cpu':
 assert (R/'CPU-start-intent.json').exists()
 for n in ['CPU-policy-job.json','CPU-policy-result.json','CPU-policy.stderr.log','CPU-policy.stdout.log']:fetch(n)
 print(json.dumps(read(R/'CPU-policy-job.json')));raise SystemExit()
if a.action=='start':
 assert read(R/'deployment-state.json')['status']=='complete' and read(R/'CPU-policy-job.json')['returncode']==0 and read(R/'CPU-policy-result.json')['status']=='pass_actual_CPU_policy_and_compat_layer_dtypes_no_forward'
 assert (R/'sequence-release-authorization.json').exists(),'Need actual sequence release before remaining-budget retry'
 assert not(R/'start-intent.json').exists() and not(R/'managed-job.json').exists()
 request={'command':shlex.quote(PYTHON)+' launch.py','cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}};save('start-intent.json',{'request':request,'plan_sha256':expected['plan.json'],'source_sha256':sha(Path(__file__).read_bytes())});job=call('start_process',request);save('managed-job.json',job);print(json.dumps(job));raise SystemExit()
job=read(R/'managed-job.json');status=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096});save('process-status.json',status)
if status['running']:print(json.dumps({'running':True}));raise SystemExit()
state=json.loads(fetch('run-state.json'))
for n in ['result.json','probe.stdout.log','probe.stderr.log']:
 try:fetch(n)
 except Exception as e:save(n+'.collection-error.json',{'error':str(e)})
print(json.dumps(state))
