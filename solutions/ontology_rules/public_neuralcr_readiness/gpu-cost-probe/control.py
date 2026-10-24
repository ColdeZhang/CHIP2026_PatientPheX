"""Bounded small-file deployment, unique GPU probe start and exact result collection."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,sys
R=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/gpu-cost-probe';PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';PYTHON=PROJECT+'/public-phenotagger-reuse/.venv/bin/python';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,o):(R/n).write_text(json.dumps(o,indent=2)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','start','collect']);a=parser.parse_args();plan=read(R/'plan.json')
expected={n:sha((R/n).read_bytes()) for n in ['probe.py','launch.py','plan.json','probe-inputs.json','cpu-reference.json']}
assert all(expected[n]==digest for n,digest in plan['deployed_sources'].items())
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 result=c.request('tools/call',{'name':n,'arguments':{'mount_id':MOUNT,**args}})
 if result.get('isError'):raise RuntimeError(str(result.get('content'))[:600])
 return result['structuredContent']
def fetch(n):
 result=call('read_file',{'path':REMOTE+'/'+n,'offset':0,'limit':65536});b=base64.b64decode(result['content']) if result.get('encoding')=='base64' else result['content'].encode();assert len(b)<65536,'Must explicitly collect additional ranges, not silently truncate';(R/n).write_bytes(b);return b
if a.action=='deploy':
 assert not (R/'deployment-state.json').exists()
 state={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':expected['plan.json'],'source_sha256':sha(Path(__file__).read_bytes()),'verified_files':[]}
 save('deployment-state.json',state)
 try:
  for n,digest in expected.items():
   b=(R/n).read_bytes();assert len(b)<=65536;state['active_request']={'action':'write_file','file':n,'bytes':len(b),'sha256':digest};save('deployment-state.json',state)
   result=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert result['sha256']==digest and result['size']==len(b)
   state['verified_files'].append({'name':n,'bytes':len(b),'sha256':digest});save('deployment-state.json',state)
  code="from pathlib import Path;import hashlib,json\nexpected="+repr(expected)+"\nactual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected\nfor n in ['probe.py','launch.py']:compile(Path(n).read_text(),n,'exec')\nx=json.loads(Path('probe-inputs.json').read_text());assert len(x['batches'])==3 and all(len(b)==512 for b in x['batches'])\nprint(json.dumps({'source_sha256':actual,'python_version':__import__('sys').version,'batches':3,'queries':1536,'no_NN_or_GPU':True}))"
  result=call('run_command',{'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});save('remote-CPU-preparation-tool.json',result);assert result['exit_code']==0
  state.update(status='complete',actual_remote=json.loads(result['stdout']),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save('deployment-state.json',state);print(json.dumps({'deployment':'complete','no_GPU_or_NN':True}));raise SystemExit()
 except Exception as e:
  state.update(status='failed_or_unknown_outcome_preserved',error=str(e));save('deployment-state.json',state);raise
if a.action=='start':
 assert read(R/'deployment-state.json')['status']=='complete' and read(R/'deployment-state.json')['plan_sha256']==expected['plan.json']
 assert not (R/'start-intent.json').exists() and not(R/'managed-job.json').exists()
 request={'command':shlex.quote(PYTHON)+' launch.py','cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
 save('start-intent.json',{'request':request,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':expected['plan.json'],'source_sha256':sha(Path(__file__).read_bytes())});job=call('start_process',request);save('managed-job.json',job);print(json.dumps(job));raise SystemExit()
job=read(R/'managed-job.json');status=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096});save('process-status.json',status)
if status['running']:print(json.dumps({'running':True}));raise SystemExit()
state=json.loads(fetch('run-state.json'))
for n in ['result.json','probe.stdout.log','probe.stderr.log']:
 try:fetch(n)
 except Exception as e:save(n+'.collection-error.json',{'error':str(e)})
print(json.dumps(state))
