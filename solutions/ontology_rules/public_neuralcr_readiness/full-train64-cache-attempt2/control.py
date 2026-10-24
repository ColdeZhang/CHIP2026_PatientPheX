"""Initialize-repair deployment, actual same-entry CPU gate and unique remaining-budget run."""
from pathlib import Path
import argparse,base64,hashlib,json,shlex,sys
R=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/full-train64-cache-attempt2';PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';PYTHON=PROJECT+'/public-phenotagger-reuse/.venv/bin/python';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f';plan=json.loads((R/'plan.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','cpu','collect-cpu','start','collect']);action=p.parse_args().action
expected={n:sha((R/n).read_bytes()) for n in [*plan['deployed_sources'],'plan.json','bound-plan.json','preparation.json','preparation-job.json']};assert all(expected[n]==d for n,d in plan['deployed_sources'].items())
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 x=c.request('tools/call',{'name':n,'arguments':{'mount_id':MOUNT,**args}})
 if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
 return x['structuredContent']
def fetch(n):
 x=call('read_file',{'path':REMOTE+'/'+n,'offset':0,'limit':65536});b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)<65536;(R/n).write_bytes(b);return b
if action=='deploy':
 assert not(R/'deployment-state.json').exists();s={'status':'running','plan_sha256':expected['plan.json'],'verified_files':[]};save('deployment-state.json',s)
 try:
  for n in [*plan['deployed_sources'],'plan.json','bound-plan.json']:
   b=(R/n).read_bytes();assert len(b)<=65536;s['active_request']={'name':n,'bytes':len(b),'sha256':expected[n]};save('deployment-state.json',s);x=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert x['sha256']==expected[n] and x['size']==len(b);s['verified_files'].append(n);save('deployment-state.json',s)
  code="from pathlib import Path;import hashlib,json,os\nexpected="+repr(expected)+"\nfor n in ['preparation.json','preparation-job.json']:\n src=Path('../full-train64-cache')/n;assert hashlib.sha256(src.read_bytes()).hexdigest()==expected[n];dst=Path(n)\n if not dst.exists():os.link(src,dst)\nactual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected\nfor n in expected:\n if n.endswith('.py'):compile(Path(n).read_text(),n,'exec')\nx={'status':'actual_sources_and_same_preparation_hardlinks_verified','sha256':actual};Path('deployment-receipt.json').write_text(json.dumps(x));print(json.dumps(x))"
  req={'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'},'timeout_seconds':30};save('deployment-check-intent.json',req);x=call('run_command',req);save('deployment-tool.json',x);assert x['exit_code']==0;s.update(status='complete',actual=json.loads(x['stdout']));save('deployment-state.json',s);print('deployment complete')
 except Exception as e:s.update(status='failed_or_unknown_preserved',error=str(e));save('deployment-state.json',s);raise
 raise SystemExit()
if action=='cpu':
 assert read(R/'deployment-state.json')['status']=='complete' and not(R/'CPU-start-intent.json').exists()
 code="from pathlib import Path;import subprocess,sys,os,json,time,datetime\nassert os.environ['CUDA_VISIBLE_DEVICES']=='' and not Path('CPU-job.json').exists()\ncmd=['timeout','--signal=TERM','--kill-after=5s','30s',sys.executable,'run_cache.py','--check-initialization-only'];j={'status':'running','command':cmd,'cap_seconds':30,'TERM_grace_seconds':5,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};Path('CPU-job.json').write_text(json.dumps(j));t=time.monotonic()\nwith Path('CPU.stdout.log').open('xb') as out,Path('CPU.stderr.log').open('xb') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=37)\nj.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,elapsed_seconds=time.monotonic()-t,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());Path('CPU-job.json').write_text(json.dumps(j,indent=2));print(json.dumps(j));raise SystemExit(p.returncode)"
  # Deliberate same-entry execution, never a duplicate schema-test implementation.
 req={'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'},'timeout_seconds':40};save('CPU-start-intent.json',req);x=call('run_command',req);save('CPU-tool.json',x);print(json.dumps(x));raise SystemExit()
if action=='collect-cpu':
 for n in ['CPU-job.json','initialization-result.json','CPU.stderr.log','CPU.stdout.log']:fetch(n)
 print(json.dumps(read(R/'CPU-job.json')));raise SystemExit()
if action=='start':
 assert read(R/'deployment-state.json')['status']=='complete' and read(R/'CPU-job.json')['returncode']==0 and read(R/'initialization-result.json')['status']=='pass_actual_same_entry_initialization_no_numpy_model_or_NN' and not(R/'start-intent.json').exists()
 req={'command':shlex.quote(PYTHON)+' launch.py','cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}};save('start-intent.json',{'request':req,'plan_sha256':expected['plan.json'],'bound_sha256':expected['bound-plan.json'],'source_sha256':sha(Path(__file__).read_bytes())});x=call('start_process',req);save('managed-job.json',x);print(json.dumps(x));raise SystemExit()
job=read(R/'managed-job.json');x=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096});save('process-status.json',x)
if x['running']:
 try:
  r=json.loads(fetch('cache-state.json'));print(json.dumps({k:r.get(k) for k in ['status','completed_batches','completed_rows','elapsed_seconds']}))
 except Exception as e:print(json.dumps({'running':True,'state_read_error':str(e)}))
 raise SystemExit()
for n in ['run-state.json','cache-result.json','cache.stderr.log','cache.stdout.log']:
 try:fetch(n)
 except Exception as e:save(n+'.collection-error.json',{'error':str(e)})
print(json.dumps(read(R/'run-state.json')))
