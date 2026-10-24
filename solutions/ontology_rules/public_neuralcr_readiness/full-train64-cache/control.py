"""Small immutable-source deployment, unique prepare/cache start, bounded collection."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,sys
R=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/full-train64-cache';PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';PYTHON=PROJECT+'/public-phenotagger-reuse/.venv/bin/python';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f';plan=json.loads((R/'plan.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,o):(R/n).write_text(json.dumps(o,indent=2)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','prepare-start','prepare-collect','publish-bound','start','collect']);a=parser.parse_args()
expected={n:sha((R/n).read_bytes()) for n in [*plan['deployed_sources'],'plan.json']};assert all(expected[n]==v for n,v in plan['deployed_sources'].items())
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':MOUNT,**args}})
 if r.get('isError'):raise RuntimeError(str(r.get('content'))[:700])
 return r['structuredContent']
def fetch(n):
 x=call('read_file',{'path':REMOTE+'/'+n,'offset':0,'limit':65536});b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)<65536
 target=R/n
 if target.exists() and n in ['preparation.json','cache-result.json','preparation-job.json']:assert target.read_bytes()==b,'Immutable receipt changed'
 target.write_bytes(b);return b
if a.action=='deploy':
 assert not(R/'deployment-state.json').exists();s={'status':'running','plan_sha256':expected['plan.json'],'verified_files':[]};save('deployment-state.json',s)
 try:
  for n in expected:
   b=(R/n).read_bytes();assert len(b)<=65536;s['active_request']={'action':'write_file','name':n,'bytes':len(b),'sha256':expected[n]};save('deployment-state.json',s)
   x=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert x['sha256']==expected[n] and x['size']==len(b);s['verified_files'].append(n);save('deployment-state.json',s)
  code="from pathlib import Path;import hashlib,json\nexpected="+repr(expected)+"\nactual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected\nfor n in expected:\n if n.endswith('.py'):compile(Path(n).read_text(),n,'exec')\nreceipt={'status':'all_sources_actual_sha_and_compile','source_sha256':actual};Path('deployment-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))"
  req={'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'},'timeout_seconds':30};save('deployment-check-intent.json',req);x=call('run_command',req);save('deployment-tool.json',x);assert x['exit_code']==0;s.update(status='complete',actual_remote=json.loads(x['stdout']));save('deployment-state.json',s);print('deployment complete')
 except Exception as e:s.update(status='failed_or_unknown_preserved',error=str(e));save('deployment-state.json',s);raise
 raise SystemExit()
if a.action in ['prepare-start','start']:
 prep=a.action=='prepare-start';prefix='preparation-' if prep else '';assert read(R/'deployment-state.json')['status']=='complete' and not(R/(prefix+'start-intent.json')).exists()
 if not prep:
  assert read(R/'preparation-job.json')['returncode']==0 and read(R/'preparation.json')['status']=='complete_input_only_preparation' and read(R/'bound-deployment.json')['verified'] is True
 env={**plan['GPU_environment'],'CUDA_VISIBLE_DEVICES':''} if prep else {'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
 req={'command':shlex.quote(PYTHON)+(' prepare_launch.py' if prep else ' launch.py'),'cwd':REMOTE,'env':env};save(prefix+'start-intent.json',{'request':req,'plan_sha256':expected['plan.json'],'source_sha256':sha(Path(__file__).read_bytes()),'written_before_call':True});x=call('start_process',req);save(prefix+'managed-job.json',x);print(json.dumps(x));raise SystemExit()
if a.action=='publish-bound':
 prep=read(R/'preparation.json');assert prep['status']=='complete_input_only_preparation' and read(R/'preparation-job.json')['returncode']==0;assert prep['artifacts']['queries']['sha256']==plan['expected_keys']['sha256'];assert not(R/'bound-deployment-intent.json').exists()
 b=(R/'bound-plan.json').read_bytes();bound=json.loads(b);assert bound['plan_sha256']==expected['plan.json'] and bound['preparation_sha256']==sha((R/'preparation.json').read_bytes());save('bound-deployment-intent.json',{'bytes':len(b),'sha256':sha(b)});x=call('write_file',{'path':REMOTE+'/bound-plan.json','content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});save('bound-deployment.json',{'response':x,'verified':x['sha256']==sha(b) and x['size']==len(b)});assert read(R/'bound-deployment.json')['verified'];print('bound plan actual remote SHA verified');raise SystemExit()
prep=a.action=='prepare-collect';prefix='preparation-' if prep else '';job=read(R/(prefix+'managed-job.json'));x=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096});save(prefix+'process-status.json',x)
if x['running']:
 if not prep:
  try:state=json.loads(fetch('cache-state.json'));print(json.dumps({k:state.get(k) for k in ['status','completed_batches','completed_rows','elapsed_seconds']}))
  except Exception as e:print(json.dumps({'running':True,'state_read_error':str(e)}))
 else:print('CPU input preparation still running')
 raise SystemExit()
names=['preparation-job.json','preparation.json','preparation.stderr.log','preparation.stdout.log'] if prep else ['run-state.json','cache-result.json','cache.stderr.log','cache.stdout.log']
for n in names:
 try:fetch(n)
 except Exception as e:save(n+'.collection-error.json',{'error':str(e)})
print(json.dumps(read(R/names[0])))
