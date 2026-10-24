"""Deploy only owner-frozen wrapper and reuse checker, then start one bounded native job."""
from pathlib import Path
import json,sys,shlex,hashlib,base64,datetime
R=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();m='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':m,**a}});assert not r.get('isError'),r;return r['structuredContent']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'run-job.json').exists()
execution=json.loads((R/'GPU-execution-plan.json').read_text());assert sha(R/'launch_cache.py')==execution['wrapper_sha256']=='c1c4c6422d6d68988fe95642ee0c02908cd65c95010b2189c08f5641a210077d';assert sha(R/'plan.json')==execution['parent_source_plan_sha256']=='7cf2a4165e066319c96ad823a801c0981cdc9cdf63dd3b0bae565d62a7472d8c';assert execution['operative_stricter_GPU_cap_seconds']==180
prep=json.loads((R/'CPU-preparation.json').read_text());assert prep['documents']==16 and prep['new_forward_keys']==50548 and prep['reused_forward_keys']==4655
receipt=json.loads((R/'CPU-preparation-terminal.json').read_text());assert all(not x['running'] for x in receipt.values())
files=['launch_cache.py','GPU-execution-plan.json','verify_reused_cache.py'];hashes={n:sha(R/n) for n in files}
# Owned new files only. Bound shared model/input/source resources are checked by the wrapper.
for n in files:call('write_file',{'path':'public-phenotagger-biobert/dev16-cache/'+n,'content':base64.b64encode((R/n).read_bytes()).decode(),'encoding':'base64','create_parents':False})
code="from pathlib import Path;import hashlib,json;D=Path('dev16-cache');assert not (D/'run-state.json').exists();assert not Path('native-dev16-cache').exists();assert not Path('native-dev16-final-cache').exists();expected="+repr(hashes)+";actual={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in expected};assert actual==expected;print(json.dumps(actual))"
r=call('run_command',{'command':'../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'public-phenotagger-biobert','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});assert r['exit_code']==0,r
(R/'GPU-wrapper-deployment.json').write_text(json.dumps({'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files_sha256':hashes,'remote_bytes_verified':json.loads(r['stdout']),'only_new_owner_files':True},indent=2)+'\n')
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1','FORCE_TF_AVAILABLE':'1','USE_TF':'1','USE_TORCH':'0'}
command='timeout --signal=TERM --kill-after=30s 360s ../public-phenotagger-reuse/.venv/bin/python dev16-cache/launch_cache.py';job=call('start_process',{'command':command,'cwd':'public-phenotagger-biobert','env':env});(R/'run-job.json').write_text(json.dumps({'command':command,'env':env,'start_source_sha256':sha(__file__),'wrapper_sha256':sha(R/'launch_cache.py'),'GPU_execution_plan_sha256':sha(R/'GPU-execution-plan.json'),'plan_sha256':sha(R/'plan.json'),'result':job},indent=2)+'\n');print(json.dumps(job))
