"""Deploy pinned small sources then start one official-release download, never model/NN."""
from pathlib import Path
import sys,json,hashlib,base64,shlex
R=Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();m='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':m,**a}});assert not r.get('isError'),r;return r['structuredContent']
assert not (R/'download-job.json').exists();files=['download_assets.py','download-plan.json'];digests={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files}
code="from pathlib import Path;assert not Path('public-neuralcr-readiness').exists();print('new_destination_available')"
r=call('run_command',{'command':'public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'.','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':10});assert r['exit_code']==0,r
for n in files:call('write_file',{'path':'public-neuralcr-readiness/'+n,'content':base64.b64encode((R/n).read_bytes()).decode(),'encoding':'base64','create_parents':True})
code="from pathlib import Path;import hashlib,json;expected="+repr(digests)+";actual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected;print(json.dumps(actual))"
r=call('run_command',{'command':'../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'public-neuralcr-readiness','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':10});assert r['exit_code']==0,r
(R/'download-deployment.json').write_text(json.dumps({'source_sha256':digests,'remote_actual':json.loads(r['stdout'])},indent=2)+'\n')
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};command='timeout --signal=TERM --kill-after=30s 600s ../public-phenotagger-reuse/.venv/bin/python download_assets.py';job=call('start_process',{'command':command,'cwd':'public-neuralcr-readiness','env':env});(R/'download-job.json').write_text(json.dumps({'command':command,'env':env,'source_sha256':digests,'start_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'result':job},indent=2)+'\n');print(json.dumps(job))
