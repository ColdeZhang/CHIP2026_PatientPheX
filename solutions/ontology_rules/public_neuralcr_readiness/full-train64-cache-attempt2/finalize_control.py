from pathlib import Path
import argparse,base64,hashlib,json,shlex,sys
R=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/full-train64-cache-attempt2';PYTHON='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/.venv/bin/python';sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','run','collect']);a=p.parse_args();sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(n,args):
 x=c.request('tools/call',{'name':n,'arguments':{'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f',**args}})
 if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
 return x['structuredContent']
if a.action=='deploy':
 assert not(R/'finalization-deploy-intent.json').exists();expected={n:sha((R/n).read_bytes()) for n in ['finalize_cpu.py','finalization-plan.json']};save('finalization-deploy-intent.json',expected)
 for n,digest in expected.items():
  b=(R/n).read_bytes();x=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert x['sha256']==digest and x['size']==len(b)
 save('finalization-deployment.json',{'status':'complete_actual_write_SHAs','files':expected});print('finalization source/plan deployed');raise SystemExit()
if a.action=='run':
 assert json.loads((R/'finalization-deployment.json').read_text())['status']=='complete_actual_write_SHAs' and not(R/'finalization-start-intent.json').exists()
 code="from pathlib import Path;import subprocess,sys,os,json,time,datetime\nassert os.environ['CUDA_VISIBLE_DEVICES']=='' and not Path('finalization-job.json').exists()\ncmd=['timeout','--signal=TERM','--kill-after=5s','45s',sys.executable,'finalize_cpu.py'];j={'status':'running','command':cmd,'cap_seconds':45,'TERM_grace_seconds':5,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};Path('finalization-job.json').write_text(json.dumps(j));t=time.monotonic()\nwith Path('finalization.stdout.log').open('xb') as out,Path('finalization.stderr.log').open('xb') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=52)\nj.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,elapsed_seconds=time.monotonic()-t,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());Path('finalization-job.json').write_text(json.dumps(j,indent=2));print(json.dumps(j));raise SystemExit(p.returncode)"
 req={'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},'timeout_seconds':55};save('finalization-start-intent.json',req);x=call('run_command',req);save('finalization-tool.json',x);print(json.dumps(x));raise SystemExit()
for n in ['finalization-job.json','finalization-result.json','finalization.stdout.log','finalization.stderr.log']:
 x=call('read_file',{'path':REMOTE+'/'+n,'offset':0,'limit':65536});b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)<65536;(R/n).write_bytes(b)
print((R/'finalization-job.json').read_text())
