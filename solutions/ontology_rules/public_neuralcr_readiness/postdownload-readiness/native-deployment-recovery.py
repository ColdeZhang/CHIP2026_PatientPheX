"""Reconcile only immutable native deployment files after one unknown write; no repeat model start."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,sys
P=Path(__file__).resolve().parent;REMOTE='public-neuralcr-readiness/postdownload-readiness-v1';PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';PYTHON=PROJECT+'/public-phenotagger-reuse/.venv/bin/python';MOUNT='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,v):(P/n).write_text(json.dumps(v,indent=2)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inventory','start']);a=parser.parse_args()
assert not (P/'native-start-intent.json').exists() and not(P/'native-managed-job.json').exists()
names=['stage_runner.py','inspect_complete.py','repair_fasttext_header.py','check_native.py','ncrmodel.py','native-cpu-plan.json','HP.tar.gz.assembly.json','shape-inventory.json','repair-report.json']
expected={n:{'bytes':(P/n).stat().st_size,'sha256':sha((P/n).read_bytes())} for n in names}
# Bind the original experiment plan; only the transport controller differs.
assert expected['native-cpu-plan.json']['sha256']=='0107a27be02792eb4454ed3a1cf35a5b864f59afcbf45930c46b930a9e109560'
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':MOUNT,**args}})
 if r.get('isError'):raise RuntimeError(str(r.get('content'))[:600])
 return r['structuredContent']
if a.action=='inventory':
 assert not (P/'native-deployment-recovery-inventory.json').exists()
 code="from pathlib import Path;import hashlib,json\nexpected="+repr(expected)+"\nactual={}\nfor n,ref in expected.items():\n p=Path(n);assert not p.is_symlink()\n actual[n]={'exists':p.exists()}\n if p.exists():\n  b=p.read_bytes();actual[n].update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())\nprint(json.dumps(actual))"
 result=call('run_command',{'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':30})
 save('native-deployment-recovery-inventory-tool.json',result);assert result['exit_code']==0
 actual=json.loads(result['stdout']);missing=[]
 for n,ref in expected.items():
  if not actual[n]['exists']:missing.append(n)
  else:assert {k:actual[n][k] for k in ['bytes','sha256']}==ref,('unexpected_different_bytes_stop',n)
 save('native-deployment-recovery-inventory.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected':expected,'actual':actual,'missing':missing,'source_sha256':sha(Path(__file__).read_bytes()),'no_writes_or_model_load':True})
 print(json.dumps({'verified':len(expected)-len(missing),'missing':missing}));raise SystemExit()
inventory=read(P/'native-deployment-recovery-inventory.json');assert inventory['expected']==expected
for n in inventory['missing']:
 b=(P/n).read_bytes();assert len(b)<=65536
 r=call('write_file',{'path':REMOTE+'/'+n,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':False})
 assert r['sha256']==sha(b) and r['size']==len(b)
state=read(P/'verified-assets.json');hp,ft=state['assets'];links={PROJECT+'/'+REMOTE+'/extracted/HP/'+Path(x['path']).name:x['path'] for x in hp['extracted']};links[PROJECT+'/'+REMOTE+'/pmc_model_new.bin']=ft['path']
code="from pathlib import Path;import os,hashlib,json\nexpected="+repr(expected)+"\nactual={}\nfor n in expected:\n b=Path(n).read_bytes();actual[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}\nassert actual==expected\nlinks="+repr(links)+"\nfor dst,src in links.items():assert os.path.samefile(src,dst)\nprint(json.dumps(actual))"
r=call('run_command',{'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':30});save('native-deployment-recovery-final-tool.json',r);assert r['exit_code']==0
save('native-deployment.json',{'files':expected,'remote_actual':json.loads(r['stdout']),'asset_hardlinks':links,'recovered_after_unknown_write':True,'inventory_sha256':sha((P/'native-deployment-recovery-inventory.json').read_bytes())})
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1'}
command=shlex.quote(PYTHON)+' stage_runner.py native';intent={'command':command,'env':env,'cwd':REMOTE,'plan_sha256':expected['native-cpu-plan.json']['sha256'],'launcher_sha256':sha(Path(__file__).read_bytes())}
save('native-start-intent.json',intent)
job=call('start_process',{'command':command,'cwd':REMOTE,'env':env});save('native-managed-job.json',job);save('native-launch.json',intent);print(json.dumps(job))
