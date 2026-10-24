"""CPU-only checks for finite admission; no model/runtime import or remote mutation."""
import hashlib,json,os,subprocess,sys,time,ast
from pathlib import Path
from prepare import S,D,ref,save
out=D/'admission-contract-v3';out.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(S),PATIENTPHEX_DURABILITY_ROOT=str(out))
checks=[]
def run(name,code,deadline,ok):
 d=out/name;d.mkdir();e=dict(env,COPY_PROPOSER_LAUNCH_VALID_UNTIL=str(deadline));p=subprocess.run([sys.executable,'-c',code],cwd=d,env=e,capture_output=True,text=True)
 assert (p.returncode==0)==ok,(name,p.returncode,p.stderr)
 if not ok:assert not (d/'MODEL_MARKER').exists()
 checks.append(dict(name=name,returncode=p.returncode,rejection_record=(d/'admission-rejected.json').exists(),no_model_marker=not (d/'MODEL_MARKER').exists()))
run('valid',"from admission_v3 import admit; import sys; assert admit('cpu')>0; assert 'torch' not in sys.modules",time.time()+120,True)
run('expired',"from admission_v3 import admit; from pathlib import Path; admit('cpu'); Path('MODEL_MARKER').touch()",time.time()-1,False)
run('mismatch',"from admission_v3 import admit; from pathlib import Path; admit('cpu',-1); Path('MODEL_MARKER').touch()",time.time()+120,False)
for name,script,args in [('driver','job_driver_admission_v3.py',['--plan','plan.json']),('normalizer','normalize_entities_admission_v3.py',sum((['--'+x,'DOES_NOT_EXIST'] for x in ('extraction','normalizer-source','references','input','train','hpo','bank','model','output')),[]))]:
 d=out/name;d.mkdir();deadline=time.time()-1
 (d/'plan.json').write_text(json.dumps(dict(storage_parent=str(out),budget_seconds=180,grace_seconds=30,launch_not_after_unix=deadline)))
 p=subprocess.run([sys.executable,str(S/script)]+args,cwd=d,env=dict(env,COPY_PROPOSER_LAUNCH_VALID_UNTIL=str(deadline)),capture_output=True,text=True)
 assert p.returncode!=0 and 'Frozen launch validity expired' in p.stderr
 assert (d/'admission-rejected.json').exists() and not (d/'DOES_NOT_EXIST').exists()
 checks.append(dict(name=name+'_expired_before_source_or_model',returncode=p.returncode,rejection_record=True))
for name in ['remote_recovery_v3.py','collect_recovery_v3.py']:
 tree=ast.parse((S/name).read_text());strings={}
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):strings[node.targets[0].id]=node.value.value
 if name.startswith('remote'):
  for key in ['INVENTORY','DEPLOY','CLAIM','OBSERVE']:compile(strings['REMOTE_COMMON']+'\n'+strings[key],name+':'+key,'exec')
 else:
  for key,value in strings.items():
   if key.startswith('REMOTE'):compile(value,name+':'+key,'exec')
 checks.append(dict(name=name+'_embedded_source_compile',passed=True))
old=json.loads((S/'normalize-plan-essential-v2.json').read_text())
for item in old['payload_sources'].values():assert ref(item['path'])==item
save(S/'admission-contract-v3.json',dict(status='PASS_CPU_ONLY',checks=checks,original_payload_SHA_preserved=True,CUDA_hidden=True,no_model_or_remote_calls=True,source=ref(__file__)))
print(json.dumps(ref(S/'admission-contract-v3.json')))
