from pathlib import Path
import sys,json,hashlib,shlex,datetime,base64
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
E=Path(__file__).resolve().parent;A=Path('/workspace/patientphex-runs/round2-20261023/research/medical-small-model-readiness/assets')
remote='research/medgemma-relation-control/preparation';output='research/medgemma-relation-control/cpu-contract-v1'
planfile=E/'cpu-deployment-plan.json';assert not planfile.exists()
files={n:E/n for n in ['train_medgemma32.py','medgemma_interface.py','qwen35_smoke.py','qwen35_relation_cost.py','probe_longest.py','infer_medgemma80.py']}
files['expected-files.json']=A/'expected-files.json'
for n,r in json.loads((E/'data-locators.json').read_text()).items():files[n]=Path(r['path'])
refs={n:{'local':str(p),'remote':remote+'/'+n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in files.items()}
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/round2/runtime','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
argv=['env',*[k+'='+v for k,v in env.items()],'/usr/bin/timeout','--signal=TERM','--kill-after=10s','60','/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',remote+'/train_medgemma32.py','--mode','cpu','--train',remote+'/train.jsonl','--excluded-input',remote+'/excluded.input.jsonl','--data-manifest',remote+'/preparation-manifest.json','--inference-prompts',remote+'/meta.prompts.jsonl','--model','/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30','--model-source-manifest',remote+'/expected-files.json','--output-dir',output]
plan={'status':'CPU_only_before_actual_contract','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mount_id':io.M,'files':refs,'argv':argv,'no_gpu_no_weights':True,'output':output}
planfile.write_text(json.dumps(plan,indent=2)+'\n')
pre=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();assert not Path('+repr(output)+').exists();p.mkdir(parents=True,mode=0o700);print("unique CPU source path prepared")'),'cwd':'.','timeout_seconds':10})
(E/'cpu-deployment-preflight.json').write_text(json.dumps(pre,indent=2)+'\n');assert pre['exit_code']==0
for n,p in files.items():io.upload(str(p),remote+'/'+n)
code='from pathlib import Path;import hashlib,json;refs='+repr(refs)+';assert all(Path(r["remote"]).stat().st_size==r["bytes"] and hashlib.sha256(Path(r["remote"]).read_bytes()).hexdigest()==r["sha256"] for r in refs.values());print(json.dumps(dict(status="actual_sources_and_data_SHA_match",files=len(refs))))'
r=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':20});(E/'cpu-deployment-sha.json').write_text(json.dumps(r,indent=2)+'\n');assert r['exit_code']==0
(E/'cpu-execution-intent.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'timeout_seconds':60,'no_weights_no_GPU':True},indent=2)+'\n')
r=io.call('run_command',{'command':shlex.join(argv),'cwd':'.','timeout_seconds':70});(E/'actual-cpu-command.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
# Read existing outputs even for a failed CPU contract; never launch replacement here.
code='from pathlib import Path;import json,base64,hashlib;p=Path('+repr(output)+');print(json.dumps({f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),base64=base64.b64encode(f.read_bytes()).decode()) for f in p.glob("*.json") if f.stat().st_size<300000}))'
z=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':15});assert z['exit_code']==0
D=E/'actual-cpu';D.mkdir(exist_ok=False);receipt={}
for n,v in json.loads(z['stdout']).items():
 b=base64.b64decode(v.pop('base64'));assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];(D/n).write_bytes(b);receipt[n]=v
(E/'cpu-recovered-files.json').write_text(json.dumps(receipt,indent=2)+'\n')
if r['exit_code']!=0:raise RuntimeError('CPU contract failed; original outputs preserved, no repeat launched')
print('CPU_CONTRACT_COMPLETE_NO_GPU')
