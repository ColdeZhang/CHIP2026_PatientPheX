"""Freeze one authorized 53+27 campaign; no remote/model/quality operation."""
from pathlib import Path
import ast, gzip, hashlib, io, json, shutil, tarfile, time

W=Path(__file__).resolve().parent.parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/response-code-equivariance-v1')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
D=B/'gpu-v1'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def check(r):
 a=ref(r['path']);assert all(a[k]==r[k] for k in ['bytes','sha256'])
def main():
 started=time.monotonic();D.mkdir(exist_ok=False)
 native=read(B/'native100-v1/terminal/native-result.json');assert native['status']=='PASS_ACTUAL_NATIVE_CPU_CYCLIC1_NO_MODEL'
 assert (native['rows'],native['new_NN_rows'],native['cached_original_rows'])==(80,53,27)
 assert native['groups']['cyclic1']['total_tokens']==685372 and native['groups']['cyclic1']['max_tokens']==14570
 assert read(B/'native100-v1/terminal/wrapper-complete.json')['returncode']==0
 policy=read(B/'endpoint-policy.json');cache=read(policy['single_patient_cache_refs']['path'])
 for n in ['infer_full80.py','probe_runner.py','legacy_runner.py','cyclic1_codec.py','original_response_codec.py','launch.py','model-source-manifest.json']:
  shutil.copyfile(W/'inference'/n,D/n)
 shutil.copyfile(B/'preparation-v1/held.cyclic1.jsonl',D/'held.response-code.jsonl')
 audit=[x for x in read(B/'native100-v1/terminal/native-row-audit.json') if x['arm']=='cyclic1'];assert len(audit)==80 and [x['index'] for x in audit]==list(range(80))
 save(D/'row-audit.json',audit)
 cached=[r['original_index'] for r in cache['cache']];assert len(cached)==27 and cached==[i for i,a in enumerate(audit) if a['cache_original_raw']]
 cachefiles={}
 for r in cache['cache']:
  i=r['original_index'];cachefiles[str(i)]={}
  for folder,item in r['files'].items():
   check(item);name=f'cache-{folder}-{i:03d}.json';shutil.copyfile(item['path'],D/name);cachefiles[str(i)][folder]=dict(name=name,bytes=item['bytes'],sha256=item['sha256'])
 old=read('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-v1/F1-full80-v1/plan.json')
 history=read('/workspace/patientphex-runs/round2-20261023/research/nonthinking-sampled-consensus-full64-20261024/gpu-plan.json')
 runtime=dict(old['expected_runtime']);runtime['torch']='2.7.1+cu126'
 env=dict(history['env']);env['CUDA_VISIBLE_DEVICES']='GPU-126293b9-6677-8464-cee7-01e693efbf29'
 plan=dict(schema='patient-response-code-cyclic1-F1-v1',status='ROOT_AUTHORIZED_UNIQUE_53_FRESH_27_CACHE',mount_id=history['mount_id'],gpu_uuid=env['CUDA_VISIBLE_DEVICES'],physical_gpu=0,python=history['python'],model=history['model'],remote='patient-response-code-equivariance/gpu-v1',adapter=P+'/patient-response-code/normal2ep-v1/checkpoints/epoch-002/adapter',adapter_files=cache['adapter_files'],env=env,cap_seconds=1200,term_grace_seconds=30,input_limit=14848,generation_limit=1600,cached_original_indices=cached,cache_files=cachefiles,fresh_rows=53,training_completed_epochs=2,training_global_step=32,configured_eos_ids=[248044,248046],expected_runtime=runtime,runtime_source_SHA=native['runtime_source_SHA'],native_census=ref(B/'native100-v1/terminal/native-result.json'),original_C_plan=policy['original_C_plan'],original_C_terminal=policy['original_C_terminal'],policy=ref(B/'endpoint-policy.json'),diagnostic_policy_addendum=ref(W/'evidence/endpoint-policy-diagnostic-reporting-addendum.json'),raw_first=True,whole_row_abstention=True,no_repair=True,no_new_training=True,model_and_adapter_parameters=8997081600,scope=policy['causal_scope'],fresh_input_tokens=475349,cached_input_tokens=210023,all_input_tokens=685372,max_input_tokens=14570)
 assert plan['adapter_files']==old['adapter_files']
 plan['files']={p.name:{k:v for k,v in ref(p).items() if k!='path'} for p in sorted(D.iterdir()) if p.is_file()}
 save(D/'plan.json',plan)
 oldq=read('/workspace/patientphex-worktrees/explore-patient-response-code/solutions/patient_response_code/quality-plan.json')
 q={k:oldq[k] for k in ['input','gold','evaluator','validator','hpo']}
 q.update({k:policy[k] for k in ['B4_pred','B4_metrics','original_C_strict_pred']})
 q.update(prompts=ref(B/'preparation-v1/held.cyclic1.jsonl'),policy=ref(B/'endpoint-policy.json'),cache_refs=policy['single_patient_cache_refs'],source_sha256=ref(W/'quality/final.py')['sha256'],fixed_source_refs=[ref(W/'quality'/n) for n in ['common.py','literal_union_helpers.py']],gold_access='Only after both complete predictions common freeze',changed_primary_calls=1,unchanged_primary_calls=0,diagnostic_calls=1,old_calls=0)
 save(W/'evidence/quality-plan.json',q);shutil.copyfile(W/'evidence/quality-plan.json',D/'quality-plan.local.json')
 # Deployment deliberately excludes local quality/gold references and operations.
 members=list(plan['files'])+['plan.json']
 with tarfile.open(D/'deployment.tar.gz','w:gz') as tar:
  for n in members:tar.add(D/n,arcname=n)
 save(D/'deployment-manifest.json',dict(archive=ref(D/'deployment.tar.gz'),files={n:{k:v for k,v in ref(D/n).items() if k!='path'} for n in members}))
 # Discriminating new controls: runtime/model generation AST unchanged; 27 exact
 # caches; strict cyclic parser handles inverse, duplicate/unknown/missing keys.
 import sys;sys.path.insert(0,str(W));import cyclic1_codec as codec
 synthetic=dict(patient_ids=['p0','p1','p2'],code_map={'C001':'HP:0000001'},response_patient_codes=['PT001','PT002','PT003'],response_code_map={'PT001':'p2','PT002':'p0','PT003':'p1'},response_representation=codec.original.REPRESENTATION,messages=[dict(role='user',content=codec.table_text({'PT001':'p2','PT002':'p0','PT003':'p1'}))])
 assert codec.decode_response('{"C001":["PT001","PT003"]}',synthetic)=={'C001':['p2','p1']}
 bad=['{"C001":["PT001","PT001"]}','{"C001":["PT004"]}','{}','{"C001":[],"C001":[]}']
 for text in bad:
  try:codec.decode_response(text,synthetic)
  except ValueError:pass
  else:raise AssertionError('strict parser accepted invalid')
 original=Path('/workspace/patientphex-worktrees/explore-patient-response-code/solutions/patient_response_code/inference/infer_full80.py')
 def calls(path):return [ast.dump(x,include_attributes=False) for x in ast.walk(ast.parse(path.read_text())) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='generate']
 assert calls(original)==calls(W/'inference/infer_full80.py') and len(calls(original))==1
 checked=[]
 for folder in ['inference','quality']:
  for p in (W/folder).glob('*.py'):compile(p.read_text(),str(p),'exec');checked.append(ref(p))
 result=dict(status='PASS_NEW_CYCLIC_CACHE_SOURCE_CPU_CONTRACT',seconds=time.monotonic()-started,caches=27,cache_SHA_files=81,new_NN_rows=53,all80_actual_native_evidence_reused=True,exact_original_generate_AST=True,strict_inverse_and_4_rejection_cases=True,source_compile=checked,no_tokenizer_or_model=True,no_quality=True,no_gold_read=True,plan=ref(D/'plan.json'),quality_plan=ref(W/'evidence/quality-plan.json'))
 save(W/'evidence/GPU-source-CPU-contract.json',result)
 save(W/'evidence/GPU-plan.json',plan)
 save(W/'evidence/GPU-ready.json',dict(directory=str(D),plan=ref(D/'plan.json'),deployment_manifest=ref(D/'deployment-manifest.json'),CPU_contract=ref(W/'evidence/GPU-source-CPU-contract.json'),native_terminal=ref(B/'native100-v1/terminal-manifest.json'),start='python solutions/response_code_equivariance/inference/operations.py start --directory '+str(D),status='PREPARED_NOT_STARTED'))
 print(json.dumps({k:v for k,v in result.items() if k!='source_compile'}))

if __name__=='__main__':main()
