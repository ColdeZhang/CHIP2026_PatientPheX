"""Freeze successor before F1 quality; metadata-only CPU preflight, no native/GPU/quality."""
from pathlib import Path
import json,hashlib,copy,shutil,datetime,time,sys,tarfile,difflib
F=Path(__file__).parent;D=F.parent;I=D/'full80-preparation';S=F/'full58-source-capacity-fix';P=F/'final58-preparation';P.mkdir(exist_ok=True);assert not (P/'preparation-manifest.json').exists();stage=P/'deployment';stage.mkdir(exist_ok=True)
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness';REMOTE=ROOT+'/f0-full58-v1';OLD=ROOT+'/f0-longest-v2'
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 p.parent.mkdir(exist_ok=True,parents=True)
 if p.name=='conditional-decision.json' and p.exists():return
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def remote_ref(p,path):return {**ref(p),'path':path}
assert not (I/'quality').exists(),'Do not claim decision frozen before F1 quality after quality exists'
observed=json.loads((I/'monitor-state.json').read_text());assert not observed['terminal']
row=json.loads((F/'probe-attempt2/recovered/terminal/row0.json').read_text());assert row['generated_tokens']==3427 and row['native_EOS_observed'] and row['normalized_valid']
decision={'status':'ROOT_SUCCESSOR_DECISION_FROZEN_BEFORE_F1_NEW_QUALITY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_F1_plan':ref(I/'plan.json'),'original_F1_evaluation_template':ref(I/'evaluation-template.json'),'original_F1_evaluator':ref(I/'evaluate_full80.py'),'original_policy_unchanged':'F1 primary R1-alone, only-secondary oldSFT32 union R1, both original retain advisories, original automatic_migration=false preserved','new_successor_gate':'Only complete and verified F1 oldSFT32_union_R1 endpoint satisfying original retain permits F0. Standalone result still reported under original primary name; no retroactive policy change.','exact_retain':'candidate.micF1 + candidate.macF1 > old.micF1 + old.macF1 AND candidate.macF1 >= old.macF1','old_nonrelation_bytes_fixed_equivalence':'With unchanged menF1/docF1 and nonrelation bytes, this is equivalent to Score strict increase and macro nondecrease.','F0_if_gate_pass':{'main_seconds':6000,'TERM_seconds':30,'cached_indices':[30],'fresh_rows':57,'seed_every_fresh_row':20261024,'max_new_tokens':4096,'endpoints':['R1 standalone','oldSFT48 union R1'],'two_complete_predictions_frozen_before_each_unique_v2':True},'if_gate_fails':'Close F0 full58 extension; retain longest format/cost, do not start GPU or run F0 quality','latest_F1_progress_at_freeze':{'at':observed['observed_at'],'rows':observed['progress']['rows_completed'],'terminal':observed['terminal']},'F1_quality_directory_existed_at_freeze':False,'cost_evidence':ref(F/'probe-attempt2/summary.json'),'estimated_max57_generation_seconds_from_single_probe':57*4096/row['tokens_per_second_including_prefill'],'cost_caveat':'Single longest throughput extrapolation only;6000+30 is fixed cap, not guaranteed completion.','independent_F1_terminal_review_required':True}
save(P/'conditional-decision.json',decision)
plan=json.loads((S/'plan.template.json').read_text());oldplan=json.loads((F/'probe-attempt2/plan.json').read_text());deployment=[]
for name,digest in plan['sources'].items():
 assert ref(S/name)['sha256']==digest;shutil.copyfile(S/name,P/name)
launcher=P/'launch_full58.py';before=launcher.read_text();needle='validate_plan(plan)';assert needle in before
# Launcher is the only process starter. CPU input preflight never calls it.
addition="assert plan['conditional_F1_gate']['status']=='PASS_F1_UNION_ORIGINAL_RETAIN_VERIFIED', 'F1 conditional successor gate pending or failed';gate_ref=plan['conditional_F1_gate']['evidence'];gate_path=Path(gate_ref['path']);assert gate_path.stat().st_size==gate_ref['bytes'] and sha(gate_path)==gate_ref['sha256'];gate=json.loads(gate_path.read_text());assert gate['status']=='PASS_F1_UNION_ORIGINAL_RETAIN_VERIFIED' and gate['union_original_retain'] is True and gate['F1_terminal_review_passed'] is True\n    "
launcher.write_text(before.replace(needle,addition+needle,1));(P/'conditional-launch-gate.diff').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),launcher.read_text().splitlines(keepends=True),fromfile='capacity-fixed-disabled-launcher',tofile='conditional-successor-launcher')));plan['sources']['launch_full58.py']=ref(launcher)['sha256']
for key in ['assets_result','header_inventory','expected_files','yarn_result','derived_shapes','derived_config','CPU_result','all58_audit','target_copy_ACK','transport_manifest']:plan[key]=oldplan[key]
plan.update(execution_ready=False,status='CONDITIONAL_F0_FULL58_PREPARED_WAITING_F1_UNION_GATE',full_inference_authorization='Root authorized6000+TERM30/idx30 cache1+57fresh only after original complete F1 union endpoint passes original retain and terminal verification. Until then execution_ready=false.',process_budget_seconds=6000,TERM_grace_seconds=30,cached_indices=[30],new_indices=[i for i in range(58) if i!=30],conditional_F1_gate={'status':'PENDING_F1_COMPLETE_TWO_ENDPOINTS_AND_REVIEW','evidence':None},conditional_decision=remote_ref(P/'conditional-decision.json',REMOTE+'/conditional-decision.json'))
plan['evaluation_contract_template'].update(process_budget_seconds=6000,TERM_grace_seconds=30,cached_indices=[30]);plan['scope']='Conditional successor only, fixed F0 58 original inputs,1 exact completed cache+57fresh; same model/tokenizer/scale1/BF16/codec and per-rowseed,6000+30. No F0 quality before complete58 and paired freeze.'
plan['cache_policy']='Exactly original index30 completed A800 probe reused without new NN; full input/model/generation/raw/IDs/source/wrapper binds,57 other original rows fresh same seed. No new sample represented as recovered raw.'
for key,p,name in [('all58_encoded',F/'cpu/recovered/all58-published-encoded.jsonl.gz','inputs/all58-published-encoded.jsonl.gz'),('prompts_gzip',F/'cpu/prompts.jsonl.gz','inputs/prompts.jsonl.gz')]:
 plan[key]=remote_ref(p,REMOTE+'/'+name);q=stage/name;q.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,q);deployment.append({'name':name,'local':ref(p),'remote':plan[key]})
T=F/'probe-attempt2';plan['cached_rows']={'30':remote_ref(T/'recovered/terminal/row0.json',OLD+'/terminal/row0.json')};proof={'plan':(T/'plan.json',OLD+'/plan.json'),'result':(T/'recovered/terminal/result.json',OLD+'/terminal/result.json'),'wrapper':(T/'recovered/wrapper-result.json',OLD+'/wrapper-result.json'),'source':(T/'runner.py',OLD+'/runner.py'),'launcher':(T/'launcher.py',OLD+'/launcher.py'),'raw':(T/'recovered/terminal/row0.raw.json',OLD+'/terminal/row0.raw.json'),'ids':(T/'recovered/terminal/row0.ids.json',OLD+'/terminal/row0.ids.json')};plan['cached_provenance']={'30':{k:remote_ref(p,r) for k,(p,r) in proof.items()}}
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
save(P/'plan.pending.json',plan)
# Same actual contract, only an in-memory CPU shadow maps immutable refs to local identical bytes.
# No executable on-disk plan is created, no launcher/native model/tokenizer is invoked.
local=copy.deepcopy(plan);local['execution_ready']=True
known={}
for p in D.rglob('*'):
 if p.is_file() and p.stat().st_size<1000000 and not str(p).startswith(str(P)):
  r=ref(p);known.setdefault((r['bytes'],r['sha256']),p)
def relocate(v):
 if isinstance(v,dict):
  if set(v)=={'path','bytes','sha256'} and (v['bytes'],v['sha256']) in known:return {**v,'path':str(known[(v['bytes'],v['sha256'])])}
  return {k:relocate(x) for k,x in v.items()}
 if isinstance(v,list):return [relocate(x) for x in v]
 return v
local=relocate(local);sys.path.insert(0,str(P));from full58_contract import load_inputs
started=time.monotonic();rows,enc,aud,cache=load_inputs(local);seconds=time.monotonic()-started
CPU={'status':'PASS_ACTUAL_F0_58_INPUT_AND_ONE_EXACT_CACHE_CPU_ONLY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_pending':ref(P/'plan.pending.json'),'source':ref(P/'full58_contract.py'),'program_seconds':seconds,'rows':len(rows),'tokens':sum(len(v['input_ids']) for v in enc),'cached_indices':sorted(cache),'new_indices':plan['new_indices'],'cached_generated_IDs_exact':cache[30]['generated_ids']==row['generated_ids'],'actual_cache_bytes':ref(T/'recovered/terminal/row0.json'),'CPU_shadow_only':'execution_ready true only in in-memory pure input-loader shadow with local identical-byte refs; no executable plan/GPU/launcher or native tokenizer/model imports','real_plan_execution_ready':False,'F1_quality_read':False,'gold_or_evaluator_read':False,'model_or_GPU_initialized':False};save(P/'CPU-input-contract.json',CPU)
for name in plan['sources']:
 shutil.copyfile(P/name,stage/name);deployment.append({'name':name,'local':ref(P/name),'remote':remote_ref(P/name,REMOTE+'/'+name)})
for name in ['conditional-decision.json','plan.pending.json']:
 shutil.copyfile(P/name,stage/name);deployment.append({'name':name,'local':ref(P/name),'remote':remote_ref(P/name,REMOTE+'/'+name)})
save(P/'deployment-manifest.json',{'remote':REMOTE,'files':deployment,'unchanged_existing_remote_metadata_and_cache_refs_not_reuploaded':True,'no_GPU_start':True})
with tarfile.open(P/'deployment.tar.gz','w:gz') as tf:
 for v in deployment:
  p=stage/v['name'];info=tf.gettarinfo(str(p),arcname=v['name']);info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o600
  with p.open('rb') as f:tf.addfile(info,f)
save(P/'preparation-manifest.json',{'status':'F0_CONDITIONAL_FULL58_CPU_PREPARATION_COMPLETE_NOT_EXECUTABLE','conditional_decision':ref(P/'conditional-decision.json'),'plan':ref(P/'plan.pending.json'),'CPU':ref(P/'CPU-input-contract.json'),'source_diff':ref(P/'conditional-launch-gate.diff'),'sources':[ref(P/n) for n in plan['sources']],'deployment':ref(P/'deployment-manifest.json'),'bundle':ref(P/'deployment.tar.gz'),'no_F1_quality_or_F0_gold_model_GPU':True})
print(json.dumps({'decision':ref(P/'conditional-decision.json'),'plan':ref(P/'plan.pending.json'),'CPU':CPU,'manifest':ref(P/'preparation-manifest.json'),'bundle':ref(P/'deployment.tar.gz')}))
