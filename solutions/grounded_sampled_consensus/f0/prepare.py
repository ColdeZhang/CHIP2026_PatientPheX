"""Freeze the same train32 recipe on pure F0 47 inputs; never read F0 gold."""
import copy,gzip,hashlib,json,os
from pathlib import Path
T=Path(__file__).resolve().parent;S=T.parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1')
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/grounded-sampled-consensus-v1/native-F0-cpu-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 result=json.load(open(D/'quality-F1-v1/result.json'));assert result['keep'] and result['new_v2']==1 and result['old_v2']==0
 old=json.load(open(G/'f0/infer-plan.json'));v=old['payload_sources']
 rows=[json.loads(l) for l in Path(v['held-prompts.jsonl']['path']).read_text().splitlines()];ids=json.loads(gzip.decompress(Path(v['held-actual-ids.json.gz']['path']).read_bytes()));audit=json.load(open(v['held-audit.json']['path']))
 assert len(rows)==len(ids)==len(audit)==47 and len({x['pmc_id'] for x in rows})==16
 assert len({(x['pmc_id'],x['target_patient_id']) for x in rows})==47 and all('answer' not in x and 'labels' not in x for x in rows)
 assert [x['row_id'] for x in rows]==[x['row_id'] for x in ids]==[x['row_id'] for x in audit]
 for r,i,a in zip(rows,ids,audit):assert len(i['input_ids'])==a['input_tokens'] and hashlib.sha256(json.dumps(i['input_ids']).encode()).hexdigest()==a['input_ids_sha256']
 tokens=sum(len(x['input_ids']) for x in ids);maximum=max(len(x['input_ids']) for x in ids);assert tokens==289420 and maximum==9935
 p=json.load(open(S/'runtime/cpu-plan.json'));oldroot=p['output_dir'].rsplit('/',1)[0]
 def replace(x):
  if isinstance(x,str):return x.replace(oldroot,ROOT)
  if isinstance(x,list):return [replace(y) for y in x]
  if isinstance(x,dict):return {replace(k):replace(y) for k,y in x.items()}
  return x
 p=replace(p);host=json.load(open(S/'a30/infer-plan.json'))
 p.update(schema='grounded-F0-quote-sampling-native47-CPU-v1',held_rows=47,python=host['python'],model=host['model'],checkpoint_directory=host['checkpoint_directory'],operations_source=ref(T/'cpu_remote.py'))
 p['environment']['PYTHONPATH']=host['environment']['PYTHONPATH'];p['environment']['CUDA_VISIBLE_DEVICES']=''
 items={'native_cpu.py':ref(T/'native_cpu.py'),'cpu_driver.py':ref(T/'cpu_driver.py'),'sampling.py':ref(T/'sampling.py'),'held.input.jsonl':v['held-prompts.jsonl'],'held-audit.json':v['held-audit.json'],'held-local-ids.json.gz':v['held-actual-ids.json.gz'],'model-source-manifest.json':v['model-source-manifest.json'],'original-sampling-interface.json':ref(D/'native-cpu-v1/terminal/actual/sampling-interface.json')}
 for x in items.values():assert ref(x['path'])==x
 p['payload_sources']=items;p['files']={ROOT+'/'+k:ident(x) for k,x in items.items()};p['prior_sampling_interface']=ROOT+'/original-sampling-interface.json'
 p['argv']=[p['python'],'-u',ROOT+'/native_cpu.py','--plan',ROOT+'/cpu-plan.json','--output',ROOT+'/actual']
 p['input_sources']=dict(F1_result=ref(D/'quality-F1-v1/result.json'),original_F0_plan=ref(G/'f0/infer-plan.json'),original_F0_preparation=old['input_sources']['F0_preparation'],F0_ancestry=old['input_sources']['F0_ancestry'],original_F0_native_collection=old['input_sources']['native_CPU_collection'],original_F1_sampler_CPU=ref(D/'native-cpu-v1/collection.json'),same_A30_model=ref(S/'a30/infer-plan.json'))
 save(T/'cpu-plan.json',p)
 bdir=Path('/workspace/patientphex-runs/round2-20261023/research/four-source-fusion-20261024/F0-v1');bf=json.load(open(bdir/'freeze.json'));bm=json.load(open(bdir/'metrics.json'));assert bm['metrics']['Score']==.7281956289113009 and bf['final_relations']==343
 decision=dict(schema='grounded-sampling-F0-fixed-after-positive-F1-v1',F1_result=ref(D/'quality-F1-v1/result.json'),model_adapter_sha256='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba',input_refs={k:v[k] for k in ['held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json']},input_tokens_per_seed=tokens,max_input_tokens=maximum,patients47=47,documents=16,fresh_rows=94,seeds=[20261024,20261025],generation=host['generation'],presence_penalty=0,budget_seconds=3600,grace_seconds=30,gpu_uuid=host['gpu_uuid'],native_cpu_seconds=120,native_cpu_grace=5,prior_sampler_contract_reused_no_retesting=True,no_GPU_probe=True,
  cost_proxy=dict(F1_actual_seconds=1411.159027302172,input_token_ratio=tokens/203588,proxy_seconds=1411.159027302172*tokens/203588,with_1p5_margin_and_180=1411.159027302172*tokens/203588*1.5+180,not_a_guarantee=True),
  formula='F0 B4 UNION (normalized sample1 INTERSECT normalized sample2)',baseline_prediction=ref(bdir/'prediction.jsonl'),baseline_metrics=ref(bdir/'metrics.json'),baseline_freeze=ref(bdir/'freeze.json'),baseline_plan=bf['plan'],baseline_score=.7281956289113009,baseline_relations=343,entities=947,patients=47,new_v2_max=1,old_v2=0,byteexact_zero_additions_zero_v2=True,keep_rule='Both F1 and F0 total Score strictly exceed each fixed B4; macro disclosed, no extra veto',no_gold_read=True,no_other_arm_or_seed_scan=True,no_originaldev_or_official_action=True)
 save(T/'decision.json',decision);print(json.dumps(dict(cpu_plan=ref(T/'cpu-plan.json'),decision=ref(T/'decision.json'))))
if __name__=='__main__':main()
