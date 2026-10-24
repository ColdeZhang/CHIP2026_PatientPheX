"""Freeze excluding-F0 source triples and actual full47 CPU inputs; no gold/score."""
import argparse,copy,gzip,hashlib,json,os,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;C=S.parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/copy-grounded-decode-v1/F0-v1';P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/copy-grounded-decode-v1/F0-v1/cpu'
sys.path.insert(0,str(C));from local_common import ref,save
sys.path.insert(0,'/workspace/patientphex-worktrees/explore-grounded-generation');from solutions.grounded_generation.schema import locate,grounded,units
SG=Path('/workspace/patientphex-worktrees/explore-grounded-sampled-consensus/solutions/grounded_sampled_consensus/f0')
def read(p):return json.loads(Path(p).read_text())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gliner-entities',required=True);ap.add_argument('--proposer-manifest',required=True);a=ap.parse_args()
 def deny(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and '.gold.' in os.fsdecode(args[0]):raise PermissionError('No held gold during pool preparation')
 sys.addaudithook(deny)
 old=read(SG/'infer-plan.json');sources=old['payload_sources'];inputs=Path('/workspace/patientphex-data/round2-inner4/fold-0/validation16.input.jsonl');docs=[json.loads(x) for x in inputs.read_text().splitlines()];by={str(d['pmc_id']):d for d in docs};assert len(by)==16 and sum(len(d['patient']) for d in docs)==47 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs)
 strong=R/'research/lightgbm-selector-readiness/formal-control/fold0/lightgbm.nested.jsonl';assert ref(strong)['sha256']=='79d3568e60de2ab9424c0ece5e3d6b1f95bb1a32da388bdc76ca29ec78d222d5'
 strongrows=[json.loads(x) for x in strong.read_text().splitlines()];gliner=read(a.gliner_entities);assert set(gliner)==set(by)=={str(d['pmc_id']) for d in strongrows}
 spans={did:set() for did in by}
 for did,entities in [(str(d['pmc_id']),d['entities']) for d in strongrows]+list(gliner.items()):
  for e in entities:
   assert e['length']==len(e['text']);triple=locate(by[did],e['text'],e['offset']);assert triple is not None
   anchor=grounded(triple,{u['unit_id']:u for u in units(by[did])});assert anchor['offset']==e['offset'] and anchor['length']==e['length'];spans[did].add(tuple(triple))
 assert all(spans.values())
 for k in ['held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json']:assert ref(sources[k]['path'])==sources[k]
 prompts=[json.loads(x) for x in Path(sources['held-prompts.jsonl']['path']).read_text().splitlines()];ids=read_gzip(sources['held-actual-ids.json.gz']['path']);assert len(prompts)==len(ids)==47
 assert sum(len(x['input_ids']) for x in ids)==289420 and max(len(x['input_ids']) for x in ids)==9935
 pools=[]
 for row in prompts:
  did=str(row['pmc_id']);assert row['units']==units(by[did]) and set(row)=={'row_id','pmc_id','target_patient_id','messages','units'}
  pools.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],target_patient_id=row['target_patient_id'],triples=sorted(spans[did])))
 out=D/'prepared';out.mkdir(parents=True,exist_ok=False);q=out/'candidates.jsonl'
 with q.open('xb') as f:f.write(b''.join((json.dumps(x,ensure_ascii=False)+'\n').encode() for x in pools));f.flush();os.fsync(f.fileno())
 bdir=R/'research/four-source-fusion-20261024/F0-v1';metrics=read(bdir/'metrics.json');bf=read(bdir/'freeze.json');assert metrics['metrics']['Score']==.7281956289113009 and bf['final_relations']==343
 decision=dict(schema='copy-grounded-F0-fixed-after-positive-F1-v2',F1_result=ref(C/'evidence/F1-fallback-v2-result.json'),F1_abstention_policy=ref(C/'fallback_v2/decision.json'),formula='B4_F0 UNION (constrained_G_valid_rows INTERSECT (original_pure_G UNION S UNION Q UNION C))',model_adapter_sha256='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba',generation=dict(do_sample=False,num_beams=1,use_cache=True,max_new_tokens=1600),rows=47,documents=16,entities=947,baseline_relations=343,baseline_score=.7281956289113009,baseline=ref(bdir/'prediction.jsonl'),baseline_metrics=ref(bdir/'metrics.json'),baseline_freeze=ref(bdir/'freeze.json'),sources=dict(input=ref(inputs),strong=ref(strong),gliner_accepted=ref(a.gliner_entities),gliner_proposer_manifest=ref(a.proposer_manifest),prompts=sources['held-prompts.jsonl'],encoded=sources['held-actual-ids.json.gz'],audit=sources['held-audit.json'],pool=ref(q)),pool_counts=dict(strong=sum(len(x['entities']) for x in strongrows),gliner=sum(map(len,gliner.values())),union_exact_triples=sum(map(len,spans.values())),document_unique_quotes=sum(len({t[1] for t in v}) for v in spans.values())),F0_all_pool_ancestors_must_exclude_F0=True,whole_row_abstention_on_cap_or_nonEOS_or_original_strict_failure=True,no_prefix_salvage=True,all_raw_before_parse=True,full47_required=True,inference_budget_seconds=2400,grace_seconds=30,cost_proxy_seconds=725.9330830061808*1.4216*1.5+180,no_GPU_probe=True,CPU_budget_seconds=120,CPU_grace_seconds=5,necessary_FP16_Sap_budget_seconds=180,Sap_grace_seconds=30,new_v2_max=1,old_v2=0,byteexact_inherit_zero=True,no_single_arm_or_valid_subset=True,no_gold_opened=True)
 save(S/'decision.json',decision)
 p=read(C/'cpu-plan.json');p.update(id='E-20261024-COPY-GROUNDED-001-F0-CPU',schema='copy-grounded-actual-CPU-F0-v1',expected_rows=47,expected_status='PASS_ACTUAL_HF_CPU_ORDERED_TRIE47',output_dir=ROOT+'/actual',argv=[p['python'],ROOT+'/cpu_runtime.py','--plan',ROOT+'/cpu-plan.json'],operations_source=ref(S/'cpu_remote.py'),collection_source=ref(S/'cpu_collect.py'),input_sources=dict(F0_decision=ref(S/'decision.json'),original_F0_sampler_plan=ref(SG/'infer-plan.json')))
 for key in ['prompts','held_ids','held_audit']:p['assets'][key]=old['assets'][key]
 payload={name:ref((S if name in ['cpu_runtime.py','job_driver.py'] else C)/name) for name in ['cpu_runtime.py','job_driver.py','processors.py','triple_trie.py','raw_store.py']};payload['candidates.jsonl']=ref(q);p['payload_sources']=payload;p['files']={ROOT+'/'+n:ident(v) for n,v in payload.items()}
 for key,name in [('runner','cpu_runtime.py'),('driver','job_driver.py'),('processors','processors.py'),('trie','triple_trie.py'),('raw_store','raw_store.py'),('candidates','candidates.jsonl')]:p['assets'][key]=dict(path=ROOT+'/'+name,**ident(payload[name]))
 save(S/'cpu-plan.json',p);save(out/'complete.json',dict(source=ref(__file__),decision=ref(S/'decision.json'),CPU_plan=ref(S/'cpu-plan.json'),pool=ref(q),no_gold_model_or_quality=True));print(json.dumps(ref(out/'complete.json')))
def read_gzip(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
if __name__=='__main__':main()
