"""Bind actual complete stages to unchanged G32 strict/semantic interpretation.

No prediction, gold, model or quality execution here. Stage identities are
append-only, after the predecessor is complete; policy is fixed before results.
"""
import argparse,json,os
from pathlib import Path
from common import OLD,RUN,GOLD,BASE_SHA,BASE_SCORE,POLICY_SHA,checked,guard,read,ref,save,source_contract,load
S=Path(__file__).resolve().parent;Q=S.parent
REFERENCE=Path('/workspace/patientphex-worktrees/explore-grounded-sampled-consensus/solutions/grounded_sampled_consensus/postprocess/quality-plan.json')
PINS_PATH=S/'fixed-references.json'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','finalize','quality']);a=ap.parse_args();guard()
 fixed=read(PINS_PATH)
 assert fixed['gold_metadata_only']==GOLD and fixed['baseline_score']==BASE_SCORE
 for item in fixed['inputs'].values():checked(item)
 assert fixed['inputs']['baseline']['sha256']==BASE_SHA
 refs=fixed['inputs'];plans=RUN/'postprocess-plans';plans.mkdir(exist_ok=True)
 if a.stage=='extract':
  d=RUN/'inference-F1-v1';c=read(d/'collection.json')
  assert c['all_actual_SHA_verified'] and c['inference_reported_success']
  for item in c['files'].values():checked(item)
  p=read(Q/'inference/infer-plan.json');files=c['files']
  assert c['wrapper']['complete39_actual_members_verified']
  inputs=dict(complete=files['output/complete.json'],responses=files['output/responses.jsonl'],wrapper=files['wrapper-complete.json'],
   F1_input=refs['F1_input'],prompts=ref(RUN/'F1-input-preparation-recovery-v1/prompts.jsonl'))
  sources={name:ref(path) for name,path in [('extract_quotes.py',OLD/'inference/extract_quotes.py'),('decode.py',OLD/'inference/decode.py'),('schema.py',OLD/'schema.py'),('run.py',Q/'inference/run.py')]}
  assert sources['run.py']['sha256']==p['assets']['runner']['sha256']
  plan=dict(schema='grounded-extract-complete39-v1',sources=sources,inputs=inputs,gold_metadata_only=GOLD,
   remote_output=p['output_dir'],local_output=str(d/'terminal/output'),output_dir=str(RUN/'F1-extraction-v1'),
   extra_lineage=dict(collection=ref(d/'collection.json'),training=ref(RUN/'formal-v1/collection.json'),fixed=ref(PINS_PATH),complete39_onepass=True))
  path=plans/'extract-plan.json';save(path,plan)
  print(json.dumps(dict(plan=ref(path),command=['env','PYTHONPATH='+str(OLD.parents[1]),'python',str(OLD/'inference/extract_quotes.py'),'--plan',str(path)],next_prepare=['python',str(OLD/'normalization/held_prepare.py'),'--extraction-manifest',str(RUN/'F1-extraction-v1/manifest.json'),'--output-dir',str(RUN/'F1-normalization-preparation-v1')])));return
 if a.stage=='finalize':
  prep=read(RUN/'F1-normalization-preparation-v1/manifest.json');sources={n:ref(OLD/'normalization'/n) for n in ['held_finalize.py','held_prepare.py','strict.py','semantic.py','policy_v1.json']}
  plan=dict(schema='grounded-held-normalization-finalize-v1',sources=sources,preparation=ref(RUN/'F1-normalization-preparation-v1/manifest.json'),output_dir=str(RUN/'F1-normalized-v1'))
  if prep['necessary_public_query_count']:
   cdir=RUN/'normalization-F1-v1';c=read(cdir/'collection.json');assert c['all_actual_SHA_verified'] and c['query_reported_success']
   plan.update(query_collection=ref(cdir/'collection.json'),query_complete=c['files']['output/complete.json'],query_scores=c['files']['output/query-scores.json'])
  else:plan.update(query_collection=None,query_complete=None,query_scores=None)
  path=plans/'finalize-plan.json';save(path,plan);print(json.dumps(dict(plan=ref(path),command=['python',str(OLD/'normalization/held_finalize.py'),'--plan',str(path)])));return
 norm=RUN/'F1-normalized-v1/manifest.json';n=read(norm);assert n['no_quality'] and n['policy_sha256']==POLICY_SHA
 inputs={**refs, 'normalization':ref(norm),'collection':ref(RUN/'inference-F1-v1/collection.json'),'prompts':ref(RUN/'F1-input-preparation-recovery-v1/prompts.jsonl')}
 plan=dict(schema='quote-completion-postprocess-v1',stage='quality',state='BOUND_COMPLETE_ACTUAL_ARTIFACTS',sources=source_contract(),
  gold_metadata_only=GOLD,inputs=inputs,output_directory=str(RUN/'quality-F1-v1'),formula='B4 UNION all completion relations',
  baseline_score=BASE_SCORE,expected_baseline_relations=524,new_v2_max=1,old_v2=0,score_budget_seconds=30,term_grace_seconds=1,
  keep_rule='Score strictly above B4, macro disclosed without additional veto',no_added_means_baseline_byteexact_and_zero_v2=True,
  no_single_source_reselection=True,prelocked_references=ref(PINS_PATH),inference_policy=ref(Q/'inference/policy.json'))
 path=plans/'quality-plan.json';save(path,plan);print(json.dumps(ref(path)))
if __name__=='__main__':main()
