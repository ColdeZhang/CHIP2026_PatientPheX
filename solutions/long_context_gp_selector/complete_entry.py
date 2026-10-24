"""Once-only final metadata from completed original artifacts; no model/fit/gold/v2."""
import json
from pathlib import Path
from cpu_common import ref,write_json,read_jsonl
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1')

def main():
 assert not (D/'final-review-entry.json').exists()
 c=D/'collected';q=D/'quality';end=json.load(open(c/'job/ended.json'));nn=json.load(open(c/'output/complete.json'));result=json.load(open(q/'result.json'));plan=json.load(open(S/'quality-plan.json'))
 assert end['returncode']==0 and not end['timed_out'] and nn['status']=='COMPLETE_THREE_FITS_FOUR_FULL_SCORE_PASSES'
 assert nn['model_fits']==3 and nn['score_passes']==4 and nn['actual_adam_updates']==896
 assert result['fit_count']==1 and result['new_v2_calls']==1 and result['old_v2_calls']==0
 jobs=[];total=0;states=[]
 for job in plan['jobs']:
  p=c/'output'/job['name'];r=json.load(open(p/'complete.json'));assert r['target_spans']==job['target_spans'] and r['no_score_floor'] and not r['held_gold_read']
  assert r['output']['sha256']==ref(p/'target-scores.jsonl')['sha256'];assert set(r['train_ids'])==set(job['train_ids']) and set(r['held_ids'])==set(job['input_ids'])
  assert set(r['train_ids']).isdisjoint(set(r['held_ids'])|set(plan['outer1_held_ids']))
  expected=0 if job['name']=='inner0' else 8*job['training_documents'];assert r['actual_adam_updates']==expected;total+=expected
  if expected:
   ack=json.load(open(p/'final-state-ACK.json'));assert ack['status']=='FULL_STATE_FSYNC_REOPEN_RESTORE_PASS' and ack['actual_adam_updates']==expected
   assert ack['checkpoint']==r['model_checkpoint'];states.append(dict(mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c',checkpoint=ack['checkpoint'],ACK=ref(p/'final-state-ACK.json')))
  jobs.append(dict(name=job['name'],complete=ref(p/'complete.json'),scores=ref(p/'target-scores.jsonl')))
 assert total==896 and len(states)==3
 base=read_jsonl(plan['strong_baseline']['path']);pred=read_jsonl(q/'prediction.jsonl');assert [x['pmc_id'] for x in base]==[x['pmc_id'] for x in pred]
 keys=lambda rows:{(x['pmc_id'],int(e['offset']),int(e['length']),e['identifier']) for x in rows for e in x['entities']}
 before,after=keys(base),keys(pred)
 for a,b in zip(base,pred):assert {k:v for k,v in a.items() if k!='entities'}=={k:v for k,v in b.items() if k!='entities'}
 obj=dict(status='F1_CANDIDATE_ONLY' if result['retain'] else 'CLOSED_FIXED_F1_NEGATIVE_OR_EQUAL_NO_SWEEP',source=ref(__file__),source_plan=ref(S/'runtime/formal-plan.json'),quality_plan=ref(S/'quality-plan.json'),source_checkpoint='See branch commit; actual source bytes bound by plan and collected package',collection=ref(D/'collection-complete.json'),terminal_package=ref(D/'terminal-package.json'),terminal=ref(c/'job/ended.json'),NN_complete=ref(c/'output/complete.json'),cpu=ref(c/'cpu-complete.json'),source_models=jobs,new_fullstate_locators=states,assets_copy=ref(D.parent/'asset-copy-v1/copy-complete.json'),source_HTTP_closed=ref(D.parent/'asset-copy-v1/source-closure.json'),data_preparation=ref(D/'local-preparation.json'),mapping_contract=ref(D/'mapping-contract.json'),quality={n:ref(q/n) for n in ['result.json','complete-freeze.json','metrics.json','prediction.jsonl','model-vectorizer.pkl','format.json','evaluation-intent.json']},baseline=plan['strong_baseline'],baseline_metrics=plan['strong_metrics'],metrics=result['metrics'],deltas=result['delta'],retain=result['retain'],counts=dict(train_documents=48,train_candidates=45403,input_documents=16,input_candidates=16595,old_features=101,new_features=103,baseline_entities=sum(len(x['entities']) for x in base),new_entities=sum(len(x['entities']) for x in pred),literal_entity_added=len(after-before),literal_entity_removed=len(before-after),other_objects_exact=True),cost=dict(NN_program_seconds=nn['seconds'],driver_seconds=end['elapsed_seconds'],CPU_fit_decode_one_v2_seconds=result['seconds'],peak_allocated_bytes=nn['peak_allocated_bytes'],peak_reserved_bytes=nn['peak_reserved_bytes']),interpretation='New full-context GP source, same previously-tested two-feature integration. Faithful strong48 ancestry; F0 labels may enter F1 training. Not originaldev best or blind evaluation.',automatic_F0=False,official_actions=0,new_model_calls_in_this_entry=0)
 write_json(D/'final-review-entry.json',obj);print(json.dumps(ref(D/'final-review-entry.json')))
if __name__=='__main__':main()
