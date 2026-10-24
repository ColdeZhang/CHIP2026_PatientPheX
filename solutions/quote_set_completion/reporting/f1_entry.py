"""Freeze one completed training/held39/normalization/quality review entry.

Metadata only: no GPU, decoding, normalization, gold or evaluator execution.
Works for positive, negative and byteexact-zero-new quality endpoints.
"""
import hashlib,json,os,sys
from pathlib import Path
S=Path(__file__).resolve().parents[1]
D=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1')
sys.path.insert(0,str(S/'postprocess'))
from common import checked,guard,read,ref,save

def main():
 guard();q=D/'quality-F1-v1';result=read(q/'result.json');freeze=read(q/'freeze.json')
 assert result['freeze']==ref(q/'freeze.json') and result['old_v2']==0 and result['new_v2'] in [0,1]
 assert freeze['formula']=='B4 UNION all completion relations' and freeze['complete_all39']
 assert freeze['nonphenotype_bytes_exact'] and freeze['baseline_relation_lists_prefix_exact']
 checked(freeze['prediction']);assert freeze['documents']==16 and freeze['patients']==39 and freeze['entities']==1189
 train=read(D/'formal-v1/collection.json');ack=read(D/'formal-v1/epoch-001-export-ack.json')
 assert train['all_actual_SHA_verified'] and train['formal_reported_success']
 assert ack['status']=='persistent_checkpoint_verified' and ack['completed_epochs']==1 and ack['global_step']==33
 infer=read(D/'inference-F1-v1/collection.json');assert infer['all_actual_SHA_verified'] and infer['inference_reported_success']
 generation=read(D/'inference-F1-v1/terminal/output/complete.json');assert generation['rows']==39
 normalization=read(D/'F1-normalized-v1/manifest.json');extraction=read(D/'F1-extraction-v1/manifest.json')
 assert normalization['documents']==16 and normalization['patients']==39 and normalization['no_fit'] and normalization['no_quality']
 assert extraction['complete_original39'] and extraction['cap_rows']==generation['cap_rows']
 names=['train-only-cpu-v2-recovery/manifest.json','native-cpu-v2/collection.json','probe-v2/collection.json',
  'probe-path-recovery-v1/collection.json','probe-two-shape-v1/collection.json','formal-cost-guard.json',
  'formal-v1/preparation-complete.json','formal-v1/campaign.json','formal-v1/campaign-complete.json',
  'formal-v1/controller-plan.json','formal-v1/collection.json','formal-v1/epoch-001-export-ack.json',
  'formal-v1/epoch-001-state.json','F1-input-preparation-recovery-v1/manifest.json',
  'inference-F1-v1/collection.json','F1-extraction-v1/manifest.json',
  'F1-normalization-preparation-v1/manifest.json','F1-normalized-v1/manifest.json',
  'postprocess-plans/extract-plan.json','postprocess-plans/finalize-plan.json','postprocess-plans/quality-plan.json']
 names+=['quality-F1-v1/'+name for name in ['freeze.json','result.json','evaluation-launch.json','evaluation-terminal.json','prediction.jsonl','all-completion-keys.json','all-and-additions.json','consumption.json']]
 if result['new_v2']:
  assert result['status']=='COMPLETE_ONE_FIXED_ENDPOINT' and result['Task1_unchanged']
  names+=['quality-F1-v1/metrics.json','quality-F1-v1/evaluation-intent.json']
 else:
  assert result['status']=='NO_ADDED_KEYS_BASE_BYTEEXACT_ZERO_V2' and not result['keep'] and result['no_gold_read']
  assert freeze['added']==0 and freeze['no_added_prediction_byteexact']
 norm_query_seconds=0
 if normalization['necessary_public_queries']:
  nc=read(D/'normalization-F1-v1/collection.json');assert nc['all_actual_SHA_verified'] and nc['query_reported_success']
  names+=['normalization-F1-v1/collection.json'];norm_query_seconds=nc['wrapper']['elapsed_seconds']
 sources={str(p.relative_to(S)):ref(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'} and 'evidence' not in p.parts}
 actual_train=read(D/'formal-v1/terminal/output/training_complete.json');state=read(D/'formal-v1/epoch-001-state.json')
 entry=dict(schema='quote-set-completion-v2-F1-review-entry-v1',status='POSITIVE_F1_CANDIDATE' if result['keep'] else 'NEGATIVE_OR_BYTEEXACT_F1_CLOSED',
  refs={name:ref(D/name) for name in names},source_refs=sources,result=result,
  training=dict(rows=261,documents=32,patients=87,epochs=1,updates=33,warmup=2,tail=5,warm_d52=True,fresh_optimizer_RNG=True,
    formal_program_seconds=actual_train['elapsed_seconds'],wrapper_seconds=state['wrapper']['elapsed_seconds'],
    source_SHM_not_persistent=True,persistent_ACK=ref(D/'formal-v1/epoch-001-export-ack.json'),persistent_state=ack['persistent_directory'],
    actual_external_checkpoints=train['actual_external_checkpoints'],export_seconds=ack['seconds'],max_allocated_bytes=actual_train['max_gpu_allocated_bytes']),
  generation=dict(rows=39,documents=16,EOS=generation['EOS_rows'],cap=generation['cap_rows'],wrapper_seconds=infer['wrapper']['elapsed_seconds'],native_seconds=generation['seconds'],max_allocated_bytes=generation['max_gpu_allocated_bytes'],one_completion_pass=True,all_raw_preserved=True),
  normalization=dict(anchors=normalization['records'],necessary_queries=normalization['necessary_public_queries'],status_counts=normalization['status_counts'],query_wrapper_seconds=norm_query_seconds,policy=normalization['policy_sha256']),
  fixed_endpoint=dict(baseline_score=.701325377212172,formula=freeze['formula'],new_v2=result['new_v2'],old_v2=0,added=freeze['added'],full_relations=freeze['final_relations']),
  no_official_action=True,originaldev_gold_read=False,no_partial_prefix_or_retry=True,
  limitations=['F1 is an internal development fold, not originaldev best','Training known is a true target subset; deployment known may contain false positives or synonyms','Targets are weak representative phenotype phrases, not human-verified patient evidence','Warm-start target/LR change is a joint method; no claim of a single causal numeric factor'])
 target=D/'F1-complete-review-entry.json';save(target,entry)
 save(S/'evidence/F1-result.json',dict(entry=ref(target),result=ref(q/'result.json'),prediction=freeze['prediction'],freeze=ref(q/'freeze.json'),keep=result['keep'],new_v2=result['new_v2'],old_v2=0,independent_review='pending',originaldev_best_unchanged=True))
 print(json.dumps(ref(target)))
if __name__=='__main__':main()
