"""One metadata-only final F0 entry; never run models, parser, normalizer or quality."""
import json,sys
from pathlib import Path
S=Path(__file__).resolve().parents[1]
D=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1')
sys.path.insert(0,str(S/'postprocess_F0'))
from common import checked,guard,read,ref,save

def main():
 guard();q=D/'quality-F0-v1';r=read(q/'result.json');f=read(q/'freeze.json')
 assert r['freeze']==ref(q/'freeze.json') and r['old_v2']==0 and r['new_v2'] in [0,1]
 assert f['formula']=='B4 UNION all completion relations' and f['complete_all47']
 assert f['nonphenotype_bytes_exact'] and f['baseline_relation_lists_prefix_exact']
 checked(f['prediction']);assert (f['documents'],f['patients'],f['entities'],f['baseline_relations'])==(16,47,947,343)
 c=read(D/'inference-F0-A30-v1/collection.json');g=read(D/'inference-F0-A30-v1/terminal/output/complete.json')
 assert c['all_actual_SHA_verified'] and c['inference_reported_success'] and g['rows']==47
 prep=read(D/'F0-input-preparation-v1/manifest.json');assert prep['known_triples']==204 and prep['ancestry']['disjoint']
 n=read(D/'F0-normalized-v1/manifest.json');x=read(D/'F0-extraction-v1/manifest.json')
 assert n['patients']==47 and n['no_fit'] and n['no_quality'] and x['complete_original47']
 assert x['cap_rows']==g['cap_rows'] and x['EOS_rows']==g['EOS_rows']
 names=['F0-input-preparation-v1/manifest.json','inference-F0-v1/started.json','inference-F0-v1/managed-stderr-initial.json','inference-F0-v1/fresh-resource-gate-failure.json','inference-F0-A30-v1/package.json','inference-F0-A30-v1/collection.json','F0-extraction-v1/manifest.json','F0-normalization-preparation-v1/manifest.json','F0-normalized-v1/manifest.json','postprocess-F0-plans/extract-plan.json','postprocess-F0-plans/finalize-plan.json','postprocess-F0-plans/quality-plan.json']
 names+=['quality-F0-v1/'+v for v in ['freeze.json','result.json','evaluation-launch.json','evaluation-terminal.json','prediction.jsonl','all-completion-keys.json','all-and-additions.json','consumption.json']]
 if r['new_v2']:
  assert r['status']=='COMPLETE_ONE_FIXED_ENDPOINT' and r['Task1_unchanged'];names+=['quality-F0-v1/metrics.json','quality-F0-v1/evaluation-intent.json']
 else:assert r['status']=='NO_ADDED_KEYS_BASE_BYTEEXACT_ZERO_V2' and not r['keep'] and r['no_gold_read'] and f['added']==0
 ns=0
 if n['necessary_public_queries']:
  nc=read(D/'normalization-F0-v1/collection.json');assert nc['all_actual_SHA_verified'] and nc['query_reported_success'];names+=['normalization-F0-v1/collection.json'];ns=nc['wrapper']['elapsed_seconds']
 sources={}
 for folder in ['f0','f0_a30','postprocess_F0','normalization_F0']:
  for p in sorted((S/folder).iterdir()):
   if p.is_file() and p.suffix in ['.py','.json']:sources[str(p.relative_to(S))]=ref(p)
 sources['reporting/f0_entry.py']=ref(__file__)
 F1=read(D/'quality-F1-v1/result.json');assert F1['keep']
 review=Path('/workspace/patientphex-runs/round2-20261023/validation/quote-set-completion-F1-final-review/review.json')
 entry=dict(schema='quote-set-completion-F0-review-entry-v1',status='BOTH_FIXED_FOLDS_POSITIVE_CANDIDATE' if r['keep'] else 'F0_NONPOSITIVE_METHOD_CLOSED',refs={name:ref(D/name) for name in names},source_refs=sources,result=r,prior_F1_entry=ref(D/'F1-complete-review-entry.json'),prior_F1_review=ref(review),prior_F1_persistent=ref(D/'endpoint-F1-supplement-v1/persistent-receipt.json'),same_trained_adapter='cd977a24255667bfc3c1610ae261b44c9e124ca586998d9edaed8cbc5b36c04f',same_train32_excludesF0=prep['ancestry'],generation=dict(rows=47,EOS=g['EOS_rows'],cap=g['cap_rows'],native_seconds=g['seconds'],wrapper_seconds=c['wrapper']['elapsed_seconds'],max_allocated_bytes=g['max_gpu_allocated_bytes'],one_completion_pass=True,raw_preserved=True),normalization=dict(anchors=n['records'],necessary_queries=n['necessary_public_queries'],counts=n['status_counts'],policy=n['policy_sha256'],query_seconds=ns),endpoint=dict(baseline_score=.7281956289113009,formula=f['formula'],source_relations=f['source_relations'],added=f['added'],final_relations=f['final_relations'],new_v2=r['new_v2'],old_v2=0),hardware_migration=read(S/'f0_a30/infer-plan.json')['hardware_migration'],full64_training_authorized=False,originaldev_gold_not_read=True,no_official_actions=True)
 target=D/'F0-complete-review-entry.json';save(target,entry)
 save(S/'evidence/F0-result.json',dict(entry=ref(target),result=ref(q/'result.json'),prediction=f['prediction'],freeze=ref(q/'freeze.json'),both_folds_keep=r['keep'],independent_review='pending',originaldev_best_unchanged=True))
 print(json.dumps(ref(target)))
if __name__=='__main__':main()
