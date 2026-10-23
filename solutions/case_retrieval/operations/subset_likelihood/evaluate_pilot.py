#!/usr/bin/env python3
"""Prewritten finite-set diagnostic, only after all98 scores frozen; no threshold search."""
from pathlib import Path
import argparse,hashlib,json,math,datetime,gzip
import numpy as np
from sklearn.metrics import roc_auc_score
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):
 with p.open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lse(x):
 m=max(x);return m+math.log(sum(math.exp(v-m) for v in x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory.resolve();plan=json.loads((b/'run-plan.json').read_text());assert sha(__file__)==plan['evaluator_sha256'];f=json.loads((b/'scores-freeze.json').read_text());assert f['plan_sha256']==sha(b/'run-plan.json') and f['raw_sha256']==sha(b/'scores.jsonl')
 raw=[json.loads(l) for l in (b/'scores.jsonl').read_text().splitlines()];encoded=json.loads(gzip.decompress((b/'selected98-encoded.json.gz').read_bytes()));by={(x['row_id'],x['cid'],x['mask']):x for x in encoded};assert len(raw)==len(by)==98
 for x in raw:
  e=by[(x['row_id'],x['cid'],x['mask'])];assert x['score_label_ids']==e['score_labels'] and x['input_ids_sha256']==hashlib.sha256(json.dumps(e['input_ids']).encode()).hexdigest()
  assert len(x['token_logprobs'])==x['logits_to_keep'] and abs(sum(x['token_logprobs'])-x['sum_logprob'])<1e-4 and all(math.isfinite(v) and v<=0 for v in x['token_logprobs'])
 selected=json.loads((b/'selected-inputs.json').read_text());greedy={r['row_id']:r['predicted_answer'] for r in json.loads((b/'selected-greedy-context.json').read_text())};probability=[];sets=[]
 for item in selected:
  p=item['prompt'];cid=item['selected_cid'];group=sorted([x for x in raw if x['row_id']==p['row_id'] and x['cid']==cid],key=lambda x:x['mask']);assert {x['mask'] for x in group}==set(range(1<<len(p['patient_ids'])));z=lse([x['sum_logprob'] for x in group]);ps=[math.exp(x['sum_logprob']-z) for x in group];assert abs(sum(ps)-1)<1e-10
  best=max(group,key=lambda x:(x['sum_logprob'],-x['mask']));sets.append(dict(row_id=p['row_id'],cid=cid,MAP_subset=best['subset'],old_greedy_subset=greedy[p['row_id']][cid],raw_empty_probability=ps[0],MAP_matches_greedy=set(best['subset'])==set(greedy[p['row_id']][cid]),restricted_normalizer_logmass=z))
  for pid in p['patient_ids']:
   yes=lse([x['sum_logprob'] for x in group if pid in x['subset']]);no=lse([x['sum_logprob'] for x in group if pid not in x['subset']]);m=math.exp(yes-z);assert 0<=m<=1;probability.append(dict(row_id=p['row_id'],cid=cid,patient_id=pid,marginal=m,log_odds=yes-no,predicted=m>=.5,old_greedy=pid in greedy[p['row_id']][cid]))
 save(b/'conditional-probabilities-frozen.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scores_freeze=ref(b/'scores-freeze.json'),patients=probability,sets=sets,not_unconditional_posterior=True))
 # First read of the already frozen local F1 labels is below complete score/probability freeze.
 save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),score_freeze=ref(b/'scores-freeze.json'),probability_freeze=ref(b/'conditional-probabilities-frozen.json'),fixed_threshold=.5))
 labelref=plan['local_labels_after_freeze'];assert sha(labelref['path'])==labelref['sha256'];labels={x['row_id']:json.loads(x['answer']) for x in map(json.loads,Path(labelref['path']).read_text().splitlines())};y=np.array([x['patient_id'] in labels[x['row_id']][x['cid']] for x in probability],dtype=int);prob=np.array([x['marginal'] for x in probability]);old=np.array([x['old_greedy'] for x in probability],dtype=int)
 def counts(pred):
  tp=int(sum((pred==1)&(y==1)));fp=int(sum((pred==1)&(y==0)));fn=int(sum((pred==0)&(y==1)));return dict(tp=tp,fp=fp,fn=fn,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 result=dict(status='complete_predefined12CID26pair_mechanism_pilot_not_Score',scope='SFT32 excluded F1; input-only12x1CID, rawHPO training label convention; tiny development mechanism pilot, not generalized calibration or publicScore',score_freeze=ref(b/'scores-freeze.json'),probability_freeze=ref(b/'conditional-probabilities-frozen.json'),rows=12,patient_pairs=len(y),positive_labels=int(y.sum()),old_greedy=counts(old),fixed_marginal_point5=counts((prob>=.5).astype(int)),Brier_old=float(np.mean((old-y)**2)),Brier_marginal=float(np.mean((prob-y)**2)),ROC_AUC_marginal=float(roc_auc_score(y,prob)) if len(set(y))==2 else None,ROC_AUC_old=float(roc_auc_score(y,old)) if len(set(y))==2 else None,MAP_vs_greedy_different=sum(not x['MAP_matches_greedy'] for x in sets),threshold_search=False,no_new_predictions_on_dev=True,no_training=True,complete=ref(b/'complete.json'))
 save(b/'pilot-summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
