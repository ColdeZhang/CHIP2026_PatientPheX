#!/usr/bin/env python3
"""Strict final-channel training-only fit; all12 predictions freeze before targets."""
from pathlib import Path
import argparse,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def strict_final(text,codes,patients,hit_token_limit):
 if hit_token_limit:return None,'token_limit'
 if '</think>' not in text:return None,'missing_think_close'
 final=text.split('</think>',1)[1].strip()
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate key')
   d[k]=v
  return d
 try:obj=json.loads(final,object_pairs_hook=pairs)
 except (ValueError,TypeError):return None,'invalid_complete_final_JSON'
 if not isinstance(obj,dict) or set(obj)!=set(codes):return None,'candidate_keys'
 if not all(isinstance(v,list) and all(isinstance(x,str) and x in patients for x in v) and len(v)==len(set(v)) for v in obj.values()):return None,'patient_values'
 return obj,None

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory
 plan=json.loads((b/'run-plan.json').read_text());assert sha(__file__)==plan['evaluator_sha256'];frozen=json.loads((b/'predictions-freeze.json').read_text())
 assert frozen['status']=='complete_thinking12_training_predictions_frozen_before_fit_score' and frozen['rows']==12
 assert frozen['plan_sha256']==sha(b/'run-plan.json') and frozen['raw_sha256']==sha(b/'epoch2.responses.jsonl') and frozen['manifest_sha256']==sha(b/'epoch2.responses.manifest.json')
 assert not (b/'fit-result.json').exists()
 refs=plan['local_references']
 for key in ['prompts','targets','old_false_fit','old_false_raw','old_false_freeze']:
  ref=refs[key];assert sha(ref['path'])==ref['sha256']
 prompts={r['row_id']:r for r in map(json.loads,Path(refs['prompts']['path']).read_text().splitlines())}
 targets={r['row_id']:json.loads(r['answer']) for r in map(json.loads,Path(refs['targets']['path']).read_text().splitlines())}
 raw={r['row_id']:r for r in map(json.loads,(b/'epoch2.responses.jsonl').read_text().splitlines())};assert set(prompts)==set(targets)==set(raw)
 rows=[];tp=fp=fn=valid=exact=closed=0
 for rid,prompt in prompts.items():
  r=raw[rid];obj,reason=strict_final(r['response'],set(prompt['code_map']),set(prompt['patient_ids']),r['hit_token_limit']);ok=obj is not None
  valid+=ok;closed+='</think>' in r['response'];pred={(cid,pid) for cid,vs in (obj or {}).items() for pid in vs};gold={(cid,pid) for cid,vs in targets[rid].items() for pid in vs}
  t=len(pred&gold);f=len(pred-gold);n=len(gold-pred);tp+=t;fp+=f;fn+=n;exact+=ok and pred==gold
  rows.append(dict(row_id=rid,strict_final=ok,reason=reason,think_closed='</think>' in r['response'],exact_assignment=ok and pred==gold,tp=t,fp=f,fn=n,hit_token_limit=r['hit_token_limit'],output_tokens=r['output_tokens']))
 old=json.loads(Path(refs['old_false_fit']['path']).read_text());meta=json.loads((b/'epoch2.responses.manifest.json').read_text());f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 result=dict(status='complete_training_only_comparison_not_publicScore_or_generalization',plan_sha256=sha(b/'run-plan.json'),source_sha256=sha(__file__),freeze_sha256=sha(b/'predictions-freeze.json'),raw_sha256=sha(b/'epoch2.responses.jsonl'),old_false_fit_sha256=refs['old_false_fit']['sha256'],rows=12,strict_final_rows=valid,closed_think_rows=closed,exact_assignment_rows=exact,exact_assignment_rate=exact/12,relation=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=f1),delta_vs_frozen_false=dict(f1=f1-old['relation']['f1'],exact_rows=exact-old['exact_assignment_rows'],tp=tp-old['relation']['tp'],fp=fp-old['relation']['fp'],fn=fn-old['relation']['fn']),elapsed_seconds=meta['elapsed_seconds'],generation_seconds=sum(r['batch_seconds'] for r in raw.values()),output_tokens=sum(r['output_tokens'] for r in raw.values()),hit_token_limits=sum(r['hit_token_limit'] for r in raw.values()),row_results=rows,invalid_predictions_count_as_empty_without_fallback=True,targets_not_uploaded=True,no_originaldev_or_outergold=True,no_new_training=True)
 with (b/'fit-result.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='row_results'}))
if __name__=='__main__':main()
