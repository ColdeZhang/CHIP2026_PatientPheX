#!/usr/bin/env python3
"""Fit on12 already-trained prompts only; never a validation/generalization score."""
from pathlib import Path
import argparse,ast,hashlib,json,re

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory
 plan=json.loads((b/'run-plan.json').read_text());assert sha(__file__)==plan['evaluator_sha256']
 frozen=json.loads((b/'predictions-freeze.json').read_text());assert frozen['plan_sha256']==sha(b/'run-plan.json') and frozen['raw_sha256']==sha(b/'epoch2.responses.jsonl')
 assert not (b/'fit-result.json').exists()
 # Import only the unchanged original parse_json function, without unrelated CRF dependencies.
 parser_ref=plan['local_references']['original_parser'];assert sha(parser_ref['path'])==parser_ref['sha256'];tree=ast.parse(Path(parser_ref['path']).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_json');namespace={'json':json,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),parser_ref['path'],'exec'),namespace);parse=namespace['parse_json']
 prompt_ref=plan['local_references']['prompts'];target_ref=plan['local_references']['targets'];assert sha(prompt_ref['path'])==prompt_ref['sha256'] and sha(target_ref['path'])==target_ref['sha256']
 prompts={r['row_id']:r for r in map(json.loads,Path(prompt_ref['path']).read_text().splitlines())};targets={r['row_id']:json.loads(r['answer']) for r in map(json.loads,Path(target_ref['path']).read_text().splitlines())};raw={r['row_id']:r for r in map(json.loads,(b/'epoch2.responses.jsonl').read_text().splitlines())};assert set(prompts)==set(targets)==set(raw)
 rows=[];tp=fp=fn=0;exact=valid=0
 def strict(text):
  def pairs(items):
   d={}
   for k,v in items:
    if k in d:raise ValueError('Duplicate key')
    d[k]=v
   return d
  try:return json.loads(text,object_pairs_hook=pairs)
  except (ValueError,TypeError):return None
 for rid,prompt in prompts.items():
  r=raw[rid];obj=parse(r['response']) if not r['hit_token_limit'] else None;pure=strict(r['response']);patients=set(prompt['patient_ids']);codes=set(prompt['code_map']);good=lambda x:isinstance(x,list) and all(isinstance(y,str) and y in patients for y in x) and len(x)==len(set(x))
  strict_ok=isinstance(pure,dict) and set(pure)==codes and all(good(x) for x in pure.values()) and not r['hit_token_limit'];valid+=strict_ok
  predicted=set();unresolved=0
  for cid in codes:
   v=obj.get(cid) if isinstance(obj,dict) else None
   # Same candidate validity and ID conversion as original merger; no label/baseline fallback.
   if not isinstance(v,list) or any(str(pid) not in patients for pid in v):unresolved+=1;continue
   predicted.update((cid,str(pid)) for pid in v)
  gold={(cid,pid) for cid,v in targets[rid].items() for pid in v};t=len(predicted&gold);f=len(predicted-gold);n=len(gold-predicted);tp+=t;fp+=f;fn+=n;row_exact=strict_ok and predicted==gold;exact+=row_exact
  rows.append(dict(row_id=rid,strict_format=strict_ok,exact_assignment=row_exact,tp=t,fp=f,fn=n,unresolved_candidates=unresolved,hit_token_limit=r['hit_token_limit']))
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 result=dict(status='training_fit_diagnostic_only_not_generalization_Score',source_sha256=sha(__file__),plan_sha256=sha(b/'run-plan.json'),prediction_freeze_sha256=sha(b/'predictions-freeze.json'),raw_sha256=sha(b/'epoch2.responses.jsonl'),targets_sha256=target_ref['sha256'],rows=12,strict_format_rows=valid,exact_assignment_rows=exact,exact_assignment_rate=exact/12,relation=dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1),row_results=rows,original_parser_reused=True,no_originaldev_or_outergold=True,no_new_training=True)
 with (b/'fit-result.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='row_results'},indent=2))
if __name__=='__main__':main()
