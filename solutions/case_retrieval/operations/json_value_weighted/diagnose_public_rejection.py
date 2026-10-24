#!/usr/bin/env python3
"""Read-only F1 development bucket diagnosis from two already frozen public62 outputs.
Never produces a competing full prediction/Score or changes the fixed support gate.
"""
from pathlib import Path
import argparse,hashlib,json,importlib.util,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();start=time.monotonic();b=a.base;old=b/'public-new-relations';new=b/'public-training-candidate-preparation/F1-new62';out=b/'public-training-candidate-preparation/F1-rejection-diagnostic';out.mkdir()
 plan=read(new/'run-plan.json');r=plan['references'];h=mod('publicdiag_tree',r['tree_helper']['path']);f=mod('publicdiag_fusion',r['fusion_helper']['path']);c=mod('publicdiag_support',r['support_helper']['path']);h.verify_parser(r['parser']['path'])
 refs={k:r[k] for k in ['prompts','support_prediction','support_helper','tree_helper','fusion_helper','old_baseline_pred']};refs['new_fullfreeze']=ref(new/'paired-predictions-freeze.json');refs['new_summary']=ref(new/'summary.json');refs['old_fullfreeze']=ref(old/'support-cut03-control/prediction-freeze.json');refs['old_summary']=ref(old/'support-cut03-control/summary.json')
 for v in refs.values():assert ref(v['path'])==v
 prompts=rows(r['prompts']['path']);pairs=h.pair_rows(prompts);assert len(prompts)==62 and len(pairs)==1251 and sha(old/'prompts.jsonl')==sha(r['prompts']['path'])
 support=c.entity_support(r['support_prediction']['path']);preds={}
 for name,d in [('old124',old),('new212',new)]:
  pl=read(d/'run-plan.json');fr=read(d/'responses-freeze.json');assert fr['raw_sha256']==sha(d/'responses.jsonl') and fr['manifest_sha256']==sha(d/'responses.manifest.json')
  raw=rows(d/'responses.jsonl');f.verify_sft_response(h,read(d/'responses.manifest.json'),raw,d/'responses.jsonl',d/'prompts.jsonl',pl['adapter_files'],read(r['model_source']['path']))
  flags,parsing=h.llm_features(prompts,raw,pairs);preds[name]={(v['pmc_id'],v['patient_id'],v['concept']) for v,flag in zip(pairs,flags) if flag[0]};refs[name+'_raw_freeze']=ref(d/'responses-freeze.json')
 # Original F1 labels already opened for both frozen arms, solely explanatory diagnosis.
 assert sha(plan['gold_path'])==plan['historical_F1_gold_sha256'];gold=c.literal_gold(rows(plan['gold_path']));allpairs={(v['pmc_id'],v['patient_id'],v['concept']) for v in pairs};assert len(allpairs)==1251
 def counts(s):return dict(n=len(s),TP=len(s&gold),FP=len(s-gold))
 def supported(s):return {v for v in s if (v[0],v[2]) in support}
 summaries={n:dict(raw=counts(s),supported=counts(supported(s)),rejected_by_fixed_support=counts(s-supported(s))) for n,s in preds.items()}
 added=preds['new212']-preds['old124'];removed=preds['old124']-preds['new212']
 result=dict(status='read_only_F1_development_fixed_public62_bucket_diagnosis',source=ref(__file__),sources=refs,gold_sha256=sha(plan['gold_path']),available_new_pair_truth=len(allpairs&gold),arms=summaries,raw_delta=dict(added=counts(added),removed=counts(removed),added_supported=counts(supported(added)),removed_supported=counts(supported(removed)),added_rejected=counts(added-supported(added)),removed_rejected=counts(removed-supported(removed))),fixed_support_not_changed=True,no_competing_full_raw_prediction_or_Score=True,no_threshold_or_example_selection=True,elapsed_seconds=time.monotonic()-start)
 assert result['elapsed_seconds']<30
 (out/'diagnose.py').write_bytes(Path(__file__).read_bytes());(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
