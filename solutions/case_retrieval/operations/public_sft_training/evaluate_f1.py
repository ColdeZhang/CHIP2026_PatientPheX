#!/usr/bin/env python3
"""One prospective supported public62 arm; old80 decisions/nonrelations unchanged."""
from pathlib import Path
import argparse,copy,datetime,hashlib,importlib.util,json,subprocess,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def checked(r):assert ref(r['path'])==r,r['path']
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory;start=time.monotonic();plan=read(b/'run-plan.json');r=plan['references'];assert sha(__file__)==plan['control_sha256']
 for x in r.values():checked(x)
 h=module('public_aug_tree',r['tree_helper']['path']);f=module('public_aug_fusion',r['fusion_helper']['path']);c=module('public_aug_support',r['support_helper']['path']);h.verify_parser(r['parser']['path'])
 assert sha(r['support_helper']['path'])=='684bf9b6b24dd4d72f072815360e29f308cca2ae825b2262eca344d7ad0b1b21'
 assert sha(r['support_prediction']['path'])=='8ea3a1da772d8aeacb9fa376d1908601a8121d519094b66277fca075f303a426'
 terminal=read(r['training_complete']['path']);cp=read(r['checkpoint']['path']);assert terminal['epochs']==2 and terminal['global_step']==54 and terminal['elapsed_seconds']<2400
 assert cp['completed_epochs']==2 and cp['global_step']==54 and not cp['partial_not_for_inference']
 rawfreeze=read(b/'responses-freeze.json');assert rawfreeze['plan_sha256']==sha(b/'run-plan.json') and rawfreeze['raw_sha256']==sha(b/'responses.jsonl') and rawfreeze['manifest_sha256']==sha(b/'responses.manifest.json')
 prompts=rows(r['prompts']['path']);raw=rows(b/'responses.jsonl');f.verify_sft_response(h,read(b/'responses.manifest.json'),raw,b/'responses.jsonl',r['prompts']['path'],plan['adapter_files'],read(r['model_source']['path']))
 pairs=h.pair_rows(prompts);assert len(prompts)==62 and len(pairs)==1251
 flags,parse=h.llm_features(prompts,raw,pairs);added={(x['pmc_id'],x['patient_id'],x['concept']) for x,v in zip(pairs,flags) if v[0]}
 support=c.entity_support(r['support_prediction']['path']);kept={k for k in added if (k[0],k[2]) in support}
 base=rows(r['old_baseline_pred']['path']);oldkeys={(str(d['pmc_id']),a['patient_id'],v) for d in base for a in d['association'] for v in a['phenotype']}
 oldall={(x['pmc_id'],x['patient_id'],x['concept']) for x in h.pair_rows(rows(r['old_prompts']['path']))}
 newall={(x['pmc_id'],x['patient_id'],x['concept']) for x in pairs};assert not oldall&newall and not oldkeys&kept and len(oldkeys)==383
 assert len(base)==16 and sum(len(d['entities']) for d in base)==1189
 pred=c.append_keys(base,kept);path=b/'supported.pred.jsonl';c.save_rows(path,pred)
 assert {(str(d['pmc_id']),a['patient_id'],v) for d in pred for a in d['association'] for v in a['phenotype']}==oldkeys|kept
 for x,y in zip(base,pred):
  assert {k:v for k,v in x.items() if k!='association'}=={k:v for k,v in y.items() if k!='association'}
  for aa,bb in zip(x['association'],y['association']):assert {k:v for k,v in aa.items() if k!='phenotype'}=={k:v for k,v in bb.items() if k!='phenotype'} and bb['phenotype'][:len(aa['phenotype'])]==aa['phenotype']
 save(b/'new-supported-keys.json',sorted(kept))
 freeze=dict(status='old_supported_and_new_complete_predictions_frozen_before_unique_eval',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(b/'run-plan.json'),old_supported=r['old_supported_pred'],new_supported=ref(path),new_keys=ref(b/'new-supported-keys.json'),raw_freeze=ref(b/'responses-freeze.json'),old383_preserved=True,entities1189_preserved=True,raw_added_count=len(added),supported_added_count=len(kept),raw_unfiltered_arm_not_scored=True,parse=parse)
 save(b/'paired-predictions-freeze.json',freeze)
 save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=ref(b/'paired-predictions-freeze.json'),only_new_arm=True,originaldev_and_F0_unused=True))
 # First new quality access only after both complete prediction references freeze.
 assert sha(plan['gold_path'])==plan['historical_F1_gold_sha256']
 c.command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',str(path),'--hpo',r['hpo']['path'],'--output',str(b/'validation.json')],b/'validation-command.json');assert read(b/'validation.json')['valid']
 c.command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',str(path),'--body-scope','all','--no-id-doc','text','--output',str(b/'metrics.json')],b/'evaluation-command.json')
 old=read(r['old_supported_metrics']['path']);new=read(b/'metrics.json');assert new['documents']==16 and new['patients']==39 and all(old['metrics'][k]==new['metrics'][k] for k in ['menF1','docF1'])
 prior=rows(r['old_supported_pred']['path']);prior_keys={(str(d['pmc_id']),a['patient_id'],v) for d in prior for a in d['association'] for v in a['phenotype']};nowkeys=oldkeys|kept;gold=c.literal_gold(rows(plan['gold_path']));take=nowkeys-prior_keys;drop=prior_keys-nowkeys
 assert new['counts']['relation']['tp']==len(nowkeys&gold) and new['counts']['relation']['fp']==len(nowkeys-gold) and new['counts']['relation']['fn']==len(gold-nowkeys)
 elapsed=time.monotonic()-start
 result=dict(status='complete_public_expanded_training_fixed_F1_new62_supported_control',scope='Training-side F1 development, stricttrain32 SFT; fixed pooled64-calibrated support reused; no blind/generalization/official claim',prediction=ref(path),freeze=ref(b/'paired-predictions-freeze.json'),metrics_file=ref(b/'metrics.json'),format=ref(b/'validation.json'),old_cached_metrics=r['old_supported_metrics'],old_metrics=old['metrics'],new_metrics=new['metrics'],counts=new['counts'],delta={k:new['metrics'][k]-old['metrics'][k] for k in new['metrics']},new_supported_additions=c.delta_counts(kept,gold),delta_decisions=dict(added=len(take),added_TP=len(take&gold),added_FP=len(take-gold),removed=len(drop),removed_TP=len(drop&gold),removed_FP=len(drop-gold)),keep_by_predeclared_score_increase=new['metrics']['Score']>old['metrics']['Score'],old383_exact=True,entities1189_exact=True,no_gate_or_threshold_search=True,raw_arm_not_scored=True,elapsed_CPU_seconds=elapsed)
 checked(freeze['new_supported']);checked(freeze['old_supported']);save(b/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
