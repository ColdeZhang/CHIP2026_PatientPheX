#!/usr/bin/env python3
"""Single supported append on originaldev; freeze before gold, no raw-arm score."""
from pathlib import Path
import argparse,importlib.util,json,shutil,datetime

def load(b):
 plan=json.loads((b/'run-plan.json').read_text());r=plan['references']
 s=importlib.util.spec_from_file_location('frozen_append',r['append_helper']['path']);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
 assert h.sha(__file__)==plan['control_sha256']
 for value in r.values():h.checked(value)
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 tree=module('frozen_tree_dev',r['tree_helper']['path']);verifier=module('frozen_sft_verifier_dev',r['fusion_helper']['path']);tree.verify_parser(r['parser']['path'])
 return plan,h,tree,verifier

def prepare(b):
 plan,h,tree,v=load(b);r=plan['references'];oldp=h.rows(r['old_prompts']['path']);newp=h.rows(r['prompts']['path']);oldraw=h.rows(r['old_raw']['path'])
 v.verify_sft_response(tree,h.read(r['old_manifest']['path']),oldraw,r['old_raw']['path'],r['old_prompts']['path'],plan['adapter_files'],h.read(r['model_source']['path']))
 oldpairs=tree.pair_rows(oldp);newpairs=tree.pair_rows(newp);oldconcepts={(x['pmc_id'],x['concept']) for x in oldpairs};newconcepts={(x['pmc_id'],x['concept']) for x in newpairs}
 assert len(oldp)==64 and len(oldconcepts)==466 and len(newp)==42 and len(newconcepts)==276 and len(newpairs)==649
 assert not oldconcepts&newconcepts,'Difference must exclude ALL old candidates, not only selected relations'
 d=h.read(r['input_only_difference_freeze']['path']);assert d['artifacts']['union']['prompts']['sha256']==h.sha(r['prompts']['path']) and d['statistics']['union']['new_concepts']==276
 baseline=h.rows(r['baseline_prediction']['path']);oldfull=h.rows(r['old_full64_prediction']['path']);assert len(baseline)==16 and sum(len(x['entities']) for x in baseline)==plan['fixed_control']['entities_preserved']
 by={str(x['pmc_id']):x for x in oldfull};assert all(x['association']==by[str(x['pmc_id'])]['association'] for x in baseline)
 keys={(str(x['pmc_id']),a['patient_id'],c) for x in baseline for a in x['association'] for c in a['phenotype']};assert len(keys)==333 and all((did,c) in oldconcepts for did,pid,c in keys)
 assert h.read(r['baseline_freeze']['path'])['artifacts']['calibrated_point3']['sha256']==h.sha(r['baseline_prediction']['path'])
 assert all(e.get('note') is None for d in baseline for e in d['entities'])
 with (b/'baseline.pred.jsonl').open('xb') as f:f.write(Path(r['baseline_prediction']['path']).read_bytes())
 result=dict(status='existing6401_baseline_preserved_input_only',plan=h.ref(b/'run-plan.json'),prediction=h.ref(b/'baseline.pred.jsonl'),old_all_candidates=466,new_all_candidates=276,newpairs=649,prompts=42,old_relations=333,entities=plan['fixed_control']['entities_preserved'],old_associations_equal_full64=True,no_support_prefilter_of_prompts=True,no_gold_read=True)
 h.save(b/'baseline-prepared.json',result);return result

def freeze(b):
 plan,h,tree,v=load(b);r=plan['references'];prior=h.read(b/'baseline-prepared.json');assert prior['plan']==h.ref(b/'run-plan.json');h.checked(prior['prediction'])
 rf=h.read(b/'responses-freeze.json');assert rf['plan_sha256']==h.sha(b/'run-plan.json') and rf['raw_sha256']==h.sha(b/'responses.jsonl') and rf['manifest_sha256']==h.sha(b/'responses.manifest.json')
 p=h.rows(r['prompts']['path']);raw=h.rows(b/'responses.jsonl');meta=h.read(b/'responses.manifest.json');v.verify_sft_response(tree,meta,raw,b/'responses.jsonl',r['prompts']['path'],plan['adapter_files'],h.read(r['model_source']['path']))
 assert len(raw)==42 and all(not x['hit_token_limit'] and isinstance(json.loads(x['response']),dict) for x in raw)
 pairs=tree.pair_rows(p);flags,stats=tree.llm_features(p,raw,pairs);assert stats['valid_candidate_decisions']==276 and stats['unresolved_candidate_decisions']==0
 rawkeys={(x['pmc_id'],x['patient_id'],x['concept']) for x,flag in zip(pairs,flags) if flag[0]};support=h.entity_support(r['baseline_prediction']['path']);kept={x for x in rawkeys if (x[0],x[2]) in support};baseline=h.rows(prior['prediction']['path']);supported=h.append_keys(baseline,kept)
 h.save_rows(b/'supported.pred.jsonl',supported);h.save(b/'raw-addition-keys-unscored.json',sorted(rawkeys));h.save(b/'supported-addition-keys.json',sorted(kept))
 result=dict(status='existing_baseline_and_single_supported_prediction_frozen',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=h.ref(b/'run-plan.json'),baseline=prior['prediction'],supported=h.ref(b/'supported.pred.jsonl'),raw_freeze=h.ref(b/'responses-freeze.json'),raw_keys_unscored=h.ref(b/'raw-addition-keys-unscored.json'),supported_keys=h.ref(b/'supported-addition-keys.json'),raw_new_relation_count=len(rawkeys),supported_new_relation_count=len(kept),parse=stats,old_all_preserved=True,no_raw_fullprediction_or_score=True,no_gold_read=True)
 h.save(b/'paired-predictions-freeze.json',result);return result

def evaluate(b):
 plan,h,tree,v=load(b);r=plan['references'];f=h.read(b/'paired-predictions-freeze.json');assert f['status']=='existing_baseline_and_single_supported_prediction_frozen' and f['plan']==h.ref(b/'run-plan.json')
 for key in ['baseline','supported','raw_freeze','raw_keys_unscored','supported_keys']:h.checked(f[key])
 assert Path(f['baseline']['path']).read_bytes()==Path(r['baseline_prediction']['path']).read_bytes()
 h.save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=h.ref(b/'paired-predictions-freeze.json'),scope='Once originaldev single supported arm; baseline metrics reused, raw arm not evaluated'))
 assert h.sha(plan['gold_path'])==plan['historical_gold_sha256']
 h.command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',f['supported']['path'],'--hpo',r['hpo']['path'],'--output',str(b/'supported.validation.json')],b/'validation-command.json');assert h.read(b/'supported.validation.json')['valid']
 h.command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',f['supported']['path'],'--body-scope','all','--no-id-doc','text','--output',str(b/'supported.metrics.json')],b/'evaluation-command.json')
 new=h.read(b/'supported.metrics.json');old=h.read(r['baseline_metrics']['path']);assert new['documents']==16 and new['patients']==36
 assert all(new['metrics'][k]==old['metrics'][k] for k in ['menF1','docF1']);extra={tuple(x) for x in h.read(f['supported_keys']['path'])};counts=h.delta_counts(extra,h.literal_gold(h.rows(plan['gold_path'])))
 assert new['counts']['relation']['tp']-old['counts']['relation']['tp']==counts['tp'] and new['counts']['relation']['fp']-old['counts']['relation']['fp']==counts['fp']
 h.checked(f['baseline']);h.checked(f['supported'])
 result=dict(status='complete_single_supported_originaldev_transfer',scope=plan['evaluation_scope'],baseline=dict(metrics=old['metrics'],counts=old['counts'],prediction=f['baseline'],reused_metrics=r['baseline_metrics']),supported=dict(metrics=new['metrics'],counts=new['counts'],prediction=f['supported'],metrics_file=h.ref(b/'supported.metrics.json'),format=h.ref(b/'supported.validation.json')),delta={k:new['metrics'][k]-old['metrics'][k] for k in new['metrics']},new_relations=counts,freeze=h.ref(b/'paired-predictions-freeze.json'),plan=h.ref(b/'run-plan.json'),entities_and_old_relations_preserved=True,no_raw_arm_evaluation=True,no_training=True,official_submission=False)
 h.save(b/'summary.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','freeze','evaluate']);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();print(json.dumps(globals()[a.phase](a.directory.resolve()),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
