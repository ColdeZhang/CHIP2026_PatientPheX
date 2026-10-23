#!/usr/bin/env python3
"""One fixed cut.3 support filter after F1 rawappend, no model or threshold fitting."""
from pathlib import Path
import argparse,importlib.util,json,datetime

def main():
 p=argparse.ArgumentParser();p.add_argument('--f1-directory',type=Path,required=True);p.add_argument('--support-prediction',type=Path,required=True);p.add_argument('--control-source',type=Path,required=True);a=p.parse_args()
 s=importlib.util.spec_from_file_location('support_helper',a.control_source);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
 assert h.sha(a.control_source)=='684bf9b6b24dd4d72f072815360e29f308cca2ae825b2262eca344d7ad0b1b21'  # frozen outer source
 b=a.f1_directory.resolve();out=b/'support-cut03-control';out.mkdir();plan=h.read(b/'run-plan.json');r=plan['references'];oldfreeze=h.read(b/'paired-predictions-freeze.json')
 for name in ['baseline','augmented','new_associations','raw_freeze']:h.checked(oldfreeze[name])
 assert h.sha(a.support_prediction)=='8ea3a1da772d8aeacb9fa376d1908601a8121d519094b66277fca075f303a426'
 baseline=h.rows(oldfreeze['baseline']['path']);support=h.entity_support(a.support_prediction);extra={tuple(x) for x in h.read(oldfreeze['new_associations']['path'])};assert len(extra)==193
 assert {str(d['pmc_id']) for d in baseline}=={str(d['pmc_id']) for d in h.rows(a.support_prediction)}
 kept={x for x in extra if (x[0],x[2]) in support};pred=h.append_keys(baseline,kept);h.save_rows(out/'supported.pred.jsonl',pred);h.save(out/'addition-keys.json',sorted(kept))
 frozen=dict(status='fixed_F1_support_full_prediction_frozen_before_once_evaluation',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=h.ref(__file__),helper=h.ref(a.control_source),support=h.ref(a.support_prediction),baseline=oldfreeze['baseline'],raw_augmented=oldfreeze['augmented'],raw_additions=oldfreeze['new_associations'],raw_freeze=h.ref(b/'paired-predictions-freeze.json'),prediction=h.ref(out/'supported.pred.jsonl'),additions=h.ref(out/'addition-keys.json'),raw_added=len(extra),supported_added=len(kept),scope='Post-F1 raw-result development control; one fixed cut.3 from pooled64 OOF entity objective including F1, no new threshold scan, no blind/generalization claim; only new relations filtered, all old entities/relations/text preserved')
 h.save(out/'prediction-freeze.json',frozen)
 h.save(out/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=h.ref(out/'prediction-freeze.json')))
 assert h.sha(plan['gold_path'])==plan['historical_F1_gold_sha256']
 h.command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',str(out/'supported.pred.jsonl'),'--hpo',r['hpo']['path'],'--output',str(out/'validation.json')],out/'validation-command.json');assert h.read(out/'validation.json')['valid']
 h.command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',str(out/'supported.pred.jsonl'),'--body-scope','all','--no-id-doc','text','--output',str(out/'metrics.json')],out/'evaluation-command.json')
 v=h.read(out/'metrics.json');old=h.read(b/'baseline.metrics.json');raw=h.read(b/'augmented.metrics.json');gold=h.literal_gold(h.rows(plan['gold_path']));count=h.delta_counts(kept,gold)
 assert v['counts']['relation']['tp']-old['counts']['relation']['tp']==count['tp'] and v['counts']['relation']['fp']-old['counts']['relation']['fp']==count['fp']
 assert all(v['metrics'][k]==old['metrics'][k] for k in ['menF1','docF1']);h.checked(frozen['prediction'])
 result=dict(status='complete_fixed_F1_support_development_control',scope=frozen['scope'],freeze=h.ref(out/'prediction-freeze.json'),format=h.ref(out/'validation.json'),metrics_file=h.ref(out/'metrics.json'),metrics=v['metrics'],counts=v['counts'],new_relations=count,delta_vs_baseline={k:v['metrics'][k]-old['metrics'][k] for k in v['metrics']},delta_vs_raw={k:v['metrics'][k]-raw['metrics'][k] for k in v['metrics']},old_metrics_reused=True,no_gpu=True,no_training=True,no_originaldev_or_outergold=True)
 h.save(out/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
