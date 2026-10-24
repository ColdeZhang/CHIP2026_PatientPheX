"""Joint freeze two new folds, then unique F2/F3 and pooled quality calls."""
import argparse,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
from common import read_jsonl,write_json,write_jsonl,ref,sha

def checked(r):
    assert sha(r['path'])==r['sha256'];return Path(r['path'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());out=Path(a.output);start=time.monotonic()
    assert not (out/'evaluation-intent.json').exists(),'Existing intent: recover, never reevaluate'
    for path,h in p['sources'].items():assert sha(path)==h
    evref=p['evaluation_sources']['evaluator'];checked(evref);sys.path.insert(0,str(Path(evref['path']).parent));import evaluate as ev
    predictions={};metrics={};freezes={};baselines={};plans={}
    for f in (0,1):
        old=p['known_results'][str(f)];predictions[f]=read_jsonl(checked(old['prediction.jsonl']));metrics[f]=json.loads(checked(old['metrics.json']).read_text())['metrics']
        r=json.loads(checked(old['result.json']).read_text());assert r['metrics']==metrics[f];baselines[f]=r['baseline'];freezes[str(f)]={'reused':True,**old}
    for pr in p['fold_plans']:
        plan=json.loads(checked(pr).read_text());f=plan['outer_fold'];plans[f]=plan;q=out/f'fold{f}'
        freeze=json.loads((q/'complete-freeze.json').read_text());assert freeze['plan']['sha256']==pr['sha256'] and freeze['input_gold_opened'] is False
        predictions[f]=read_jsonl(checked(freeze['prediction']));freezes[str(f)]={'reused':False,'freeze':ref(q/'complete-freeze.json'),'prediction':freeze['prediction']}
        old=json.loads(checked(plan['strong_metrics']).read_text());baselines[f]=old['metrics']
        assert sorted(str(x['pmc_id']) for x in predictions[f])==plan['outer_held_ids']
    pooled=[]
    for f in range(4):assert len(predictions[f])==16;pooled.extend(predictions[f])
    assert len({str(x['pmc_id']) for x in pooled})==64
    write_jsonl(out/'pooled-prediction.jsonl',pooled)
    write_json(out/'joint-freeze.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'plan':ref(a.plan),'source':ref(__file__),'folds':freezes,'pooled':ref(out/'pooled-prediction.jsonl'),'new_held_quality_calls_so_far':0,'known_prior_results':[0,1],'all_four_gp_predictions_used_no_per_fold_model_selection':True})
    write_json(out/'evaluation-intent.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'exact_calls':[2,3,'pooled64'],'old_calls':0,'joint_freeze':ref(out/'joint-freeze.json')})
    for f in (2,3):
        gold=read_jsonl(checked(p['evaluation_sources'][str(f)]));m=ev.evaluate(gold,predictions[f]);write_json(out/f'fold{f}'/'metrics.json',m);metrics[f]=m['metrics']
        assert all(metrics[f][k]==baselines[f][k] for k in ('micF1','macF1'))
    m=ev.evaluate(read_jsonl(checked(p['evaluation_sources']['pooled64'])),pooled);write_json(out/'pooled-metrics.json',m)
    old=json.loads(checked(p['old_pooled_result']).read_text())['pooled_lightgbm'];new=m['metrics']
    assert all(new[k]==old[k] for k in ('micF1','macF1'))
    folds={str(f):{'baseline':baselines[f],'new':metrics[f],'delta':{k:metrics[f][k]-baselines[f][k] for k in baselines[f]}} for f in range(4)}
    n=sum(v['delta']['Score']>=0 for v in folds.values());retain=new['Score']>old['Score'] and n>=3
    result={'status':'COMPLETE_FIXED_FOURFOLD_COMPLETION','folds':folds,'pooled_baseline':old,'pooled':new,'pooled_delta':{k:new[k]-old[k] for k in old},'nondecreasing_folds':n,'retain':retain,'gate':p['gate'],'new_v2_calls':3,'old_v2_calls':0,'new_classifier_fits':2,'known_first_two_fold_results_before_completion':True,'not_blind':True,'original_dev_read':False,'freeze':ref(out/'joint-freeze.json'),'seconds':time.monotonic()-start}
    write_json(out/'result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
