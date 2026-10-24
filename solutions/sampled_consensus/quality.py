"""Exactly two fixed endpoint evaluations after complete160 and joint freeze."""
import argparse
import datetime
import importlib
import json
from pathlib import Path
import sys
from assemble import reference,sha,save_json,read_jsonl,relation_set,ENDPOINTS

def verify(ref):
    assert reference(ref['path'])==ref,ref['path'];return Path(ref['path'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--freeze',required=True);ap.add_argument('--native-complete',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());freeze=json.loads(Path(a.freeze).read_text());complete=json.loads(Path(a.native_complete).read_text())
    assert freeze['status']=='BOTH_COMPLETE_PREDICTIONS_FROZEN_NO_QUALITY' and freeze['sampled_fresh_rows']==160
    assert complete['status']=='COMPLETE_FIXED160_FRESH_NATIVE_NONTHINKING_NO_QUALITY' and complete['fresh_rows']==160 and complete['cached_rows']==0
    assert complete['plan_sha256']==sha(a.plan)
    for name,seed in zip(['seed1_raw','seed2_raw'],[20261024,20261025]):
        verify(freeze['references'][name]);row=next(s for s in complete['seeds'] if s['seed']==seed);assert row['fresh_rows']==row['rows']==80 and row['responses_sha256']==freeze['references'][name]['sha256']
    assert set(freeze['endpoints'])==set(ENDPOINTS)
    for ref in freeze['endpoints'].values():verify(ref)
    assert freeze['baseline_retained_primary']==471 and freeze['baseline_outside_candidate_pool_retained_primary']==88 and freeze['entities_each']==1189
    refs=p['quality_refs'];verify(refs['evaluator']);verify(refs['validator']);sys.path.insert(0,str(Path(refs['evaluator']['path']).parent));ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions');assert ev.VERSION=='patientphex-public-approx-v2'
    base=va.load_jsonl(verify(p['baseline']));old=json.loads(verify(p['baseline_metrics']).read_text())['metrics'];assert old['Score']==p['baseline_score']
    inp=va.load_jsonl(verify(refs['input']));hpo=va.read_hpo(verify(refs['hpo']));predictions={name:va.load_jsonl(verify(freeze['endpoints'][name])) for name in ENDPOINTS};formats={name:va.validate_predictions(inp,pred,hpo) for name,pred in predictions.items()}
    assert all(f['valid'] for f in formats.values())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    # All complete predictions/format checks are frozen before the first gold read.
    save_json(out/'evaluation-intent.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':reference(a.plan),'joint_freeze':reference(a.freeze),'native_complete':reference(a.native_complete),'fixed_endpoints':list(ENDPOINTS),'new_v2_planned':2,'old_v2':0,'single_seed_v2':0,'sources':{n:reference(Path(__file__).parent/n) for n in ['quality.py','assemble.py']}})
    gold=va.load_jsonl(verify(refs['gold']));goldpairs=relation_set(gold);basepairs=relation_set(base);results={}
    for name in ENDPOINTS:
        sub=out/name;sub.mkdir();save_json(sub/'format.json',formats[name]);metric=ev.evaluate_report(gold,predictions[name],body_scope='all',no_id_doc='text');metric['gold_sha256']=refs['gold']['sha256'];metric['prediction_sha256']=freeze['endpoints'][name]['sha256'];save_json(sub/'metrics.json',metric)
        delta={k:metric['metrics'][k]-old[k] for k in old};pairs=relation_set(predictions[name]);add=pairs-basepairs;remove=basepairs-pairs
        results[name]={'metrics':metric['metrics'],'delta':delta,'prediction':freeze['endpoints'][name],'metrics_file':reference(sub/'metrics.json'),'format_file':reference(sub/'format.json'),'relations':len(pairs),'added':len(add),'added_TP':len(add&goldpairs),'added_FP':len(add-goldpairs),'removed':len(remove),'removed_TP':len(remove&goldpairs),'removed_FP':len(remove-goldpairs),'strictly_above_B':delta['Score']>0}
    best=max(ENDPOINTS,key=lambda n:results[n]['metrics']['Score']);retained=results[best]['strictly_above_B']
    save_json(out/'result.json',{'status':'COMPLETE_FIXED160_TWO_ENDPOINTS_F1_DEVELOPMENT','baseline':old,'endpoints':results,'primary_success':results[ENDPOINTS[0]]['strictly_above_B'],'retain_any_fixed_endpoint':retained,'best_fixed_endpoint':best,'gate':'Score strictly above B, micro/macro unconditionally reported; no further seed/temperature choices','new_v2':2,'old_v2':0,'single_seed_v2':0,'fit_calls':0,'original_dev_read':False,'official_calls':0,'not_blind':True,'joint_freeze':reference(a.freeze),'plan':reference(a.plan),'native_complete':reference(a.native_complete)})
    print(json.dumps({'best':best,'retain':retained,'metrics':{k:v['metrics'] for k,v in results.items()}}),flush=True)
if __name__=='__main__':main()
