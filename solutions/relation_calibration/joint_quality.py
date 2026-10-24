"""Joint freeze and the single LR F0 evaluation. No training or neural calls."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from common import checked, mkdir_new, now, read_json, ref, write_json


def imports(plan):
    evaluator = checked(plan['evaluator'])
    checked(plan['validator'])
    sys.path.insert(0, str(evaluator.parent))
    return importlib.import_module('evaluate'), importlib.import_module('validate_predictions')


def freeze(plan_path):
    p = json.loads(Path(plan_path).read_text())
    assert p['stage'] == 'JOINT_F0_FIXED_TWO_OF_THREE_AND_FROZEN_LR'
    for r in p['sources'].values(): checked(r)
    forbidden=os.path.realpath(p['gold']['path'])
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            assert os.path.realpath(os.fsdecode(args[0]))!=forbidden, 'F0 gold before joint freeze'
    sys.addaudithook(audit)
    ev,va=imports(p)
    inp=va.load_jsonl(checked(p['input']))
    hpo=va.read_hpo(checked(p['hpo']))
    assert len(inp)==16 and sum(len(d['patient']) for d in inp)==47
    f2=read_json(p['two_of_three']['freeze']);lr=read_json(p['logistic_regression']['freeze'])
    assert f2['plan']==p['two_of_three']['plan'] and lr['plan']==p['logistic_regression']['plan']
    assert f2['predictions']['two_of_three']['prediction']==p['two_of_three']['prediction']
    assert lr['prediction']==p['logistic_regression']['prediction']
    assert f2['no_gold_read'] and not lr['gold_read'] and lr['v2_calls']==0
    assert lr['model_state']['sha256']=='0fc19cbed85fadc7be7e23e8365d1d734843d33e467c7f154ebe1f8eff51d1a2'
    out=mkdir_new(p['joint_output'])
    base=va.load_jsonl(checked(p['baseline']))
    def pair(rows):return {(d['pmc_id'],pid,v) for d in rows for pid,vs in ev.associations(d).items() for v in vs}
    old=pair(base);assert len(old)==315
    by={d['pmc_id']:d for d in base};assert len(by)==16
    formats={}
    for arm in ('two_of_three','logistic_regression'):
        rows=va.load_jsonl(checked(p[arm]['prediction']))
        assert len(rows)==16 and {d['pmc_id'] for d in rows}==set(by)
        assert sum(len(d['entities']) for d in rows)==947
        for d in rows:
            b=by[d['pmc_id']]
            assert {k:v for k,v in d.items() if k!='association'}=={k:v for k,v in b.items() if k!='association'}
            assert ev.associations(d).keys()==ev.associations(b).keys()
            assert old<=pair(rows)
        valid=va.validate_predictions(inp,rows,hpo)
        formats[arm]=write_json(out/(arm+'.format.json'),valid)
        assert valid['valid']
    write_json(out/'joint-freeze.json',{'status':'TWO_COMPLETE_F0_PREDICTIONS_FROZEN_NO_GOLD','utc':now(),'plan':ref(plan_path),'two_of_three':p['two_of_three'],'logistic_regression':p['logistic_regression'],'formats':formats,'baseline':p['baseline'],'documents':16,'patients':47,'entities':947,'old_relations_protected':315,'gold_read':False,'fit_calls':0,'v2_calls':0,'gates':{'two_of_three':'Score > B0','logistic_regression':'Score > B0 AND Score > fixed two_of_three'},'not_blind':'Both are fixed development controls; F1 was explicitly meta-training for LR.'})


def evaluate_lr(plan_path,joint_path):
    started=time.monotonic();p=json.loads(Path(plan_path).read_text())
    for r in p['sources'].values():checked(r)
    joint=json.loads(Path(joint_path).read_text())
    assert joint['status']=='TWO_COMPLETE_F0_PREDICTIONS_FROZEN_NO_GOLD' and joint['plan']==ref(plan_path)
    for arm in ('two_of_three','logistic_regression'):
        assert joint[arm]==p[arm]
        for r in joint[arm].values():checked(r)
    # The 2of3 quality result is needed for the predeclared comparative gate;
    # it is read as existing metadata, never re-evaluated.
    fixed_result=json.loads(Path(p['two_of_three_result_path']).read_text())
    assert fixed_result['freeze']==p['two_of_three']['freeze'] and fixed_result['new_v2']==1 and fixed_result['old_v2']==0
    fixed_score=fixed_result['results']['two_of_three']['metrics']['Score']
    out=mkdir_new(p['lr_quality_output'])
    write_json(out/'evaluation-intent.json',{'utc':now(),'joint_freeze':ref(joint_path),'plan':ref(plan_path),'new_v2_call_index':1,'old_v2_calls':0})
    ev,va=imports(p)
    gold=va.load_jsonl(checked(p['gold']))
    prediction=va.load_jsonl(checked(p['logistic_regression']['prediction']))
    baseline_metrics=read_json(p['baseline_metrics'])['metrics']
    assert baseline_metrics['Score']==0.7115075800266846
    metrics=ev.evaluate_report(gold,prediction,body_scope='all',no_id_doc='text')
    metrics_ref=write_json(out/'metrics.json',metrics)
    delta={k:metrics['metrics'][k]-v for k,v in baseline_metrics.items()}
    assert delta['menF1']==delta['docF1']==0
    def pairs(rows):return {(d['pmc_id'],pid,v) for d in rows for pid,vs in ev.associations(d).items() for v in vs}
    base=va.load_jsonl(checked(p['baseline']));added=pairs(prediction)-pairs(base)
    assert pairs(base)<=pairs(prediction)
    tp=len(added & pairs(gold));score=metrics['metrics']['Score']
    write_json(out/'result.json',{'status':'COMPLETE_FIXED_LR_F0_META_HELD_DEVELOPMENT','utc':now(),'joint_freeze':ref(joint_path),'prediction':p['logistic_regression']['prediction'],'metrics_file':metrics_ref,'metrics':metrics['metrics'],'baseline':baseline_metrics,'delta':delta,'two_of_three_score':fixed_score,'delta_vs_two_of_three':score-fixed_score,'added':len(added),'added_TP':tp,'added_FP':len(added)-tp,'retain':score>baseline_metrics['Score'] and score>fixed_score,'gate':'Score strictly above B0 AND fixed2of3; macro disclosed','new_v2':1,'old_v2':0,'fit_calls':0,'NN_calls':0,'original_dev_read':False,'not_blind':True,'seconds':time.monotonic()-started})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--joint-freeze');parser.add_argument('mode',choices=['freeze','evaluate-lr']);a=parser.parse_args()
    if a.mode=='freeze':freeze(a.plan)
    else:assert a.joint_freeze;evaluate_lr(a.plan,a.joint_freeze)
