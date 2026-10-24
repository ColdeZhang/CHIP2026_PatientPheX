"""Explicit postaccept-only score entry, bounded to 30 seconds + TERM 1 second."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import (load_plan, check_acceptance, ref, checked, read, save, now,
                    load_module, rows, evaluation_keys, METRICS, COMPOUND)


def validate_freeze(plan_path, plan):
    root=Path(plan['output_directory']); path=root/'assembled/common-freeze.json'
    frozen=read(path)
    assert frozen['plan']==ref(plan_path) and frozen['sources']==plan['sources']
    assert frozen['both_folds_complete'] is True and frozen['evaluator_calls']==0
    assert frozen['gold_read_or_hashed'] is False and frozen['only_arm']=='N_R_P_at_least_two_votes'
    assert set(frozen['predictions'])=={'F1','F0'}
    for value in frozen['predictions'].values(): checked(value)
    for name in ['contract','preparation','push_receipt']: checked(frozen[name])
    return path,frozen


def worker(args):
    started=time.monotonic(); plan,guard=load_plan(args.plan)
    accepted=check_acceptance(args.postaccept_receipt,args.postaccept_receipt_sha256)
    freeze_path,frozen=validate_freeze(args.plan,plan)
    output=Path(plan['output_directory'])/'evaluation'
    launch=read(output/'launch.json')
    assert launch['plan']==ref(args.plan) and launch['postaccept_receipt']==accepted
    assert launch['freeze']==ref(freeze_path) and launch['sources']==plan['sources']
    save(output/'before-gold.json',{'at_utc':now(),'postaccept_receipt':accepted,
         'common_freeze':ref(freeze_path),'both_folds_complete':True,'gold_not_yet_read':True,
         'new_calls_order':['F1','F0'],'old_evaluator_calls':0})
    # Imports and all label access are after accepted receipt + both-fold freeze.
    validator=load_module('validate_predictions',plan['references']['validator'])
    evaluator=load_module('majority_frozen_evaluator',plan['references']['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    assert evaluator.COMPOUND_HPO.pattern==COMPOUND.pattern
    hpo=validator.read_hpo(plan['references']['hpo']['path'])
    assert hpo['version']=='hp/releases/2026-06-23'
    results={}
    for fold in plan['fold_order']:
        spec=plan['folds'][fold]; refs=spec['references']
        _,prediction,_=rows(frozen['predictions'][fold]['path'])
        _,inputs,_=rows(refs['input']['path'])
        validation=validator.validate_predictions(inputs,prediction,hpo)
        save(output/(fold+'.validation.json'),validation); assert validation['valid']
        guard['enabled']=True
        checked(spec['gold']); gold=validator.load_jsonl(spec['gold']['path'])
        save(output/(fold+'.call-start.json'),{'at_utc':now(),'prediction':frozen['predictions'][fold],
             'gold':spec['gold'],'evaluator':plan['references']['evaluator'],'call_number':len(results)+1})
        metrics=evaluator.evaluate_report(gold,prediction,'all','text')
        metrics['gold_sha256']=spec['gold']['sha256']
        metrics['prediction_sha256']=frozen['predictions'][fold]['sha256']
        save(output/(fold+'.metrics.json'),metrics)
        baseline=read(refs['baseline_metrics']['path'])['metrics']
        rules=read(refs['rules_metrics']['path'])['metrics']
        delta={key:metrics['metrics'][key]-baseline[key] for key in METRICS}
        assert delta['menF1']==delta['docF1']==0
        _,old,_=rows(refs['baseline']['path'])
        prior,current,truth=evaluation_keys(old),evaluation_keys(prediction),evaluation_keys(gold)
        assert current<=prior
        removed=prior-current
        results[fold]={'metrics':metrics['metrics'],'triple_union_baseline':baseline,'rules_reference':rules,
            'delta_vs_triple_union':delta,'delta_vs_rules':{k:metrics['metrics'][k]-rules[k] for k in METRICS},
            'removed_tp':len(removed&truth),'removed_fp':len(removed-truth),'added_relations':0,
            'fold_keep':delta['Score']>0,'macro_decreased':delta['macF1']<0,'old_metrics_reused':True}
    keep=all(results[fold]['fold_keep'] for fold in plan['fold_order'])
    save(output/'result.json',{'at_utc':now(),'folds':results,'keep':keep,'keep_rule':plan['keep_rule'],
        'status':'passed_fixed_twofold_Score_gate_pending_review' if keep else 'negative_closed_no_scan',
        'common_freeze':ref(freeze_path),'postaccept_receipt':accepted,'plan':ref(args.plan),
        'sources':plan['sources'],'new_evaluator_calls':2,'old_evaluator_calls':0,
        'NN_or_fit':False,'original_dev_evaluator_calls':0,'seconds':time.monotonic()-started,
        'scope':plan['scope'],'patient_F0_disclosure':plan['patient_F0_disclosure']})
    print(json.dumps(ref(output/'result.json')))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--postaccept-receipt',type=Path,required=True)
    ap.add_argument('--postaccept-receipt-sha256',required=True)
    ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.worker:
        try: worker(args)
        except BaseException:
            # The original traceback is retained; no automatic retry or rescore.
            plan=read(args.plan); output=Path(plan['output_directory'])/'evaluation'
            if output.exists():
                save(output/'failure.json',{'at_utc':now(),'traceback':traceback.format_exc()})
            raise
        return
    plan,guard=load_plan(args.plan)
    accepted=check_acceptance(args.postaccept_receipt,args.postaccept_receipt_sha256)
    freeze_path,_=validate_freeze(args.plan,plan)
    assert not guard['enabled']
    output=Path(plan['output_directory'])/'evaluation'; output.mkdir(exist_ok=False)
    save(output/'launch.json',{'at_utc':now(),'plan':ref(args.plan),'sources':plan['sources'],
        'postaccept_receipt':accepted,'freeze':ref(freeze_path),'budget_seconds':30,'term_grace_seconds':1,
        'gold_read_in_driver':False,'automatic_retry':False})
    started=time.monotonic(); timed_out=False
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(args.plan.resolve()),
        '--postaccept-receipt',str(args.postaccept_receipt.resolve()),
        '--postaccept-receipt-sha256',args.postaccept_receipt_sha256]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
        process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
        try: rc=process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            timed_out=True; os.killpg(process.pid,signal.SIGTERM)
            try: rc=process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); rc=process.wait()
    save(output/'terminal.json',{'at_utc':now(),'returncode':rc,'timed_out':timed_out,
        'seconds':time.monotonic()-started,'child_pid':process.pid,'budget_seconds':30,'term_grace_seconds':1,
        'result':ref(output/'result.json') if (output/'result.json').exists() else None,
        'postaccept_receipt':accepted,'freeze':ref(freeze_path)})
    print(json.dumps(ref(output/'terminal.json')))
    if rc or timed_out: raise SystemExit(1)


if __name__=='__main__': main()
