"""Two frozen F1 endpoints, one v2 each, 30 seconds plus TERM 1; no arm scan."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from support import (ARMS,METRICS,KEEP,ref,checked,read,save,now,load_plan,load_module,rows,validate_freeze)


def evaluation_keys(evaluator,docs):
    # Frozen evaluator set semantics accept legitimate repeated gold phenotypes.
    # Never pass gold through prediction duplicate guards or majority helpers.
    return {(str(doc['pmc_id']),pid,value) for doc in docs
            for pid,values in evaluator.associations(doc).items() for value in values}


def worker(plan_path):
    started=time.monotonic();plan,guard=load_plan(plan_path);refs=plan['references']
    assert plan['sources']['score.py']==ref(__file__)
    freeze_path,frozen=validate_freeze(plan_path,plan);output=Path(plan['output_directory'])/'evaluation'
    launch=read(output/'launch.json')
    assert launch['plan']==ref(plan_path) and launch['freeze']==ref(freeze_path) and launch['sources']==plan['sources']
    save(output/'before-gold.json',dict(at_utc=now(),common_freeze=ref(freeze_path),
        both_complete_predictions_checked=True,gold_not_yet_read=True,arms=ARMS,old_evaluator_calls=0))
    validator=load_module('validate_predictions',refs['validator'])
    evaluator=load_module('_patient_balanced_frozen_v2',refs['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    checked(refs['hpo']);hpo=validator.read_hpo(refs['hpo']['path'])
    assert hpo['version']=='hp/releases/2026-06-23'
    _,inputs,input_ids=rows(refs['F1_input']['path'])
    _,old,_=rows(refs['method_baseline']['path'])
    baseline=read(refs['method_baseline_metrics']['path'])['metrics']
    stronger=read(refs['stronger_F1_reference']['path'])['metrics']
    guard['enabled']=True
    checked(plan['gold']);gold=validator.load_jsonl(plan['gold']['path'])
    assert {str(doc['pmc_id']) for doc in gold}==set(input_ids)
    truth=evaluation_keys(evaluator,gold);prior=evaluation_keys(evaluator,old)
    results={};calls=0
    for arm in ARMS:
        prediction_ref=frozen['predictions'][arm]
        _,prediction,ids=rows(prediction_ref['path']);assert ids==input_ids
        metric_path=output/(arm+'.metrics.json');attempt_path=output/(arm+'.call-start.json')
        if metric_path.exists():
            # Recovery may reuse an already complete metric; never evaluate it again.
            metrics=read(metric_path);attempt=read(attempt_path)
            assert attempt['prediction']==prediction_ref and attempt['freeze']==ref(freeze_path)
            assert metrics['prediction_sha256']==prediction_ref['sha256'] and metrics['gold_sha256']==plan['gold']['sha256']
            assert metrics['evaluator_version']=='patientphex-public-approx-v2'
            assert metrics['configuration']['body_scope']=='all' and metrics['configuration']['unlinked_document_mode']=='text'
            reused=True
        else:
            assert not attempt_path.exists(),'An attempted endpoint without a metric must not be called again'
            validation=validator.validate_predictions(inputs,prediction,hpo)
            save(output/(arm+'.validation.json'),validation);assert validation['valid']
            save(attempt_path,dict(at_utc=now(),arm=arm,prediction=prediction_ref,gold=plan['gold'],
                evaluator=refs['evaluator'],freeze=ref(freeze_path),plan=ref(plan_path),exactly_one_new_v2=True))
            metrics=evaluator.evaluate_report(gold,prediction,'all','text');calls+=1
            metrics.update(gold_sha256=plan['gold']['sha256'],prediction_sha256=prediction_ref['sha256'])
            # Persist before any additional diagnostic assertion or calculation.
            save(metric_path,metrics);reused=False
        delta={key:metrics['metrics'][key]-baseline[key] for key in METRICS}
        assert delta['menF1']==delta['docF1']==0
        current=evaluation_keys(evaluator,prediction);added=current-prior;removed=prior-current
        results[arm]=dict(metrics=metrics['metrics'],metric_ref=ref(metric_path),existing_metric_reused=reused,
            original_triple_baseline=baseline,delta_vs_original_triple=delta,
            stronger_fulltext_reference=stronger,delta_vs_stronger_fulltext={k:metrics['metrics'][k]-stronger[k] for k in METRICS},
            added_tp=len(added&truth),added_fp=len(added-truth),removed_tp=len(removed&truth),removed_fp=len(removed-truth),
            macro_decreased_vs_original_triple=delta['macF1']<0,old_metrics_reused_not_rescored=True)
    keep=results['primary']['metrics']['Score']>plan['baseline_score']
    save(output/'result.json',dict(at_utc=now(),endpoints=results,keep=keep,keep_rule=KEEP,
        diagnostic_not_substituted_for_primary=True,macro_has_no_additional_veto=True,
        status='primary_Score_gate_passed_pending_review' if keep else 'fixed_method_negative_closed_no_scan',
        common_freeze=ref(freeze_path),plan=ref(plan_path),sources=plan['sources'],
        new_evaluator_calls_this_worker=calls,completed_new_endpoint_metrics=2,old_evaluator_calls=0,
        original_dev_evaluator_calls=0,NN_or_fit=False,seconds=time.monotonic()-started,
        scope='Known F1 development only; stronger reference disclosed, no automatic F0/originaldev migration or official action'))
    print(json.dumps(ref(output/'result.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path)
    ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=ap.parse_args();args.plan=args.plan.resolve()
    if args.worker:
        try:worker(args.plan)
        except BaseException:
            output=Path(read(args.plan)['output_directory'])/'evaluation'
            if output.exists():save(output/'failure.json',dict(at_utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
            raise
        return
    started=time.monotonic();plan,guard=load_plan(args.plan);freeze_path,frozen=validate_freeze(args.plan,plan)
    assert plan['sources']['score.py']==ref(__file__)
    assert not guard['enabled'];output=Path(plan['output_directory'])/'evaluation'
    if output.exists():
        # Read-only completed inventory is safe; an incomplete attempt is preserved.
        assert (output/'result.json').is_file() and (output/'terminal.json').is_file(),'Prior incomplete evaluation; no automatic retry'
        result=read(output/'result.json');terminal=read(output/'terminal.json')
        assert result['plan']==ref(args.plan) and result['common_freeze']==ref(freeze_path)
        assert terminal['returncode']==0 and terminal['timed_out'] is False
        for arm in ARMS:checked(result['endpoints'][arm]['metric_ref'])
        print(json.dumps(dict(existing_result=ref(output/'result.json'),new_evaluator_calls=0)));return
    output.mkdir()
    save(output/'launch.json',dict(at_utc=now(),plan=ref(args.plan),sources=plan['sources'],freeze=ref(freeze_path),
        budget_seconds=30,term_grace_seconds=1,gold_read_in_supervisor=False,automatic_retry=False))
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(args.plan)]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    timed_out=False
    with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=environment,start_new_session=True)
        save(output/'child-started.json',dict(pid=child.pid,at_utc=now()))
        try:code=child.wait(timeout=max(.001,started+30-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=max(.001,started+31-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        for stream in [stdout,stderr]:stream.flush();os.fsync(stream.fileno())
    save(output/'terminal.json',dict(at_utc=now(),returncode=code,timed_out=timed_out,seconds=time.monotonic()-started,
        child_pid=child.pid,budget_seconds=30,term_grace_seconds=1,common_freeze=ref(freeze_path),
        result=ref(output/'result.json') if (output/'result.json').exists() else None))
    print(json.dumps(ref(output/'terminal.json')))
    if code or timed_out:raise SystemExit(1)


if __name__=='__main__':main()
