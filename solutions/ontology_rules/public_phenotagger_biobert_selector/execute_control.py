"""Bound four fits to120s; decode/freeze/evaluate the sole endpoint within30s."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from prepare_pool import FEATURES,sha,ref,checked,read,rows,write,sources

def now():return datetime.now(timezone.utc).isoformat()
def nonentities(d):return {k:v for k,v in d.items() if k!='entities'}
def keyed(p):
    data=rows(p);out={x['pmc_id']:x for x in data};assert len(data)==len(out);return out

def finalize(plan_path):
    started=time.monotonic();plan=read(plan_path);sources(plan);root=Path(plan['output_dir']);assert plan['execution_ready'] is True
    fitstage=read(root/'fit-stage.json');assert fitstage['plan_sha256']==sha(plan_path) and fitstage['all_four_fits_complete'] is True
    assert not (root/'complete-freeze.json').exists() and not (root/'evaluation-start.json').exists()
    sys.path.insert(0,plan['nested_source']);import decode_nested_selector as decoder
    _,allowed,_,_=decoder.prior.base.parse_hpo(checked(plan['hpo']));allowed.discard('HP:0000118')
    prepared=read(checked(plan['prepared_manifest']));oldfreeze=read(checked(plan['baseline_freeze']))
    assert len(oldfreeze['all_eight_predictions'])==4
    freeze=[];predictions={};inputbyid={};seen=set();diagnostics=[]
    for fold in plan['folds']:
        f=fold['fold'];prep=prepared['folds'][f];target=root/('fold'+str(f));fit=target/'fitted';manifest=read(fit/'manifest.json')
        assert manifest['plan_sha256']==sha(plan_path) and manifest['fold']==f and manifest['actual_iterations']==160
        assert manifest['source_sha256']==plan['sources'][str(Path(__file__).with_name('fit_control.py').resolve())]
        assert oldfreeze['all_eight_predictions'][f]['nested']==fold['baseline_prediction']
        baseline=rows(checked(fold['baseline_prediction']));scores=rows(checked(manifest['outputs']['selector-scores.jsonl']))
        checked(manifest['outputs']['model-vectorizer.pkl']);inputs=rows(checked(prep['input']))
        ids=[x['pmc_id'] for x in baseline];assert len(ids)==len(set(ids))==16 and ids==[x['pmc_id'] for x in scores]==[x['pmc_id'] for x in inputs]
        assert not seen&set(ids);seen.update(ids);new=[]
        for old,scored,row in zip(baseline,scores,inputs):
            assert set(row['input'])=={'pmc_id','pmid','patient','full_text'}
            assert [{k:v for k,v in c.items() if k!='selector_probability'} for c in scored['candidates']]==row['candidates']
            greedy=decoder.greedy(scored['candidates'],.3)
            selected,diag=decoder.decode(scored['candidates'],greedy,row['input'],allowed,cut=.3)
            updated=copy.deepcopy(old);updated['entities']=selected;assert nonentities(updated)==nonentities(old)
            new.append(updated);inputbyid[old['pmc_id']]=row['input'];diagnostics.append(dict(fold=f,pmc_id=old['pmc_id'],**diag))
        path=target/'biobert.nested.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            for doc in new:stream.write(json.dumps(doc,ensure_ascii=False)+'\n')
        freeze.append(dict(fold=f,baseline_prediction=fold['baseline_prediction'],new_prediction=ref(path),new_manifest=ref(fit/'manifest.json'),scores=manifest['outputs']['selector-scores.jsonl'],model=manifest['outputs']['model-vectorizer.pkl']))
        predictions[f]=(baseline,new)
    assert len(seen)==64;sources(plan);write(root/'input-only-diagnostics.json',diagnostics)
    frozen=dict(frozen_at_utc=now(),plan_sha256=sha(plan_path),source_sha256=sha(__file__),folds=freeze,
                all_four_predictions_complete=True,new_quality_evaluation_started=False,all_nonentity_objects_unchanged=True,
                candidate_preservation_not_entity_preservation=True,proper_nesting_source_sha256=sha(decoder.__file__),seconds_before_labels=time.monotonic()-started)
    write(root/'complete-freeze.json',frozen)
    # Reverify all full predictions before importing evaluator and reading held quality labels.
    for fold in freeze:checked(fold['baseline_prediction']);checked(fold['new_prediction'])
    sys.path.insert(0,plan['evaluation_source']);import evaluate;import validate_predictions as validator
    assert evaluate.VERSION=='patientphex-public-approx-v2'
    assert sha(evaluate.__file__)==plan['sources'][str(Path(evaluate.__file__).resolve())]
    assert sha(validator.__file__)==plan['sources'][str(Path(validator.__file__).resolve())]
    write(root/'evaluation-start.json',dict(started_utc=now(),complete_freeze=ref(root/'complete-freeze.json'),plan_sha256=sha(plan_path),source_sha256=sha(__file__)))
    gold=rows(checked(plan['train64_gold']));truth={d['pmc_id']:d for d in gold};assert len(truth)==64 and set(truth)==seen
    for did,doc in truth.items():assert {k:v for k,v in doc.items() if k not in ('entities','association')}==inputbyid[did]
    hpo=validator.read_hpo(checked(plan['hpo']));assert hpo['version']=='hp/releases/2026-06-23'
    results=[];byid={};oldbyid={}
    for fold in plan['folds']:
        f=fold['fold'];before,after=predictions[f];goldfold=[truth[x['pmc_id']] for x in after]
        old=read(checked(fold['baseline_metrics']));oldvalid=read(checked(fold['baseline_validation']))
        report=evaluate.evaluate_report(goldfold,after,'all','text');valid=validator.validate_predictions(goldfold,after,hpo)
        assert valid['valid'] and valid['error_count']==valid['warning_count']==0
        assert oldvalid['valid'] and oldvalid['error_count']==0
        assert report['counts']['relation']==old['counts']['relation'] and report['patients']==old['patients']
        for metric in ('micF1','macF1'):assert report['metrics'][metric]==old['metrics'][metric]
        target=root/('fold'+str(f));write(target/'biobert.metrics.json',report);write(target/'biobert.validation.json',valid)
        metric=lambda r:(r['metrics']['menF1']+r['metrics']['docF1'])/2
        results.append(dict(fold=f,baseline_metrics=fold['baseline_metrics'],baseline=old['metrics'],biobert=report['metrics'],deltas={k:report['metrics'][k]-old['metrics'][k] for k in old['metrics']},baseline_objective=metric(old),biobert_objective=metric(report),objective_delta=metric(report)-metric(old)))
        byid.update({x['pmc_id']:x for x in after});oldbyid.update({x['pmc_id']:x for x in before})
    ordered=[byid[d['pmc_id']] for d in gold];pooled=evaluate.evaluate_report(gold,ordered,'all','text');validation=validator.validate_predictions(gold,ordered,hpo)
    oldpooled=read(checked(plan['baseline_pooled_metrics']));assert pooled['counts']['relation']==oldpooled['counts']['relation']
    assert validation['valid'] and validation['error_count']==validation['warning_count']==0
    write(root/'biobert.pooled.metrics.json',pooled);write(root/'biobert.pooled.validation.json',validation)
    oldobjective=(oldpooled['metrics']['menF1']+oldpooled['metrics']['docF1'])/2;objective=(pooled['metrics']['menF1']+pooled['metrics']['docF1'])/2
    result=dict(status='completed_unique_fixed_endpoint',finished_utc=now(),plan_sha256=sha(plan_path),freeze_sha256=sha(root/'complete-freeze.json'),source_sha256=sha(__file__),
                folds=results,pooled_baseline=oldpooled['metrics'],pooled_biobert=pooled['metrics'],pooled_deltas={k:pooled['metrics'][k]-oldpooled['metrics'][k] for k in oldpooled['metrics']},
                pooled_objective_delta=objective-oldobjective,nondecreasing_folds=sum(x['objective_delta']>=0 for x in results),
                prespecified_success=objective>oldobjective and sum(x['objective_delta']>=0 for x in results)>=3,
                relations_unchanged=True,old_nested_metrics_reused=True,original_dev_read=False,official_actions=False,seconds=time.monotonic()-started)
    sources(plan);write(root/'results.json',result);print(json.dumps({'status':result['status'],'pooled_objective_delta':result['pooled_objective_delta'],'nondecreasing_folds':result['nondecreasing_folds']}))

def invoke(command,seconds,root,name,state):
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    began=time.monotonic();job=dict(command=command,started_utc=now(),cap_seconds=seconds)
    with (root/(name+'.stdout')).open('x') as out,(root/(name+'.stderr')).open('x') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
        try:rc=process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            job['timed_out']=True;os.killpg(process.pid,signal.SIGTERM)
            try:rc=process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
    job.update(finished_utc=now(),returncode=rc,seconds=time.monotonic()-began);state['jobs'].append(job)
    (root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    assert rc==0 and not job.get('timed_out'),job

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--finalize',action='store_true');a=ap.parse_args();path=Path(a.plan).resolve()
    if a.finalize:return finalize(path)
    plan=read(path);assert plan['execution_ready'] is True;sources(plan)
    assert plan['fit_infer_CPU_cap_seconds']==120 and plan['decode_evaluation_CPU_cap_seconds']==30 and plan['TERM_grace_seconds']==5
    prepared=read(checked(plan['prepared_manifest']));assert prepared['new_features']==FEATURES
    assert all(f['new_train_rows']<=1.5*f['old_train_rows'] for f in prepared['folds'])
    assert prepared['estimated_four_fit_infer_seconds']<=120, 'Stop before fitting; estimated budget exceeded'
    root=Path(plan['output_dir']);root.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_utc=now(),plan_sha256=sha(path),source_sha256=sha(__file__),jobs=[])
    (root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    start=time.monotonic()
    try:
        for fold in range(4):
            remain=120-(time.monotonic()-start);assert remain>0
            invoke([plan['python'],str(Path(__file__).with_name('fit_control.py')),'--plan',str(path),'--fold',str(fold)],remain,root,'fit'+str(fold),state)
        write(root/'fit-stage.json',dict(plan_sha256=sha(path),all_four_fits_complete=True,seconds=time.monotonic()-start,manifests=[ref(root/('fold'+str(f))/'fitted/manifest.json') for f in range(4)]))
        invoke([plan['python'],str(Path(__file__).resolve()),'--plan',str(path),'--finalize'],30,root,'decode-evaluate',state)
        state.update(status='complete',finished_utc=now(),results=ref(root/'results.json'))
    except BaseException as exc:state.update(status='failed_preserved',failure=repr(exc),finished_utc=now());raise
    finally:(root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
