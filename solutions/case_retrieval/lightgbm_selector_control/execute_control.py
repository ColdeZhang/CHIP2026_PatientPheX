"""Four fixed LightGBM fits in one 180s window, then freeze and evaluate once."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,'/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector')
from prepare_pool import sha, ref, checked, read, rows, write, sources


def now():return datetime.now(timezone.utc).isoformat()


def finalize(plan_path):
    start=time.monotonic();signal.alarm(30)
    plan=read(plan_path);sources(plan);root=Path(plan['output_dir'])
    fitstage=read(root/'fit-stage.json');assert fitstage['plan_sha256']==sha(plan_path) and fitstage['all_four_fits_complete']
    assert not (root/'complete-freeze.json').exists() and not (root/'evaluation-start.json').exists()
    sys.path.insert(0,plan['literal_source_directory']);import control as literal
    assert sha(literal.__file__)==plan['sources'][str(Path(literal.__file__).resolve())]
    sys.path.insert(0,plan['nested_source']);import decode_nested_selector as decoder
    _,allowed,_,_=decoder.prior.base.parse_hpo(checked(plan['hpo']));allowed.discard('HP:0000118')
    prepared=read(checked(plan['prepared_manifest']));baseline_freeze=read(checked(plan['strong_biobert_freeze']))
    predictions={};freeze=[];diagnostics=[];inputbyid={};seen=set()
    for f in range(4):
        target=root/f'fold{f}';manifest=read(target/'fitted/manifest.json')
        assert manifest['plan_sha256']==sha(plan_path) and manifest['fold']==f and manifest['actual_iterations']==160
        assert manifest['source_sha256']==plan['sources'][str(Path(__file__).with_name('fit_fold.py').resolve())]
        assert manifest['positive_class_probability_column']==1 and manifest['saved_reload_all_input_probabilities_exact']
        previous=baseline_freeze['folds'][f]['new_prediction'];old_bytes=checked(previous).read_bytes()
        lines=old_bytes.decode('utf-8').splitlines(keepends=True)
        scored=rows(checked(manifest['outputs']['selector-scores.jsonl']))
        inputs=rows(checked(prepared['folds'][f]['input']))
        assert len(lines)==len(scored)==len(inputs)==16
        output=[];new=[];changes=0
        for line,score,row in zip(lines,scored,inputs):
            before=json.loads(line);pmc=before['pmc_id']
            assert pmc==score['pmc_id']==row['pmc_id'] and pmc not in seen;seen.add(pmc)
            assert set(row['input'])=={'pmc_id','pmid','patient','full_text'}
            assert [{k:v for k,v in x.items() if k!='selector_probability'} for x in score['candidates']]==row['candidates']
            selected,diag=decoder.decode(score['candidates'],decoder.greedy(score['candidates'],.3),row['input'],allowed,.3)
            result=literal.replace_entities(line,selected)
            changes+=before['entities']!=selected;output.append(result);new.append(json.loads(result));inputbyid[pmc]=row['input']
            oldkeys={decoder.key(x) for x in before['entities']};newkeys={decoder.key(x) for x in selected}
            diagnostics.append(dict(fold=f,pmc_id=pmc,entity_added=len(newkeys-oldkeys),entity_removed=len(oldkeys-newkeys),**diag))
        path=target/'lightgbm.nested.jsonl';path.write_bytes(''.join(output).encode('utf-8'))
        freeze.append(dict(fold=f,baseline=previous,new_prediction=ref(path),scores=manifest['outputs']['selector-scores.jsonl'],
                           model=manifest['outputs']['model-vectorizer.pkl'],manifest=ref(target/'fitted/manifest.json'),changed_documents=changes))
        predictions[f]=new
    assert len(seen)==64;sources(plan)
    write(root/'input-only-diagnostics.json',diagnostics)
    write(root/'complete-freeze.json',dict(frozen_at_utc=now(),plan_sha256=sha(plan_path),all_four_predictions_complete=True,
          new_quality_evaluation_started=False,all_nonentity_literal_bytes_unchanged=True,folds=freeze,
          diagnostics=ref(root/'input-only-diagnostics.json'),seconds_before_labels=time.monotonic()-start))
    for item in freeze:checked(item['new_prediction'])
    write(root/'evaluation-start.json',dict(started_utc=now(),freeze=ref(root/'complete-freeze.json')))
    sys.path.insert(0,plan['evaluation_source']);import evaluate;import validate_predictions as validator
    assert evaluate.VERSION=='patientphex-public-approx-v2'
    for module in (evaluate,validator):assert sha(module.__file__)==plan['sources'][str(Path(module.__file__).resolve())]
    gold=rows(checked(plan['train64_gold']));truth={x['pmc_id']:x for x in gold}
    assert len(truth)==len(gold)==64 and set(truth)==seen
    assert all({k:v for k,v in doc.items() if k not in ('entities','association')}==inputbyid[pmc] for pmc,doc in truth.items())
    hpo=validator.read_hpo(checked(plan['hpo']));results=[];newbyid={}
    for f in range(4):
        docs=predictions[f];labels=[truth[x['pmc_id']] for x in docs]
        baseline=read(checked(plan['strong_biobert_fold_metrics'][f]))
        report=evaluate.evaluate_report(labels,docs,'all','text');valid=validator.validate_predictions(labels,docs,hpo)
        assert valid['valid'] and valid['error_count']==valid['warning_count']==0
        assert report['counts']['relation']==baseline['counts']['relation']
        write(root/f'fold{f}/lightgbm.metrics.json',report);write(root/f'fold{f}/lightgbm.validation.json',valid)
        results.append(dict(fold=f,baseline=baseline['metrics'],lightgbm=report['metrics'],
                            deltas={k:report['metrics'][k]-baseline['metrics'][k] for k in baseline['metrics']},
                            counts_before=baseline['counts'],counts_after=report['counts']))
        newbyid.update({x['pmc_id']:x for x in docs})
    pooled=evaluate.evaluate_report(gold,[newbyid[x['pmc_id']] for x in gold],'all','text')
    oldpooled=read(checked(plan['strong_biobert_pooled_metrics']))
    assert pooled['counts']['relation']==oldpooled['counts']['relation']
    write(root/'lightgbm.pooled.metrics.json',pooled)
    delta=pooled['metrics']['Score']-oldpooled['metrics']['Score'];nondecreasing=sum(x['deltas']['Score']>=0 for x in results)
    result=dict(status='complete_unique_fixed_endpoint',finished_utc=now(),plan_sha256=sha(plan_path),source_sha256=sha(__file__),
                freeze=ref(root/'complete-freeze.json'),folds=results,pooled_baseline=oldpooled['metrics'],pooled_lightgbm=pooled['metrics'],
                pooled_deltas={k:pooled['metrics'][k]-oldpooled['metrics'][k] for k in oldpooled['metrics']},
                nondecreasing_folds=nondecreasing,keep=delta>0 and nondecreasing>=3,
                changed_documents=sum(x['changed_documents'] for x in freeze),entity_added=sum(x['entity_added'] for x in diagnostics),
                entity_removed=sum(x['entity_removed'] for x in diagnostics),baseline_metrics_reused=True,new_v2_calls=5,old_v2_calls=0,
                no_gpu=True,no_originaldev=True,no_official_actions=True,nonentity_literal_bytes_preserved=True,
                fit_stage_seconds=fitstage['seconds'],decode_evaluation_seconds=time.monotonic()-start)
    assert result['decode_evaluation_seconds']<30
    write(root/'result.json',result);signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('pooled_deltas','nondecreasing_folds','keep','fit_stage_seconds','decode_evaluation_seconds','changed_documents','entity_added','entity_removed')}))


def invoke(command, cap, directory, name, state, env):
    started=time.monotonic();job=dict(command=command,cap_seconds=cap,started_utc=now())
    with (directory/(name+'.stdout')).open('x') as out,(directory/(name+'.stderr')).open('x') as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True)
        job['pid']=process.pid
        try:rc=process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            job['timed_out']=True;os.killpg(process.pid,signal.SIGTERM)
            try:rc=process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
    job.update(returncode=rc,finished_utc=now(),seconds=time.monotonic()-started);state['jobs'].append(job)
    (directory/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    assert rc==0 and not job.get('timed_out'),job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.finalize:return finalize(Path(args.plan).resolve())
    plan=read(args.plan);sources(plan);assert plan['execution_ready'] and plan['fit_infer_CPU_cap_seconds']==180
    runtime=read(checked(plan['runtime_contract']));assert runtime['status']=='PASS_synthetic_runtime_only'
    assert sha(runtime['native_library']['path'])==runtime['native_library']['sha256']
    root=Path(plan['output_dir']);root.mkdir(exist_ok=False)
    env=dict(os.environ);env.update(plan['environment'])
    state=dict(status='running',started_utc=now(),plan_sha256=sha(args.plan),jobs=[])
    (root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')
    start=time.monotonic()
    try:
        for f in range(4):
            left=180-(time.monotonic()-start);assert left>0
            invoke([plan['python'],str(Path(__file__).with_name('fit_fold.py')),'--plan',args.plan,'--fold',str(f)],left,root,f'fit{f}',state,env)
        write(root/'fit-stage.json',dict(plan_sha256=sha(args.plan),all_four_fits_complete=True,seconds=time.monotonic()-start,
                                      manifests=[ref(root/f'fold{f}/fitted/manifest.json') for f in range(4)]))
        invoke([plan['python'],str(Path(__file__).resolve()),'--plan',args.plan,'--finalize'],30,root,'finalize',state,env)
        state.update(status='complete',finished_utc=now(),result=ref(root/'result.json'))
    except BaseException as exc:state.update(status='failed_preserved',finished_utc=now(),failure=repr(exc));raise
    finally:(root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
