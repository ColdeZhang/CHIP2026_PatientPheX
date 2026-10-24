"""One fixed CPU fit, then frozen parameters -> full F0 -> at most one v2.

Mode isolation and exclusive artifacts prevent a restart from silently refitting.
The original prepared plan remains immutable; activation binds actual source.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import ops

ROUTE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROUTE/'reuse'))
import common as c
import common_f0 as f0


def block(paths):
    forbidden={os.path.realpath(p) for p in paths}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) in forbidden:
                raise PermissionError('Gold access forbidden in this phase')
    sys.addaudithook(guard)


def load(activation_path):
    a=json.loads(Path(activation_path).read_text())
    assert a['execution_authorized'] is True and a['fit_calls_maximum']==1
    for r in a['sources'].values():c.checked(r)
    p=c.read_json(a['prepared_plan'])
    assert p['execution_authorized'] is False and p['parameters']==6
    assert p['training']['optimizer_options']==dict(maxiter=500,maxfun=10000,maxls=20,gtol=1e-8,ftol=1e-12)
    assert p['threshold']==.5 and p['threshold_operator']=='>='
    assert p['training']['prior_precision']==ops.PRIOR_PRECISION==1.
    assert ops.EPSILON==1e-4 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for r in p['source'].values():c.checked(r)
    for r in p['preparation_evidence'].values():c.checked(r)
    c.checked(p['inherited_C_and_fixed_B4_review'])
    return a,p,Path(p['output'])


def features(p,fold):
    s=p['training'] if fold==1 else p['endpoint'];module=c if fold==1 else f0
    candidate_plan=dict(input=s['input'],raw_sources=s['raw_sources'],documents=16,
                        patients=39 if fold==1 else 47,sample_rows_per_seed=80 if fold==1 else 58)
    u,rows,counts=module.candidates(candidate_plan)
    streams={name:set() for name in ['S1','S2']}
    for r in c.json_lines(s['raw_sources']['sampled_diagnostics']):
        if r['stream'] not in streams:continue
        if r['whole_row_abstention']:
            assert r['relations']==[]
            continue
        for k in r['relations']:
            if fold==1:streams[r['stream']].add(c.verify_key(k,u))
            else:streams[r['stream']].update(f0.keys(k,u))
    x=[];bits=[];keys=[]
    for r in rows:
        k=tuple(r['key']);b=r['features'];n=int(k in streams['S1'])+int(k in streams['S2'])
        assert b[1]==int(n==2)
        x.append([1.,math.log(len(u[k[0]])),float(n==1)]);bits.append(b);keys.append(k)
    assert len(keys)==(561 if fold==1 else 431)
    assert len(keys)==len(set(keys)) and all(any(b) for b in bits)
    return u,keys,np.asarray(x,dtype=np.float64),np.asarray(bits,dtype=np.float64),counts


def write_rows(path,rows):
    return c.write_bytes(path,b''.join((json.dumps(r,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows))


def fit(a,p,root,activation_path):
    block([p['endpoint']['gold']['path']])
    out=c.mkdir_new(root/'fit');start=time.monotonic()
    c.write_json(out/'started.json',dict(at_utc=c.now(),activation=c.ref(activation_path),pid=os.getpid(),
                                       F1_label_access_authorized=True,F0_gold_access=False))
    try:
        import scipy
        from scipy.optimize import minimize
        from threadpoolctl import threadpool_limits,threadpool_info
        assert scipy.__version__==a['runtime']['scipy'] and np.__version__==a['runtime']['numpy']
        u,keys,x,b,counts=features(p,1)
        fr=write_rows(out/'features.jsonl',[dict(key=k,context=xx.tolist(),bits=bb.tolist()) for k,xx,bb in zip(keys,x,b)])
        feature_freeze=c.write_json(out/'features-freeze.json',dict(at_utc=c.now(),features=fr,counts=counts,
            source=p['training']['raw_sources'],gold_read=False,baseline_read=False,C_read=False))
        c.write_json(out/'label-read-started.json',dict(at_utc=c.now(),feature_freeze=feature_freeze,labels=p['training']['labels'],scope='F1 meta-training only'))
        gold=c.gold_set(p['training']['labels'],u)
        y=np.asarray([int(k in gold) for k in keys],dtype=np.float64);docs=np.asarray([k[0] for k in keys])
        assert len(set(docs))==16 and set(y)=={0.,1.}
        initial=np.asarray(p['initial_parameters'],dtype=np.float64)
        initial_loss=ops.objective(initial,x,b,y,docs)[0]
        c.write_json(out/'fit-intent.json',dict(at_utc=c.now(),fit_call_index=1,optimizer=p['training']['optimizer'],
            options=p['training']['optimizer_options'],initial=initial.tolist(),feature_freeze=feature_freeze,
            target_float64_sha256=hashlib.sha256(y.tobytes()).hexdigest(),documents=16,rows=len(keys),
            positives=int(y.sum()),loss_initial=initial_loss,weighting=p['training']['objective']))
        t=time.monotonic()
        with threadpool_limits(limits=1):
            result=minimize(ops.objective,initial,args=(x,b,y,docs),method='L-BFGS-B',jac=True,
                            options=p['training']['optimizer_options'])  # Sole optimizer call.
            thread_info=threadpool_info()
        seconds=time.monotonic()-t
        convergence=dict(at_utc=c.now(),success=bool(result.success),status=int(result.status),message=str(result.message),
            theta=result.x.tolist(),objective=float(result.fun),gradient=result.jac.tolist(),iterations=int(result.nit),
            function_evaluations=int(result.nfev),gradient_evaluations=int(result.njev),fit_seconds=seconds,
            finite=bool(np.isfinite(result.x).all() and np.isfinite(result.fun) and np.isfinite(result.jac).all()),
            threads=thread_info)
        c.write_json(out/'convergence.json',convergence)  # Evidence before convergence assertion.
        assert convergence['finite'] and convergence['success'],'Fixed optimizer did not converge; stop without retry'
        theta_ref=c.write_json(out/'theta.json',dict(theta=result.x.tolist(),shape=[2,3],parameter_order=['G_intercept','G_log_patients','G_disagreement','S_intercept','S_log_patients','S_disagreement'],
            activation=c.ref(activation_path),prepared_plan=a['prepared_plan'],objective=p['training']['objective'],
            threshold=.5,threshold_operator='>=',source=c.ref(ROUTE/'ops.py')))
        theta=np.asarray(c.read_json(theta_ref)['theta'],dtype=np.float64)
        assert np.array_equal(theta,result.x)
        probabilities=ops.probabilities(result.x,x,b)[0]
        assert np.array_equal(probabilities,ops.probabilities(theta,x,b)[0])
        pr=write_rows(out/'training-probabilities.jsonl',[dict(key=k,probability=float(v)) for k,v in zip(keys,probabilities)])
        c.write_json(out/'complete.json',dict(status='ONE_FIT_COMPLETE_PARAMETERS_FROZEN_NO_F0',at_utc=c.now(),
            activation=c.ref(activation_path),theta=theta_ref,features=fr,feature_freeze=feature_freeze,
            probabilities=pr,convergence=c.ref(out/'convergence.json'),counts=counts,documents=16,
            rows=len(keys),positives=int(y.sum()),negative=int(len(y)-y.sum()),initial_loss=initial_loss,
            final_loss=float(result.fun),fit_seconds=seconds,wall_seconds=time.monotonic()-start,
            reload_parameters_and_probabilities_exact=True,fit_calls=1,v2_calls=0,F0_gold_read=False,
            scope='Descriptive meta-training only, not independent validation'))
    except BaseException as error:
        c.write_json(out/'failure.json',dict(at_utc=c.now(),type=type(error).__name__,message=str(error),
            seconds=time.monotonic()-start,rerun_or_solver_change_forbidden=True))
        raise


def imports(endpoint):
    ep=c.checked(endpoint['evaluator']);c.checked(endpoint['validator']);sys.path.insert(0,str(ep.parent))
    ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions')
    assert ev.VERSION=='patientphex-public-approx-v2'
    return ev,va


def assemble(a,p,root,activation_path):
    block([p['training']['labels']['path'],p['endpoint']['gold']['path']])
    out=c.mkdir_new(root/'F0');e=p['endpoint']
    done=json.loads((root/'fit/complete.json').read_text());assert done['activation']==c.ref(activation_path)
    theta=c.read_json(done['theta']);assert theta['prepared_plan']==a['prepared_plan']
    c.write_json(out/'started.json',dict(at_utc=c.now(),activation=c.ref(activation_path),fit_complete=c.ref(root/'fit/complete.json'),theta=done['theta'],gold_read=False))
    u,keys,x,b,counts=features(p,0);probs=ops.probabilities(theta['theta'],x,b)[0]
    selected={k for k,v in zip(keys,probs) if v>=.5};union=set(keys)
    scores=write_rows(out/'all-raw-union-scores.jsonl',[dict(key=k,context=xx.tolist(),bits=bb.tolist(),probability=float(v),selected=bool(v>=.5)) for k,xx,bb,v in zip(keys,x,b,probs)])
    C=f0.literal_predictions(e['coded'],u);fixed=C&union
    # Only after complete raw selection do we open the protected B0 and old B4.
    B=f0.literal_predictions(e['protected_base'],u);old=f0.literal_predictions(e['baseline_prediction'],u)
    final=B|selected|fixed;added=final-old;deleted=old-final
    assert len(B)==315 and len(old)==343
    if not added and not deleted:
        payload=c.checked(e['baseline_prediction']).read_bytes();preservation='inherit original B4 bytes at zero semantic delta'
    else:
        payload,detail=c.append_relations(e['protected_base'],u,(selected|fixed)-B)
        assert detail['entities']==947;preservation='B0 nonassociation bytes and complete association prefix retained'
    pred=c.write_bytes(out/'prediction.jsonl',payload)
    assert f0.literal_predictions(pred,u)==final
    ev,va=imports(e);inp=va.load_jsonl(c.checked(e['input']));rows=va.load_jsonl(c.checked(pred));base=va.load_jsonl(c.checked(e['protected_base']))
    assert len(rows)==len(inp)==16 and sum(len(d['patient']) for d in inp)==47
    assert sum(len(d['entities']) for d in rows)==947
    by={d['pmc_id']:d for d in base}
    for d in rows:
        original=by[d['pmc_id']]
        assert {k:v for k,v in d.items() if k!='association'}=={k:v for k,v in original.items() if k!='association'}
        assert [a['patient_id'] for a in d['association']]==[a['patient_id'] for a in original['association']]
        for new,old_a in zip(d['association'],original['association']):
            assert {k:v for k,v in new.items() if k!='phenotype'}=={k:v for k,v in old_a.items() if k!='phenotype'}
    valid=va.validate_predictions(inp,rows,va.read_hpo(c.checked(e['hpo'])))
    validation=c.write_json(out/'validation.json',valid);assert valid['valid']
    c.write_json(out/'freeze.json',dict(status='FULL_F0_FROZEN_BEFORE_GOLD',at_utc=c.now(),activation=c.ref(activation_path),
        fit_complete=c.ref(root/'fit/complete.json'),theta=done['theta'],prediction=pred,validation=validation,
        scores=scores,counts=counts,documents=16,patients=47,entities=947,base_relations=len(B),final_relations=len(final),
        fixed_C_intersection=len(fixed),raw_gate_selected=len(selected),added=sorted(added),deleted=sorted(deleted),
        original_B4_inherited_byteexact=not added and not deleted,preservation=preservation,
        all_B0_relations_protected=B<=final,nonassociation_objects_exact=True,gold_read=False,v2_calls=0))


def quality(a,p,root,activation_path):
    e=p['endpoint'];out=c.mkdir_new(root/'quality');fr=json.loads((root/'F0/freeze.json').read_text())
    assert fr['activation']==c.ref(activation_path) and not fr['gold_read'] and fr['v2_calls']==0
    for k in ['prediction','validation','scores','theta','fit_complete']:c.checked(fr[k])
    old_metrics=c.read_json(e['baseline_metrics'])['metrics'];assert old_metrics['Score']==e['baseline_score']
    intent=dict(at_utc=c.now(),freeze=c.ref(root/'F0/freeze.json'),activation=c.ref(activation_path))
    added=set(map(tuple,fr['added']));deleted=set(map(tuple,fr['deleted']))
    if not added and not deleted:
        assert c.checked(fr['prediction']).read_bytes()==c.checked(e['baseline_prediction']).read_bytes()
        c.write_json(out/'result.json',dict(status='ZERO_SEMANTIC_DELTA_INHERITED_B4_CLOSED',**intent,
            prediction=fr['prediction'],metrics=old_metrics,baseline=old_metrics,
            delta={k:0. for k in old_metrics},retain=False,new_v2=0,old_v2=0,gold_read=False,
            added=0,deleted=0,fit_calls_total=1,original_dev_read=False))
        return
    c.write_json(out/'evaluation-intent.json',dict(**intent,new_v2_call_index=1,old_v2=0))
    ev,va=imports(e);gold=va.load_jsonl(c.checked(e['gold']));prediction=va.load_jsonl(c.checked(fr['prediction']))
    report=ev.evaluate_report(gold,prediction,body_scope='all',no_id_doc='text')  # Sole new v2.
    metrics=c.write_json(out/'metrics.json',report);new=report['metrics'];delta={k:new[k]-v for k,v in old_metrics.items()}
    assert delta['menF1']==delta['docF1']==0
    gold_pairs={(d['pmc_id'],pid,t) for d in gold for pid,ts in ev.associations(d).items() for t in ts}
    c.write_json(out/'result.json',dict(status='COMPLETE_SINGLE_FIXED_F0_CONTROL',**intent,prediction=fr['prediction'],
        metrics_file=metrics,metrics=new,baseline=old_metrics,delta=delta,retain=new['Score']>old_metrics['Score'],
        added=len(added),added_TP=len(added&gold_pairs),added_FP=len(added-gold_pairs),
        deleted=len(deleted),deleted_TP=len(deleted&gold_pairs),deleted_FP=len(deleted-gold_pairs),
        new_v2=1,old_v2=0,fit_calls_total=1,F1_v2=0,GPU_calls=0,original_dev_read=False,
        development_not_blind=True,gate='strictly greater than fixed B4 .7281956289113009'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--activation',required=True);ap.add_argument('mode',choices=['fit','assemble','quality']);args=ap.parse_args()
    a,p,root=load(args.activation)
    dict(fit=fit,assemble=assemble,quality=quality)[args.mode](a,p,root,args.activation)
