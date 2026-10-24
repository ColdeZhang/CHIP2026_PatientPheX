"""Frozen F1 theta transfer; zero fits/GPU, full original-dev only once or inherited."""
import argparse
from collections import Counter
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import ops

ROUTE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROUTE/'reuse'))
import common as c
import common_f0 as literal


def block(paths):
    forbidden={os.path.realpath(p) for p in paths}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) in forbidden:raise PermissionError('Gold blocked before freeze')
    sys.addaudithook(guard)


def load(path):
    p=json.loads(Path(path).read_text())
    assert p['execution_authorized'] and p['fit_calls']==p['GPU_calls']==0
    for r in p['sources'].values():c.checked(r)
    for r in p['ancestry_reviews'].values():c.checked(r)
    done=c.read_json(p['F0_complete']);assert done['status']=='COMPLETE_FIXED_CPU_CONTROL'
    fit=c.read_json(done['fit']);assert fit['theta']==p['theta']
    state=c.read_json(p['theta']);assert state['prepared_plan']==p['prepared_plan']
    c.checked(p['F0_backup_complete'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    return p,Path(p['output']),np.asarray(state['theta'],dtype=np.float64)


def features(p):
    r=p['refs'];u=c.universe(r['input'],16,36)
    G=literal.literal_predictions(r['G'],u);Q=literal.literal_predictions(r['Q'],u)
    S,_=literal.sampled_intersection(r['sampled_diagnostics'],u,64)
    streams={'S1':set(),'S2':set()}
    for row in c.json_lines(r['sampled_diagnostics']):
        assert row['stream'] in streams
        if row['whole_row_abstention']:
            assert row['relations']==[];continue
        for key in row['relations']:streams[row['stream']].update(literal.keys(key,u))
    assert S==streams['S1']&streams['S2']
    keys=sorted(G|S|Q);bits=[];x=[]
    for k in keys:
        bits.append([int(k in G),int(k in S),int(k in Q)])
        x.append([1.,math.log(len(u[k[0]])),float((k in streams['S1'])!=(k in streams['S2']))])
    C=literal.literal_predictions(r['C'],u);B=literal.literal_predictions(r['B901'],u)
    assert literal.literal_predictions(r['S_protected'],u)==B|S
    b3=literal.literal_predictions(r['B3'],u);b4=literal.literal_predictions(r['B4'],u)
    return u,keys,np.asarray(x),np.asarray(bits),dict(G=G,S=S,Q=Q,C=C,B=B,B3=b3,B4=b4)


def contract(p,out,theta,path):
    block(p['blocked_gold_paths']);u,keys,x,b,sets=features(p)
    zero=ops.selected(np.zeros(6),x,b);majority={k for k,v in zip(keys,zero) if v}
    assert sets['B']|majority==sets['B3']
    assert sets['B']|majority|(sets['C']&set(keys))==sets['B4']
    additions={k:len(sets[k]-sets['B']) for k in ['G','S','Q','C']}
    assert additions==p['expected_source_additions'] and len(sets['B'])==416
    c.write_json(out/'contract.json',dict(status='PASS_FULL64_PURE_S_AND_ZERO_THETA_RECONSTRUCT_B4',at_utc=c.now(),plan=c.ref(path),
        theta=p['theta'],source_counts={k:len(v) for k,v in sets.items()},source_additions=additions,
        rows=len(keys),documents=16,patients=36,given_patient_count_histogram=dict(Counter(len(v) for v in u.values())),
        F1_training_max_patients=7,documents_above_training_patient_count=sum(len(v)>7 for v in u.values()),
        sample_disagreement=int(x[:,2].sum()),zero_added=0,zero_deleted=0,gold_read=False,fit_calls=0,v2_calls=0))


def assemble(p,out,theta,path):
    block(p['blocked_gold_paths']);contract=json.loads((out/'contract.json').read_text());assert contract['plan']==c.ref(path)
    u,keys,x,b,sets=features(p);prob=ops.probabilities(theta,x,b)[0]
    gate={k for k,v in zip(keys,prob) if v>=.5};final=sets['B']|gate|(sets['C']&set(keys))
    scores=c.write_bytes(out/'scores.jsonl',b''.join((json.dumps(dict(key=k,context=xx.tolist(),bits=bb.tolist(),probability=float(v)),separators=(',',':'))+'\n').encode() for k,xx,bb,v in zip(keys,x,b,prob)))
    inherited=next((name for name in ['B3','B4'] if final==sets[name]),None)
    if inherited:payload=c.checked(p['refs'][inherited]).read_bytes()
    else:payload,detail=c.append_relations(p['refs']['B901'],u,final-sets['B']);assert detail['entities']==1061
    prediction=c.write_bytes(out/'prediction.jsonl',payload);assert literal.literal_predictions(prediction,u)==final
    ep=c.checked(p['refs']['evaluator']);c.checked(p['refs']['validator']);sys.path.insert(0,str(ep.parent))
    va=importlib.import_module('validate_predictions');inp=va.load_jsonl(c.checked(p['refs']['input']));rows=va.load_jsonl(c.checked(prediction));base=va.load_jsonl(c.checked(p['refs']['B901']))
    assert len(rows)==len(inp)==16 and sum(len(d['patient']) for d in inp)==36 and sum(len(d['entities']) for d in rows)==1061
    by={d['pmc_id']:d for d in base}
    for d in rows:
        original=by[d['pmc_id']]
        assert {k:v for k,v in d.items() if k!='association'}=={k:v for k,v in original.items() if k!='association'}
        assert [a['patient_id'] for a in d['association']]==[a['patient_id'] for a in original['association']]
        for new,old in zip(d['association'],original['association']):
            assert new['phenotype'][:len(old['phenotype'])]==old['phenotype']
            assert {k:v for k,v in new.items() if k!='phenotype'}=={k:v for k,v in old.items() if k!='phenotype'}
    valid=va.validate_predictions(inp,rows,va.read_hpo(c.checked(p['refs']['hpo'])))
    validation=c.write_json(out/'validation.json',valid);assert valid['valid']
    changes={name:dict(added=sorted(final-sets[name]),deleted=sorted(sets[name]-final)) for name in ['B3','B4']}
    c.write_json(out/'freeze.json',dict(status='FULL_ORIGINALDEV_FROZEN_BEFORE_GOLD',at_utc=c.now(),plan=c.ref(path),
        theta=p['theta'],prediction=prediction,validation=validation,scores=scores,contract=c.ref(out/'contract.json'),
        inherited_original_file=inherited,inherited_byteexact=bool(inherited),changes=changes,final_relations=len(final),
        protected_901_relations=416,entities=1061,patients=36,documents=16,
        all_B901_relations_protected=sets['B']<=final,nonassociation_objects_exact=True,original_association_prefix_exact=True,
        gold_read=False,fit_calls=0,v2_calls=0))


def quality(p,out,theta,path):
    fr=json.loads((out/'freeze.json').read_text());assert fr['plan']==c.ref(path) and not fr['gold_read'] and fr['v2_calls']==0
    for name in ['prediction','validation','scores','contract']:c.checked(fr[name])
    prior={n:c.read_json(p['refs'][n+'_metrics'])['metrics'] for n in ['B3','B4']}
    assert prior['B3']['Score']==p['best_score'] and prior['B4']['Score']==p['old_B4_score']
    inherited=fr['inherited_original_file'];diagnosis=None
    if inherited:
        assert c.checked(fr['prediction']).read_bytes()==c.checked(p['refs'][inherited]).read_bytes()
        metrics=prior[inherited];new_v2=0;metrics_ref=p['refs'][inherited+'_metrics']
    else:
        c.write_json(out/'evaluation-intent.json',dict(at_utc=c.now(),freeze=c.ref(out/'freeze.json'),new_v2=1,old_v2=0))
        ep=c.checked(p['refs']['evaluator']);c.checked(p['refs']['validator']);sys.path.insert(0,str(ep.parent))
        ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions');assert ev.VERSION=='patientphex-public-approx-v2'
        gold=va.load_jsonl(c.checked(p['refs']['gold']));pred=va.load_jsonl(c.checked(fr['prediction']))
        report=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text')
        metrics_ref=c.write_json(out/'metrics.json',report);metrics=report['metrics'];new_v2=1
        true={(d['pmc_id'],pid,h) for d in gold for pid,terms in ev.associations(d).items() for h in terms}
        diagnosis={}
        for name,change in fr['changes'].items():
            added=set(map(tuple,change['added']));deleted=set(map(tuple,change['deleted']))
            diagnosis[name]=dict(added=len(added),added_TP=len(added&true),added_FP=len(added-true),deleted=len(deleted),deleted_TP=len(deleted&true),deleted_FP=len(deleted-true))
    assert metrics['menF1']==prior['B3']['menF1'] and metrics['docF1']==prior['B3']['docF1']
    c.write_json(out/'result.json',dict(status='COMPLETE_FIXED_THETA_ORIGINALDEV_TRANSFER',at_utc=c.now(),plan=c.ref(path),
        freeze=c.ref(out/'freeze.json'),prediction=fr['prediction'],theta=p['theta'],metrics=metrics,metrics_file=metrics_ref,
        comparisons=prior,delta_vs_best={k:metrics[k]-v for k,v in prior['B3'].items()},
        retain=metrics['Score']>p['best_score'],target_08_reached=metrics['Score']>=.8,
        inherited_original_file=inherited,incremental_diagnosis=diagnosis,new_v2=new_v2,old_v2=0,new_fit=0,GPU_calls=0,
        originaldev_development_not_blind=True,gate='strictly exceed current B3 best, not weaker B4',
        F0_limitation='tiny F0 macro gain only redistributes two false positives at9patients; no extra TP and no evidence new disagreement drives it'))


def drive(path):
    p=json.loads(Path(path).read_text());out=c.mkdir_new(p['output']);environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    c.write_json(out/'driver-started.json',dict(at_utc=c.now(),plan=c.ref(path),pid=os.getpid()))
    for mode in ['contract','assemble','quality']:
        cmd=['timeout','--signal=TERM','--kill-after=5','120',sys.executable,str(Path(__file__).resolve()),'--plan',str(Path(path).resolve()),mode]
        c.write_json(out/(mode+'-intent.json'),dict(at_utc=c.now(),command=cmd));start=time.monotonic()
        with (out/(mode+'.stdout')).open('xb') as so,(out/(mode+'.stderr')).open('xb') as se:
            process=subprocess.run(cmd,env=environment,stdout=so,stderr=se);so.flush();os.fsync(so.fileno());se.flush();os.fsync(se.fileno())
        c.write_json(out/(mode+'-process.json'),dict(at_utc=c.now(),return_code=process.returncode,seconds=time.monotonic()-start))
        if process.returncode:raise SystemExit(process.returncode)
    c.write_json(out/'complete.json',dict(status='COMPLETE_NO_REFIT_ORIGINALDEV_TRANSFER',at_utc=c.now(),plan=c.ref(path),result=c.ref(out/'result.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('mode',choices=['drive','contract','assemble','quality']);a=ap.parse_args()
    if a.mode=='drive':drive(a.plan)
    else:
        p,out,theta=load(a.plan);dict(contract=contract,assemble=assemble,quality=quality)[a.mode](p,out,theta,a.plan)
