"""One fixed B507 union (prefix-name AND grounded) endpoint, byte preserving."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import time

def load(path):
    p=json.loads(Path(path).read_text());r=p['refs'];spec=importlib.util.spec_from_file_location('_prefix_helpers',r['helper']['path']);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    assert H.ref(r['helper']['path'])==r['helper'];gate=H.gold_guard(p)
    assert p['status']=='BOUND_FIXED_PREFIX_AGREEMENT' and p['formula']=='B507 UNION (prefix_names INTERSECT all_grounded282)'
    assert p['quality_source']==H.ref(__file__) and p['baseline_score']==.7010873324879443
    for item in r.values():H.checked(item)
    for item in p['actual'].values():H.checked(item)
    actual=H.read(p['actual']['complete']['path'])
    assert actual['status']=='COMPLETE_ORIGINAL39_NATIVE_PREFIX_POLICY' and actual['rows']==39
    assert actual['strict27_original_lines_byteexact'] and actual['NN_calls']==actual['quality_calls']==0
    assert actual['counts']['strict_unchanged']==27 and actual['counts']['caps']==12
    return p,H,gate

def assemble(path):
    p,H,gate=load(path);r=p['refs'];out=Path(p['quality_output']);out.mkdir(parents=True,exist_ok=False)
    lines,base,bids=H.rows(r['baseline']['path']);_,inputs,iids=H.rows(r['input']['path']);_,grounded,gids=H.rows(r['grounded']['path'])
    assert bids==iids==gids and len(base)==16 and sum(len(d['entities']) for d in base)==1189
    old=H.jsonl(r['original_canonical']['path']);raw=Path(p['actual']['prefix']['path']).read_text().splitlines(keepends=True)
    original_lines=Path(r['original_canonical']['path']).read_text().splitlines(keepends=True);rows=[json.loads(x) for x in raw];assert len(rows)==len(old)==39
    for before,after,line,oldline in zip(old,rows,raw,original_lines):
        for k in ['row_id','pmc_id','target_patient_id','index','hit_token_limit','natural_eos']:assert before[k]==after[k]
        if before['strict_complete']:assert line==oldline and after==before
        else:assert after['prefix_items_accepted'] and not after['strict_complete'] and after['hit_token_limit']
    supplied={(str(x['pmc_id']),x['target_patient_id']):x for x in rows}
    expected={(str(d['pmc_id']),patient['patient_id']) for d in inputs for patient in d['patient']};assert len(supplied)==39 and set(supplied)==expected
    allowed=set(H.read(r['public_names']['path']).values());spans,mask=H.byte_helpers(r['byte_helper']);pred=[];head=[];old_count=grounded_count=prefix_count=0;added=[];agreements=[]
    for line,b,g,inp in zip(lines,base,grounded,inputs):
        pids=[x['patient_id'] for x in inp['patient']]
        assert [x['patient_id'] for x in b['association']]==[x['patient_id'] for x in g['association']]==pids
        assert {k:v for k,v in b.items() if k not in ['entities','association']}=={k:v for k,v in g.items() if k not in ['entities','association']}
        union={};names={}
        for ba,ga in zip(b['association'],g['association']):
            pid=ba['patient_id'];extra=supplied[str(b['pmc_id']),pid]['hpo_ids'];assert len(extra)==len(set(extra)) and set(extra)<=allowed
            original=ba['phenotype'];gr=ga['phenotype'];assert len(original)==len(set(original)) and len(gr)==len(set(gr))
            agree=set(extra)&set(gr);new=sorted(agree-set(original));union[pid]=original+new;names[pid]=sorted(extra)
            added.extend((str(b['pmc_id']),pid,x) for x in new);agreements.extend((str(b['pmc_id']),pid,x) for x in sorted(agree))
            old_count+=len(original);grounded_count+=len(gr);prefix_count+=len(extra)
        u=H.replace_phenotypes(line,union,spans,mask);n=H.replace_phenotypes(line,names,spans,mask)
        assert mask(u)==mask(n)==mask(line);pred.append(u);head.append(n)
    assert old_count==507 and grounded_count==282
    H.publish(out/'prediction.jsonl',''.join(pred).encode());H.publish(out/'prefix-source-unscored.jsonl',''.join(head).encode())
    if not added:assert (out/'prediction.jsonl').read_bytes()==Path(r['baseline']['path']).read_bytes()
    validator=H.load_module('validate_predictions',r['validator']);v=validator.validate_predictions(inputs,[json.loads(x) for x in pred],validator.read_hpo(r['hpo']['path']))
    H.save(out/'format.json',v);assert v['valid'] and v['error_count']==0 and not gate['enabled']
    H.save(out/'freeze.json',dict(plan=H.ref(path),source=H.ref(__file__),prediction=H.ref(out/'prediction.jsonl'),
        prefix_source=H.ref(out/'prefix-source-unscored.jsonl'),format=H.ref(out/'format.json'),documents=16,patients=39,entities=1189,
        baseline_relations=507,prefix_relations=prefix_count,grounded_relations=282,agreement_relations=len(agreements),
        added=len(added),added_pairs=added,final_relations=507+len(added),all_baseline_nonrelation_bytes_exact=True,
        all_baseline_relation_prefix_exact=True,zero_added_byteexact=not added,gold_read=False,new_v2=0,NN_calls=0))
    print(json.dumps(H.ref(out/'freeze.json')))

def score(path):
    p,H,gate=load(path);r=p['refs'];out=Path(p['quality_output']);f=H.read(out/'freeze.json');assert f['plan']==H.ref(path) and f['source']==H.ref(__file__)
    H.checked(f['prediction']);H.checked(f['format']);H.checked(f['prefix_source']);prior=H.read(r['baseline_metrics']['path'])['metrics'];assert prior['Score']==p['baseline_score']
    ev=out/'evaluation';ev.mkdir(exist_ok=False)
    if not f['added']:
        assert Path(f['prediction']['path']).read_bytes()==Path(r['baseline']['path']).read_bytes() and not gate['enabled']
        H.save(ev/'result.json',dict(status='BYTEEXACT_NO_ADDITIONS_ZERO_GOLD_ZERO_V2',baseline=prior,keep=False,new_v2=0,old_v2=0,single_head_v2=0,NN_calls=0,freeze=H.ref(out/'freeze.json')));return
    H.save(ev/'before-gold.json',dict(freeze=H.ref(out/'freeze.json'),gold_not_yet_opened=True,new_v2=1,old_v2=0))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Only CPU score30s')));signal.alarm(30)
    try:
        validator=H.load_module('validate_predictions',r['validator']);evaluator=H.load_module('_prefix_v2',r['evaluator']);assert evaluator.VERSION=='patientphex-public-approx-v2' and evaluator.validate_predictions is validator.validate_predictions
        gate['enabled']=True;H.checked(p['gold']);gold=validator.load_jsonl(p['gold']['path']);pred=validator.load_jsonl(f['prediction']['path'])
        H.save(ev/'call-start.json',dict(freeze=H.ref(out/'freeze.json'),new_v2=1,old_v2=0,body_scope='all',no_id_doc='text'))
        metrics=evaluator.evaluate_report(gold,pred,'all','text');H.save(ev/'metrics.json',metrics);delta={k:metrics['metrics'][k]-prior[k] for k in H.METRICS};assert delta['menF1']==delta['docF1']==0
        before=H.read(r['baseline_metrics']['path'])['counts']['relation'];after=metrics['counts']['relation']
        H.save(ev/'result.json',dict(status='COMPLETE_UNIQUE_PREFIX_AGREEMENT_F1',metrics=metrics['metrics'],baseline=prior,delta=delta,keep=delta['Score']>0,
            added=f['added'],added_TP=after['tp']-before['tp'],added_FP=after['fp']-before['fp'],new_v2=1,old_v2=0,single_head_v2=0,NN_calls=0,
            development_choice_after_observed_caps=True,blind_test=False,original_dev_read=False,freeze=H.ref(out/'freeze.json'),metrics_ref=H.ref(ev/'metrics.json')))
        print(json.dumps(H.ref(ev/'result.json')))
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','score']);p.add_argument('--plan',required=True);a=p.parse_args();globals()[a.stage](a.plan)
