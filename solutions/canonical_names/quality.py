"""Complete39 strict output -> unchanged strong B union -> one frozen v2 call."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

def ref(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(item):assert ref(item['path'])==item;return Path(item['path'])
def load(path):
    p=json.loads(Path(path).read_text());checked(p['helper'])
    spec=importlib.util.spec_from_file_location('_canonical_quality_helpers',p['helper']['path']);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    gate=H.gold_guard(p)
    assert p['status']=='BOUND_COMPLETE39_FIXED_STRONG_UNION' and p['source']==ref(__file__)
    assert p['new_evaluator_calls']==1 and p['old_evaluator_calls']==0
    assert p['baseline_score']==.6863389812716092
    for item in p['references'].values():checked(item)
    actual=json.loads(checked(p['actual']['infer_complete']).read_text())
    assert actual['status']=='COMPLETE_ALL39_FRESH_CANONICAL_NAMES_NO_QUALITY' and actual['fresh_rows']==actual['rows']==39
    for item in p['actual'].values():checked(item)
    return p,H,gate

def assemble(path):
    p,H,gate=load(path);r=p['references'];out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    lines,base,bids=H.rows(r['baseline']['path']);_,inputs,iids=H.rows(r['input']['path'])
    assert bids==iids and len(base)==16 and sum(len(d['entities']) for d in base)==1189
    rows=H.jsonl(p['actual']['canonical']['path']);assert len(rows)==39
    expected={(str(d['pmc_id']),patient['patient_id']) for d in inputs for patient in d['patient']}
    supplied={(str(x['pmc_id']),x['target_patient_id']):x for x in rows};assert set(supplied)==expected and len(supplied)==39
    names=json.loads(checked(r['public_names']).read_text());allowed=set(names.values())
    spans,mask=H.byte_helpers(r['byte_helper']);union=[];newonly=[];old_count=new_count=added=0
    for line,b,inp in zip(lines,base,inputs):
        pids=[x['patient_id'] for x in inp['patient']];assert [a['patient_id'] for a in b['association']]==pids
        uv={};nv={}
        for a in b['association']:
            pid=a['patient_id'];row=supplied[str(b['pmc_id']),pid];extra=row['hpo_ids']
            assert isinstance(extra,list) and len(extra)==len(set(extra)) and set(extra)<=allowed
            assert row['all_generated_prefix_tokens_legal']
            if not row['strict_complete']:assert extra==[]
            old=a['phenotype'];assert len(old)==len(set(old));nv[pid]=sorted(extra)
            uv[pid]=old+[v for v in sorted(extra) if v not in set(old)]
            assert uv[pid][:len(old)]==old and set(extra)<=set(uv[pid])
            old_count+=len(old);new_count+=len(extra);added+=len(uv[pid])-len(old)
        u=H.replace_phenotypes(line,uv,spans,mask);n=H.replace_phenotypes(line,nv,spans,mask)
        assert mask(u)==mask(n)==mask(line);union.append(u);newonly.append(n)
    assert old_count==471
    H.publish(out/'prediction.jsonl',''.join(union).encode());H.publish(out/'new-only-unscored.jsonl',''.join(newonly).encode())
    validator=H.load_module('validate_predictions',r['validator'])
    validation=validator.validate_predictions(inputs,[json.loads(x) for x in union],validator.read_hpo(r['hpo']['path']))
    H.save(out/'validation.json',validation);assert validation['valid'] and validation['error_count']==0
    assert gate['enabled'] is False
    H.save(out/'freeze.json',dict(plan=ref(path),source=ref(__file__),prediction=ref(out/'prediction.jsonl'),
        new_only_unscored=ref(out/'new-only-unscored.jsonl'),validation=ref(out/'validation.json'),
        canonical=p['actual']['canonical'],documents=16,patients=39,entities=1189,baseline_relations=471,
        supplied_relations=new_count,appended=added,final_relations=471+added,
        all_baseline_prefix_and_nonrelation_bytes_exact=True,all_new_HPO_relations_used=True,
        failed_rows_retained=sum(not x['strict_complete'] for x in rows),gold_read_or_hashed=False,new_evaluator_calls=0))
    print(json.dumps(ref(out/'freeze.json')))

def score(path):
    start=time.monotonic();p,H,gate=load(path);r=p['references'];out=Path(p['output']);f=json.loads((out/'freeze.json').read_text())
    assert f['plan']==ref(path) and f['source']==ref(__file__)
    checked(f['prediction']);checked(f['validation']);checked(f['new_only_unscored'])
    assert f['all_baseline_prefix_and_nonrelation_bytes_exact'] and f['all_new_HPO_relations_used']
    ev=out/'evaluation';ev.mkdir(exist_ok=False)
    H.save(ev/'before-gold.json',dict(freeze=ref(out/'freeze.json'),gold_not_yet_opened=True,one_call_only=True))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Unique CPU score30s')));signal.alarm(30)
    try:
        validator=H.load_module('validate_predictions',r['validator']);evaluator=H.load_module('_canonical_v2',r['evaluator'])
        assert evaluator.VERSION=='patientphex-public-approx-v2' and evaluator.validate_predictions is validator.validate_predictions
        gate['enabled']=True;checked(p['gold']);gold=validator.load_jsonl(p['gold']['path'])
        _,pred,pids=H.rows(f['prediction']['path']);assert {str(d['pmc_id']) for d in gold}==set(pids)
        H.save(ev/'call-start.json',dict(prediction=f['prediction'],gold=p['gold'],evaluator=r['evaluator'],body_scope='all',no_id_doc='text',new_call=1,old_calls=0))
        metrics=evaluator.evaluate_report(gold,pred,'all','text')
        metrics.update(prediction_sha256=f['prediction']['sha256'],gold_sha256=p['gold']['sha256'])
        H.save(ev/'metrics.json',metrics);old=H.read(r['baseline_metrics']['path'])['metrics']
        assert old['Score']==p['baseline_score'];delta={k:metrics['metrics'][k]-old[k] for k in H.METRICS}
        assert delta['menF1']==delta['docF1']==0
        keep=metrics['metrics']['Score']>p['baseline_score']
        H.save(ev/'result.json',dict(status='COMPLETE_UNIQUE_FIXED_UNION_F1',metrics=metrics['metrics'],baseline=old,delta=delta,
            keep=keep,acceptance='Total Score strictly above B; micro/macro disclosed without extra gate',
            prediction=f['prediction'],metrics_ref=ref(ev/'metrics.json'),freeze=ref(out/'freeze.json'),
            new_evaluator_calls=1,old_evaluator_calls=0,single_head_evaluator_calls=0,known_F1_development=True,
            seconds=time.monotonic()-start,next='Root decides ancestry/F0' if keep else 'Close fixed recipe; no filtering or threshold sweep'))
        print(json.dumps(ref(ev/'result.json')))
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','score']);p.add_argument('--plan',required=True);a=p.parse_args();globals()[a.stage](a.plan)
