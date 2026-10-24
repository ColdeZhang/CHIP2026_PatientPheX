"""One fixed open-concept agreement endpoint; assemble and score are separate."""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def ref(path):
    p=Path(path).resolve();b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def checked(r):
    assert ref(r['path'])==r,r['path']
    return Path(r['path'])


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def run(path,mode):
    p=json.loads(Path(path).read_text());r=p['refs'];out=Path(p['output'])
    assert p['fold'] in (0,1) and p['formula']=='baseline UNION (names INTERSECT grounded)'
    assert r['source']==ref(__file__)
    checked(r['decision']);ep=checked(r['evaluator']);checked(r['validator'])
    sys.path.insert(0,str(ep.parent))
    ev=importlib.import_module('evaluate');va=importlib.import_module('validate_predictions')
    assert ev.VERSION=='patientphex-public-approx-v2'

    def pairs(rows):
        return {(d['pmc_id'],pid,h) for d in rows for pid,hs in ev.associations(d).items() for h in hs}

    if mode=='assemble':
        forbidden=os.path.realpath(r['gold']['path'])
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                if os.path.realpath(os.fsdecode(args[0]))==forbidden:
                    raise PermissionError('Held gold blocked before complete freeze')
        sys.addaudithook(guard)
        out.mkdir(parents=True,exist_ok=False)
        put(out/'started.json',dict(at_utc=now(),plan=ref(path)))
        base=va.load_jsonl(checked(r['baseline']));inputs=va.load_jsonl(checked(r['input']))
        by={d['pmc_id']:d for d in base};old=pairs(base)
        assert len(base)==len(by)==len(inputs)==16 and set(by)=={d['pmc_id'] for d in inputs}
        assert len(old)==p['baseline_relations']
        assert sum(len(d['entities']) for d in base)==p['entities']
        assert sum(len(d['association']) for d in base)==p['patients']
        sets={}
        for key in ('names','grounded'):
            source=va.load_jsonl(checked(r[key]))
            assert len(source)==16 and {d['pmc_id'] for d in source}==set(by)
            for d in source:
                b=by[d['pmc_id']]
                assert {k:v for k,v in d.items() if k not in ('entities','association')}=={k:v for k,v in b.items() if k not in ('entities','association')}
                assert ev.associations(d).keys()==ev.associations(b).keys()
            sets[key]=pairs(source)
        added=(sets['names']&sets['grounded'])-old
        predicted=copy.deepcopy(base)
        for d in predicted:
            for a in d['association']:
                a['phenotype'].extend(sorted(h for doc,pid,h in added if doc==d['pmc_id'] and pid==a['patient_id']))
        assert pairs(predicted)==old|added
        for b,d in zip(base,predicted):
            assert {k:v for k,v in b.items() if k!='association'}=={k:v for k,v in d.items() if k!='association'}
            for a,z in zip(b['association'],d['association']):
                assert z['phenotype'][:len(a['phenotype'])]==a['phenotype']
                assert {k:v for k,v in a.items() if k!='phenotype'}=={k:v for k,v in z.items() if k!='phenotype'}
        valid=va.validate_predictions(inputs,predicted,va.read_hpo(checked(r['hpo'])))
        put(out/'format.json',valid);assert valid['valid']
        pp=out/'prediction.jsonl'
        with pp.open('x') as f:
            for d in predicted:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
            f.flush();os.fsync(f.fileno())
        put(out/'freeze.json',dict(at_utc=now(),plan=ref(path),prediction=ref(pp),format=ref(out/'format.json'),
            added_pairs=sorted(added),added=len(added),source_relations={k:len(v) for k,v in sets.items()},
            baseline_relations=len(old),final_relations=len(old|added),nonassociation_objects_exact=True,
            baseline_relation_lists_prefix_exact=True,gold_read=False,NN_calls=0,fit_calls=0))
    else:
        freeze=json.loads((out/'freeze.json').read_text())
        assert freeze['plan']==ref(path)
        checked(freeze['prediction']);checked(freeze['format'])
        prior=json.loads(checked(r['baseline_metrics']).read_text())['metrics']
        assert prior['Score']==p['baseline_score']
        if not freeze['added']:
            put(out/'result.json',dict(status='EXACT_RELATION_SET_EQUIVALENT_NO_NEW_SCORE',at_utc=now(),
                baseline=prior,retain=False,new_v2=0,old_v2=0,freeze=ref(out/'freeze.json')))
            return
        put(out/'evaluation-intent.json',dict(at_utc=now(),freeze=ref(out/'freeze.json'),new_v2=1,old_v2=0))
        gold=va.load_jsonl(checked(r['gold']));pred=va.load_jsonl(freeze['prediction']['path'])
        metrics=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text')
        put(out/'metrics.json',metrics)
        delta={k:metrics['metrics'][k]-v for k,v in prior.items()}
        assert delta['menF1']==delta['docF1']==0
        added=set(map(tuple,freeze['added_pairs']));tp=len(added&pairs(gold))
        put(out/'result.json',dict(status='COMPLETE_ONE_FIXED_FOLD_ENDPOINT',at_utc=now(),fold=p['fold'],
            baseline=prior,metrics=metrics['metrics'],delta=delta,added=len(added),added_TP=tp,added_FP=len(added)-tp,
            retain=delta['Score']>0,new_v2=1,old_v2=0,NN_calls=0,fit_calls=0,development_not_blind=True,
            original_dev_read=False,freeze=ref(out/'freeze.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('mode',choices=['assemble','evaluate'])
    a=ap.parse_args();run(a.plan,a.mode)
