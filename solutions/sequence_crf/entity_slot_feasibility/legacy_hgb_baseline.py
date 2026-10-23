#!/usr/bin/env python3
"""One historical .4 HGB baseline on the already-scored strict train32/F1 pool.

Decode never reads gold or fits a model. A separate evaluation mode requires the
complete prediction freeze and evaluates this one new baseline only.
"""
from pathlib import Path
import argparse, copy, datetime, gzip, hashlib, json, math, signal, sys, time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p=Path(p); return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def load(p): return json.loads(Path(p).read_text())
def rows(p):
    p=Path(p); b=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return [json.loads(line) for line in b.splitlines()]
def save(p,x):
    with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def key(p,c): return str(p),c['offset'],c['length'],c['identifier']


def decode(candidates):
    selected=[]
    for c in sorted(candidates,key=lambda c:(-c['gate_probability'],-c['length'],c['offset'])):
        if c['gate_probability']<.4: continue
        if any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in selected): continue
        selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
    return sorted(selected,key=lambda e:e['offset'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['decode','evaluate'])
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--gold',type=Path)
    ap.add_argument('--scripts',type=Path);ap.add_argument('--hpo',type=Path);a=ap.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU stage >30s')));signal.alarm(30)
    tick=time.monotonic();b=a.directory;p=load(b/'plan.json');assert sha(__file__)==p['source_sha256']
    assert p['cut']==.4 and p['gate_cut']==.1
    for item in p['references'].values():assert sha(item['path'])==item['sha256']
    if a.mode=='decode':
        assert a.gold is None and a.scripts is None and a.hpo is None
        raw=rows(p['references']['all_prompts']['path']);gated=rows(p['references']['gated_prompts']['path'])
        scores=rows(p['references']['gate_scores']['path']);lookup={key(r['pmc_id'],r):r['gate_probability'] for r in scores}
        assert len(scores)==len(lookup)==15827
        assert all(math.isfinite(v) and 0<=v<=1 for v in lookup.values())
        all_keys=[key(r['pmc_id'],c) for r in raw for c in r['candidates']]
        gated_keys=[key(r['pmc_id'],c) for r in gated for c in r['candidates']]
        assert len(set(all_keys))==len(all_keys)==15827 and set(all_keys)==set(lookup)
        assert len(set(gated_keys))==len(gated_keys)==1589
        assert set(gated_keys)=={k for k,v in lookup.items() if v>=.1}
        eligible_full=[k for k in all_keys if lookup[k]>=.4]
        eligible_gate=[k for k in gated_keys if lookup[k]>=.4]
        assert eligible_full==eligible_gate, 'Including stable tie order, not just keys'
        inputs={str(d['pmc_id']):d for d in rows(p['references']['input']['path'])}
        templates=rows(p['references']['template']['path']);assert len(inputs)==len(templates)==16
        assert all(not {'entities','association'}&d.keys() for d in inputs.values())
        def groups(source):
            out={str(d['pmc_id']):[] for d in templates}
            for r in source:
                for c in r['candidates']:
                    k=key(r['pmc_id'],c);d=inputs[k[0]]
                    assert any(c['offset']>=v['offset'] and c['offset']+c['length']<=v['offset']+len(v['text']) and
                        v['text'][c['offset']-v['offset']:c['offset']-v['offset']+c['length']]==c['text'] for v in d['full_text'])
                    out[k[0]].append(dict(c,gate_probability=lookup[k]))
            return out
        full,gate=groups(raw),groups(gated);pred=[]
        for d in templates:
            pmc=str(d['pmc_id']);entities=decode(full[pmc]);assert entities==decode(gate[pmc])
            q=copy.deepcopy(d);q['entities']=entities;assert {k:v for k,v in q.items() if k!='entities'}=={k:v for k,v in d.items() if k!='entities'}
            pred.append(q)
        target=b/'pred.jsonl'
        with target.open('x') as f:
            for d in pred:f.write(json.dumps(d,ensure_ascii=False)+'\n')
        save(b/'freeze.json',dict(frozen_at_utc=now(),plan_sha256=sha(b/'plan.json'),source_sha256=sha(__file__),prediction=ref(target),
            full_candidates=15827,gated_candidates=1589,eligible_at_point4=len(eligible_full),
            eligible_order_and_probabilities_exact=True,full_and_gated_decoded_entities_exact=True,
            entities=sum(len(d['entities']) for d in pred),all_nonentities_exact=True,cut=.4,gold_read=False,
            fit=False,neural_forward=False,seconds=time.monotonic()-tick))
        print((b/'freeze.json').read_text())
    else:
        freeze=load(b/'freeze.json');assert freeze['gold_read'] is False and freeze['cut']==.4
        assert freeze['plan_sha256']==sha(b/'plan.json') and sha(freeze['prediction']['path'])==freeze['prediction']['sha256']
        out=b/'evaluation';out.mkdir(exist_ok=False)
        save(out/'before-gold.json',dict(time_utc=now(),freeze=ref(b/'freeze.json'),gold_read=False,source_sha256=sha(__file__),
            evaluator=ref(a.scripts/'evaluate.py'),validator=ref(a.scripts/'validate_predictions.py')))
        sys.path.insert(0,str(a.scripts));import evaluate as evaluator;import validate_predictions as validator
        assert evaluator.VERSION=='patientphex-public-approx-v2'
        inputs=validator.load_jsonl(p['references']['input']['path']);gold=validator.load_jsonl(a.gold)
        assert {d['pmc_id'] for d in gold}=={d['pmc_id'] for d in inputs}
        pred=validator.load_jsonl(freeze['prediction']['path']);hpo=validator.read_hpo(a.hpo)
        assert hpo['version']=='hp/releases/2026-06-23'
        validation=validator.validate_predictions(inputs,pred,hpo);save(out/'validation.json',validation);assert validation['valid']
        report=evaluator.evaluate_report(gold,pred,'all','text');report.update(gold_sha256=sha(a.gold),prediction_sha256=freeze['prediction']['sha256'])
        save(out/'metrics.json',report)
        previous=load(p['references']['entity_results']['path'])['metrics']
        save(out/'comparison.json',dict(completed_at_utc=now(),seconds=time.monotonic()-tick,baseline=report['metrics'],existing_entity_control=previous,
            delta_final_minus_HGB={k:previous['final'][k]-report['metrics'][k] for k in report['metrics']},
            old_entity_scores_reused_not_reevaluated=True,limitations=['This baseline is newly evaluated after the entity result, not a pre-existing result.',
              'Single historical HGB .4 cut, no search. Identical train32 and eligible candidate resources; different learned scoring objectives.',
              'Training gate is resubstitution; no claim of current97-feature system superiority.']))
        print((out/'comparison.json').read_text())

if __name__=='__main__':main()
