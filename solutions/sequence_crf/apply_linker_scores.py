#!/usr/bin/env python3
"""Reassemble context linker scores with explicit NIL/exact-alias policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import run as base


def select(group):
    # A cross-encoder trained only on HPO classes has no evidence to relink NIL.
    if group['original_identifier'] in {None,'-1'}:return group['original_identifier'] or '-1'
    if group['protected_exact']:return group['original_identifier']
    return max(group['candidates'],key=lambda r:r['probability'])['identifier']


def main():
    p=argparse.ArgumentParser()
    for k in ('train','input','hpo','predictions','scores','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();train=base.read_jsonl(a.train);docs=base.read_jsonl(a.input);preds={str(d['pmc_id']):d for d in base.read_jsonl(a.predictions)}
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in docs}
    assert all('entities' not in d and 'association' not in d for d in docs)
    groups=[json.loads(s) for s in Path(a.scores).read_text().splitlines()];stats=Counter();linker=base.Linker(train,a.hpo)
    for d in train:
        for e in d.get('entities',[]):
            hid=linker.canonical(e.get('identifier','-1'))
            if str(e.get('note') or '').upper()=='NO' and hid and hid!='-1':linker.aliases[base.norm(e['text'])][hid]+=1
    for g in groups:
        e=preds[g['pmc_id']]['entities'][g['entity_index']];assert e['identifier']==g['original_identifier']
        exact={k:v for k,v in linker.aliases.get(base.norm(e['text']),{}).items() if k!='-1'}
        if len(exact)==1 and next(iter(exact))!=e['identifier']:stats['unique_exact_original_mismatch']+=1
        chosen=select(g);stats['nil_preserved']+=e['identifier']=='-1';stats['identifier_changed']+=chosen!=e['identifier'];e['identifier']=chosen
    vec,rel,_=base.fit_relation(train,linker);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as h:
        for d in docs:
            pr=preds[str(d['pmc_id'])];vals={p['patient_id']:set() for p in d.get('patient',[])};xx=[];pairs=[]
            for e in pr['entities']:
                for patient in d.get('patient',[]):xx.append(base.pair_features(d,e,patient));pairs.append((e,patient['patient_id']))
            scores=rel.predict_proba(vec.transform(xx))[:,1] if xx else []
            for (e,pid),score in zip(pairs,scores):
                if score>=.55:vals[pid].update(base.association_keys(e))
            pr=dict(pr,association=[{'patient_id':p['patient_id'],'phenotype':sorted(vals[p['patient_id']])} for p in d.get('patient',[])]);h.write(json.dumps(pr,ensure_ascii=False)+'\n')
    out.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(a),'command':sys.argv,'policy':'Preserve original NIL and exact mappings; rerank only mapped ambiguous/nonexact entities','source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__),'inputs':{k:base.sha(getattr(a,k)) for k in ('train','input','hpo','predictions','scores')},'output_sha256':base.sha(out),'stats':dict(stats),'input_annotations_used':False},indent=2)+'\n')
if __name__=='__main__':main()
