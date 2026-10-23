#!/usr/bin/env python3
"""Add only unoccupied positive spans agreed by independent sequence models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import run as base
import combine


def main():
    p=argparse.ArgumentParser()
    for k in ('train','input','hpo','base','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--votes',type=int,default=2);p.add_argument('--relations',choices=['lr','rules'],default='rules')
    a=p.parse_args();train=base.read_jsonl(a.train);docs=base.read_jsonl(a.input);bp={str(d['pmc_id']):d for d in base.read_jsonl(a.base)};candidates=[{str(d['pmc_id']):d for d in base.read_jsonl(path)} for path in a.candidates]
    assert all(set(c)==set(bp) for c in candidates);assert not {str(d['pmc_id']) for d in train}&set(bp);assert all('entities' not in d and 'association' not in d for d in docs)
    linker=base.Linker(train,a.hpo);vec,rel,_=base.fit_relation(train,linker);added=[];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as h:
        for d in docs:
            pmc=str(d['pmc_id']);pr=bp[pmc];entities=list(pr['entities']);votes=Counter();lookup={}
            for c in candidates:
                seen=set()
                for e in c[pmc]['entities']:
                    key=(e['offset'],e['length'],e['identifier']);seen.add(key);lookup[key]=e
                votes.update(seen)
            for key,n in sorted(votes.items(),key=lambda x:(-x[1],x[0])):
                if n<a.votes:continue
                e=lookup[key];off=e['offset'];end=off+e['length']
                if any(off<x['offset']+x['length'] and x['offset']<end for x in entities):continue
                _,passage=base.passage_for(d,off);b=int(passage.get('offset',0))
                if combine.ontology.is_negated(passage.get('text',''),off-b,end-b):continue
                entities.append(e);added.append({'pmc_id':pmc,'offset':off,'length':e['length'],'identifier':e['identifier'],'votes':n})
            if a.relations=='rules':association=combine.assign(d,entities)
            else:
                vals={p['patient_id']:set() for p in d.get('patient',[])};xx=[];pairs=[]
                for e in entities:
                    for patient in d.get('patient',[]):xx.append(base.pair_features(d,e,patient));pairs.append((e,patient['patient_id']))
                scores=rel.predict_proba(vec.transform(xx))[:,1] if xx else []
                for (e,pid),score in zip(pairs,scores):
                    if score>=.55:vals[pid].update(base.association_keys(e))
                association=[{'patient_id':p['patient_id'],'phenotype':sorted(vals[p['patient_id']])} for p in d.get('patient',[])]
            pr=dict(pr,entities=sorted(entities,key=lambda e:(e['offset'],e['length'],e['identifier'])),association=association);h.write(json.dumps(pr,ensure_ascii=False)+'\n')
    out.with_suffix('.added.json').write_text(json.dumps(added,indent=2)+'\n')
    out.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(a),'source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__),'combine_source_sha256':base.sha(combine.__file__),'ontology_source_sha256':base.sha(combine.ontology.__file__),'input_hashes':{k:base.sha(getattr(a,k)) for k in ['train','input','hpo','base']},'candidate_hashes':{x:base.sha(x) for x in a.candidates},'output_sha256':base.sha(out),'added_spans':len(added),'input_annotations_used':False},indent=2)+'\n')
if __name__=='__main__':main()
