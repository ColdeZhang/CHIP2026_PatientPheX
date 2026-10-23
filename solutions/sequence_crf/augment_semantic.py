#!/usr/bin/env python3
"""Add independently scored semantic spans to CRF entities, then reassign patients.

Semantic candidates must be fitted on the same training fold. Their generation
manifest is required and its train/input/HPO hashes are checked before use.
"""
import argparse
import gzip
import json
from pathlib import Path
import run as base
from combine import ontology,assign


def main():
    p=argparse.ArgumentParser()
    for k in ['train','input','hpo','base','candidates','candidate-manifest','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--probability',type=float,default=.65);p.add_argument('--similarity',type=float,default=.92)
    p.add_argument('--relation-mode',choices=['lr','rules'],default='lr');p.add_argument('--relation-threshold',type=float,default=.55)
    a=p.parse_args();m=json.loads(Path(a.candidate_manifest).read_text())
    for k in ['train','input','hpo']:
        if m['inputs'][k]!=base.sha(getattr(a,k)):raise ValueError('Semantic candidate fold/HPO mismatch: '+k)
    if m['output']['sha256']!=base.sha(a.candidates):raise ValueError('Candidate bytes mismatch')
    docs=base.read_jsonl(a.input);train=base.read_jsonl(a.train);bases={str(d['pmc_id']):d for d in base.read_jsonl(a.base)}
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert not {str(d['pmc_id']) for d in docs}&{str(d['pmc_id']) for d in train}
    with gzip.open(a.candidates,'rt') as f:cs={str(d['pmc_id']):d['candidates'] for d in map(json.loads,f)}
    if a.relation_mode=='lr':linker=base.Linker(train,a.hpo);vectorizer,relation,_=base.fit_relation(train,linker)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);added=0
    with out.open('w') as h:
        for d in docs:
            es=list(bases[str(d['pmc_id'])]['entities']);accepted=[]
            for c in cs[str(d['pmc_id'])]:
                if c['exact'] or c['probability']<a.probability or c['similarity']<a.similarity:continue
                _,passage=base.passage_for(d,c['offset']);off=int(passage.get('offset',0))
                if ontology.is_negated(passage.get('text',''),c['offset']-off,c['offset']-off+c['length']):continue
                accepted.append(c)
            for c in sorted(accepted,key=lambda c:(-(.65*c['probability']+.3*c['similarity']+.025*min(4,len(ontology.TOKEN.findall(c['text'])))),-c['length'],c['offset'])):
                if any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in es):continue
                es.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None});added+=1
            es.sort(key=lambda e:(e['offset'],e['length'],e['identifier']))
            if a.relation_mode=='rules':associations=assign(d,es)
            else:
                rows=[];pairs=[];values={p['patient_id']:set() for p in d.get('patient',[])}
                for e in es:
                    for patient in d.get('patient',[]):rows.append(base.pair_features(d,e,patient));pairs.append((e,patient))
                probs=relation.predict_proba(vectorizer.transform(rows))[:,1] if rows else []
                for (e,patient),prob in zip(pairs,probs):
                    if prob>=a.relation_threshold:values[patient['patient_id']].update(base.association_keys(e))
                associations=[{'patient_id':p['patient_id'],'phenotype':sorted(values[p['patient_id']])} for p in d.get('patient',[])]
            h.write(json.dumps({'pmc_id':d['pmc_id'],'pmid':d.get('pmid'),'entities':es,'association':associations},ensure_ascii=False)+'\n')
    manifest={'config':vars(a),'source_sha256':base.sha(__file__),'sequence_source_sha256':base.sha(base.__file__),'ontology_source_sha256':base.sha(ontology.__file__),
      'inputs':{k:base.sha(getattr(a,k)) for k in ['train','input','hpo','base','candidates','candidate_manifest']},'output_sha256':base.sha(out),'added_entities':added,'input_annotations_used':False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
