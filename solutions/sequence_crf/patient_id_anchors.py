#!/usr/bin/env python3
"""Derive clinical-number anchors from the official O+article-identifier schema.

P-prefixed IDs are annotation order, never interpreted as clinical identifiers.
Only complete unique identifiers are used; bare numeric/letter IDs require an
explicit patient/case/individual/subject label in the original input text.
"""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import re

SCHEMA_SOURCE={'cached_relative_path': 'docs/competition/sources/2026-10-23/official-github/README.md', 'sha256': '644398fd9de7473af11bd502e3ffa9192dd00c9b92a796be86665a38aadb5e51', 'quote': '||patient\\_id|OII\\.1|患者唯一标识，文章出现具体编号则“O\\+文中的编号”，其他为P开头按出现顺序编号|', 'source_url': 'https://github.com/DUTIR-BioNLP/PatientPheX'}

DASH='-\u2010\u2011\u2012\u2013\u2014\u2015\u2212'
FAMILY=re.compile(r'^([IVX]+)\s*[-:.\u2010-\u2015\u2212]?\s*(\d+)$',re.I)


def clinical_identity(pid):
    if not str(pid).startswith('O') or len(str(pid))<2:return None
    text=str(pid)[1:].strip();family=FAMILY.fullmatch(text)
    if family:
        generation,number=family.groups();pattern=re.escape(generation)+r'\s*[-:.\u2010-\u2015\u2212]?\s*'+re.escape(number)
        return generation.upper()+':'+number,re.compile(r'(?<!\w)'+pattern+r'(?!\w)',re.I)
    if not re.fullmatch(r'[A-Za-z0-9]+(?:[ .:_\-][A-Za-z0-9]+)*',text):return None
    pattern=re.escape(text).replace(r'\ ',r'\s+')
    if not re.match(r'(?i)^(?:patient|case|individual|subject)\s*\d',text):pattern=r'\b(?:patient|case|individual|subject)\s*(?:number\s*|no\.?\s*|#\s*)?'+pattern
    return text.casefold(),re.compile(r'(?<!\w)'+pattern+r'(?!\w)',re.I)


def expand(doc,keep_annotations=False):
    result=copy.deepcopy(doc if keep_annotations else {k:v for k,v in doc.items() if k not in {'entities','association'}})
    identities=defaultdict(set);patterns={};original=[]
    for p in result.get('patient',[]):
        ident=clinical_identity(p['patient_id'])
        if ident:identities[ident[0]].add(p['patient_id']);patterns[p['patient_id']]=ident
        for m in p.get('mention',[]):original.append((int(m['offset']),int(m['offset'])+int(m['length']),p['patient_id']))
    added=[]
    for p in result.get('patient',[]):
        pid=p['patient_id']
        if pid not in patterns:continue
        key,pattern=patterns[pid]
        if identities[key]!={pid}:continue
        occupied={(int(m['offset']),int(m['length'])) for m in p.get('mention',[])}
        for passage in result.get('full_text',[]):
            for match in pattern.finditer(passage.get('text','')):
                offset=int(passage.get('offset',0))+match.start();end=offset+len(match.group())
                if (offset,end-offset) in occupied or any(owner!=pid and offset<b and a<end for a,b,owner in original):continue
                mention={'text':match.group(),'offset':offset,'length':end-offset};p.setdefault('mention',[]).append(mention);occupied.add((offset,end-offset));added.append({'pmc_id':str(doc['pmc_id']),'patient_id':pid,'identity_key':key,**mention})
    return result,added


def main():
    import run as base
    import combine
    p=argparse.ArgumentParser()
    for k in ['train','input','hpo','predictions','output-dir']:p.add_argument('--'+k,required=True)
    a=p.parse_args();train=base.read_jsonl(a.train);docs=base.read_jsonl(a.input);preds={str(d['pmc_id']):d for d in base.read_jsonl(a.predictions)}
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in docs};assert all('entities' not in d and 'association' not in d for d in docs)
    expanded_train=[];expanded_input=[];train_added=[];input_added=[]
    for d in train:
        x,add=expand(d,True);expanded_train.append(x);train_added.extend(add)
    for d in docs:
        x,add=expand(d);expanded_input.append(x);input_added.extend(add)
    linker=base.Linker(train,a.hpo);vec,rel,_=base.fit_relation(expanded_train,linker);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);outputs={}
    derived={}
    for name,documents in [('train',expanded_train),('input',expanded_input)]:
        path=out/('derived-'+name+'.jsonl');path.write_text(''.join(json.dumps(d,ensure_ascii=False,sort_keys=True)+'\n' for d in documents));derived[name]={'path':str(path),'sha256':base.sha(path),'contains_annotations':name=='train'}
    for method in ['lr','rules']:
        path=out/(method+'.pred.jsonl')
        with path.open('w') as h:
            for d in expanded_input:
                pr=preds[str(d['pmc_id'])];entities=pr['entities']
                if method=='rules':association=combine.assign(d,entities)
                else:
                    vals={p['patient_id']:set() for p in d.get('patient',[])};xx=[];pairs=[]
                    for e in entities:
                        for patient in d.get('patient',[]):xx.append(base.pair_features(d,e,patient));pairs.append((e,patient['patient_id']))
                    scores=rel.predict_proba(vec.transform(xx))[:,1] if xx else []
                    for (e,pid),score in zip(pairs,scores):
                        if score>=.55:vals[pid].update(base.association_keys(e))
                    association=[{'patient_id':p['patient_id'],'phenotype':sorted(vals[p['patient_id']])} for p in d.get('patient',[])]
                h.write(json.dumps(dict(pr,association=association),ensure_ascii=False)+'\n')
        outputs[method]={'path':str(path),'sha256':base.sha(path)}
    (out/'anchor-additions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in input_added))
    manifest={'official_schema_source':SCHEMA_SOURCE,'derived_documents':derived,'method':'Official O+clinical identifier schema, unique complete original-text anchors','configuration':vars(a),'source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__),'combine_source_sha256':base.sha(combine.__file__),'ontology_source_sha256':base.sha(combine.ontology.__file__),'input_files':{k:base.sha(getattr(a,k)) for k in ['train','input','hpo','predictions']},'train_added_anchors':len(train_added),'input_added_anchors':len(input_added),'documents_with_new_input_anchors':len({x['pmc_id'] for x in input_added}),'outputs':outputs,'input_annotations_used':False,'original_input_files_modified':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
