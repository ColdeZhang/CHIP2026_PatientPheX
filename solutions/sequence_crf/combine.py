#!/usr/bin/env python3
"""Deterministic ablation: sequence entities with ontology patient state machine.

Imports the already merged first-day ontology implementation; no fitting or gold
inputs. Optional extra spans are appended only in unoccupied, nonnegative text.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import run as sequence

path=Path(__file__).resolve().parents[1]/'ontology_rules'/'run.py'
spec=importlib.util.spec_from_file_location('sequence_ontology_rules',path)
ontology=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ontology;spec.loader.exec_module(ontology)


def assign(doc,entities,anchors=None):
    if anchors is None:anchors=ontology.patient_anchors(doc)
    out={p['patient_id']:set() for p in doc.get('patient',[])};active=None;previous=None
    by_passage={}
    for e in entities:
        pi,p=sequence.passage_for(doc,e['offset']);by_passage.setdefault(int(p.get('offset',0)),[]).append(e)
    for p in sorted(doc.get('full_text',[]),key=lambda p:int(p.get('offset',0))):
        text=p.get('text','');off=int(p.get('offset',0));section=str(p.get('section_type','')).upper()
        if section in ontology.STOP_SECTIONS:continue
        if section!=previous and section in ontology.GENERAL_SECTIONS:active=None
        local=[a for a in anchors if off<=a[0]<off+len(text)]
        for e in by_passage.get(off,[]):
            a=e['offset']-off;b=a+e['length'];ids=tuple(e['identifier'].split(';'))
            if ontology.is_negated(text,a,b):continue
            mention=ontology.Mention(a,b,ontology.token_key(e['text']),ids,0)
            for pid in ontology.assign_patients(doc,p,mention,anchors,active):out[pid].update([e['text']] if ids==('-1',) else ids)
        if local:active=local[-1][2]
        previous=section
    return [{'patient_id':p['patient_id'],'phenotype':sorted(out[p['patient_id']])} for p in doc.get('patient',[])]


def main():
    p=argparse.ArgumentParser()
    for k in ('input','base','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--extra');p.add_argument('--keep-relations',action='store_true')
    a=p.parse_args();docs=sequence.read_jsonl(a.input);bases={str(d['pmc_id']):d for d in sequence.read_jsonl(a.base)}
    extra={str(d['pmc_id']):d for d in sequence.read_jsonl(a.extra)} if a.extra else {}
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert set(bases)=={str(d['pmc_id']) for d in docs}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);added=0
    with out.open('w') as h:
        for doc in docs:
            pred=dict(bases[str(doc['pmc_id'])]);entities=list(pred['entities'])
            for e in extra.get(str(doc['pmc_id']),{}).get('entities',[]):
                off,end=e['offset'],e['offset']+e['length']
                if any(off<x['offset']+x['length'] and x['offset']<end for x in entities):continue
                _,passage=sequence.passage_for(doc,off);baseoff=int(passage.get('offset',0))
                if ontology.is_negated(passage.get('text',''),off-baseoff,end-baseoff):continue
                entities.append(e);added+=1
            pred['entities']=sorted(entities,key=lambda e:(e['offset'],e['length'],e['identifier']))
            if not a.keep_relations:pred['association']=assign(doc,entities)
            h.write(json.dumps(pred,ensure_ascii=False)+'\n')
    manifest={'config':vars(a),'source_sha256':sequence.sha(__file__),'ontology_source_sha256':sequence.sha(path),'sequence_source_sha256':sequence.sha(sequence.__file__),
      'inputs':{k:sequence.sha(getattr(a,k)) for k in ['input','base','extra'] if getattr(a,k,None)},'added_entities':added,'output_sha256':sequence.sha(out),'input_annotations_used':False}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
