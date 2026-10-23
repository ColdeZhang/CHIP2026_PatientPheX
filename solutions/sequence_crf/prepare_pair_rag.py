#!/usr/bin/env python3
"""Phenotype evidence batches with explicit patient competition, based on official baseline design."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
# Adapted from sibling case_retrieval at 5937b80; prompt semantics preserved.
from run import read_jsonl,parse_hpo,sha as sha256,association_keys

def load_hpo(path):
    terms,allowed,alt,version=parse_hpo(path)
    return {k:dict(v,name=v['names'][0]) for k,v in terms.items()},allowed,alt,version

def grouped_concepts(entities):
    out=defaultdict(list)
    for entity in entities:
        for key in association_keys(entity):out[key].append(entity)
    return out

def mark_patients(text,absolute_start,doc):
    groups=defaultdict(set)
    for patient in doc.get('patient',[]):
        for mention in patient.get('mention',[]):
            start=int(mention['offset'])-absolute_start;end=start+int(mention.get('length',len(mention['text'])))
            if 0<=start<end<=len(text):groups[(start,end)].add(str(patient['patient_id']))
    pieces=[];position=0
    for (start,end),patients in sorted(groups.items()):
        if start<position:continue
        pieces.extend([text[position:start],'[PATIENT '+'/'.join(sorted(patients))+': '+text[start:end]+']']);position=end
    pieces.append(text[position:]);return ''.join(pieces)

SYSTEM='''Determine which supplied patients actually have each candidate clinical phenotype in a biomedical report.
Return one JSON object mapping EVERY candidate code to a list of supplied patient IDs. Return [] when no supplied patient is supported. Use only listed candidate codes and patient IDs.
Each candidate includes its HPO meaning and verbatim local evidence. Inline [PATIENT P1: text] markers identify supplied patient mentions. Use the global patient anchors to resolve case numbers, family members and unambiguous pronouns across paragraphs. A phenotype may belong to multiple patients only when the text explicitly describes those patients or an explicit group including them.
A supplied patient can be a relative; report that relative's own findings for their patient ID.
Assign observed, present findings only. Reject negated/normal findings, hypothetical possibilities, general syndrome descriptions and other cited cases. Do not infer symptoms from a diagnosis. When several patients occur, compare their evidence and do not copy one patient's findings to everyone. Clinical abnormality diagnoses such as glaucoma and diabetes can be phenotypes when supported. If the specific HPO meaning differs from the quoted evidence, reject that candidate. No explanation or markdown.'''


def patient_anchors(doc,limit=4000):
    rows=[]
    for p in doc.get('patient',[]):
        snippets=[];seen=set()
        for mention in p.get('mention',[]):
            pos=int(mention['offset']);length=int(mention['length'])
            for passage in doc['full_text']:
                begin=int(passage['offset']);local=pos-begin
                if not(0<=local<len(passage['text'])):continue
                left=max(0,local-130);right=min(len(passage['text']),local+length+170)
                text=mark_patients(passage['text'][left:right],begin+left,doc)
                if text not in seen:snippets.append(text);seen.add(text)
                break
            if len(snippets)>=3:break
        rows.append(str(p['patient_id'])+': '+' | '.join(snippets))
    # Keep all patients, cap each patient's anchor rather than dropping the tail patients.
    budget=max(140,limit//max(1,len(rows)))
    return '\n'.join(x[:budget] for x in rows)


def evidence(doc,entities,radius):
    rows=[]
    for entity in entities:
        for passage in doc['full_text']:
            offset=int(passage['offset']);local=int(entity['offset'])-offset
            if not(0<=local and local+int(entity['length'])<=len(passage['text'])):continue
            left=max(0,local-radius);right=min(len(passage['text']),local+int(entity['length'])+radius)
            text=mark_patients(passage['text'][left:right],offset+left,doc)
            section=str(passage.get('section_type',''))+' '+str(passage.get('type',''))
            clinical=bool(re.search(r'case|patient|result|clinical|method',section,re.I))
            rows.append((int('[PATIENT' in text)+int(clinical),int(entity['offset']),{'quote':entity['text'],'section':section,'context':text}))
            break
    unique=[];seen=set()
    for priority,offset,row in sorted(rows,key=lambda x:(-x[0],x[1])):
        if row['context'] not in seen:unique.append(row);seen.add(row['context'])
        if len(unique)>=4:break
    return unique


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--candidates',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--batch-concepts',type=int,default=8);ap.add_argument('--context-radius',type=int,default=500)
    args=ap.parse_args();inputs=read_jsonl(args.input);pred={str(x['pmc_id']):x for x in read_jsonl(args.candidates)};terms,_,_,_=load_hpo(args.hpo)
    assert not any(x.get('entities') or x.get('association') for x in inputs)
    assert set(pred)=={str(x['pmc_id']) for x in inputs}
    rows=[]
    for doc in inputs:
        concepts=grouped_concepts(pred[str(doc['pmc_id'])]['entities']);ordered=sorted(concepts);anchors=patient_anchors(doc)
        for start in range(0,len(ordered),args.batch_concepts):
            selected=ordered[start:start+args.batch_concepts];code_map={f'C{i:03d}':identifier for i,identifier in enumerate(selected)};items=[]
            for code,identifier in code_map.items():
                items.append({'code':code,'meaning':terms.get(identifier,{}).get('name',identifier),'evidence':evidence(doc,concepts[identifier],args.context_radius)})
            content='Patient identities and global anchors:\n'+anchors+'\n\nCandidate phenotypes:\n'+json.dumps(items,ensure_ascii=False)
            rows.append({'row_id':f"{doc['pmc_id']}:relations:{start}",'pmc_id':doc['pmc_id'],'code_map':code_map,'patient_ids':[str(p['patient_id']) for p in doc.get('patient',[])],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}]})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(args),'documents':len(inputs),'prompts':len(rows),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['input','candidates','hpo','output']}},indent=2)+'\n');print(json.dumps({'prompts':len(rows),'bytes':out.stat().st_size}))
if __name__=='__main__':main()
