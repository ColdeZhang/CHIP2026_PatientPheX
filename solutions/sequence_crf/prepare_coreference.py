#!/usr/bin/env python3
"""Prepare additional patient mention discovery using supplied identity anchors."""
import argparse
import json
from pathlib import Path
from run import read_jsonl,sha as sha256
import re

# Window function copied unchanged from case_retrieval/prepare_extraction.py.
def windows(doc, max_chars=4000, overlap=250):
    """Pack adjacent passages, preserving an explicit original-coordinate segment map."""
    segments=[]
    for pi, passage in enumerate(doc['full_text']):
        body=passage['text'];start=0
        if not body.strip():continue
        while start<len(body):
            end=min(start+max_chars,len(body))
            if end<len(body):
                ends=[m.end() for m in re.finditer(r'[.!?;]\s+',body[start:end])]
                if ends and ends[-1]>max_chars//2:end=start+ends[-1]
            segments.append({'passage_index':pi,'start':start,'end':end,'offset':int(passage['offset'])+start,
                             'text':body[start:end],'section':str(passage.get('section_type',''))+' '+str(passage.get('type',''))})
            if end==len(body):break
            start=max(start+1,end-overlap)
    batch=[];length=0;index=0
    for segment in segments:
        if batch and length+len(segment['text'])+2>max_chars:
            yield {'pmc_id':str(doc['pmc_id']),'chunk_index':index,'segments':batch,'text':'\n\n'.join(x['text'] for x in batch)}
            index+=1;batch=[];length=0
        batch.append(segment);length+=len(segment['text'])+2
    if batch:yield {'pmc_id':str(doc['pmc_id']),'chunk_index':index,'segments':batch,'text':'\n\n'.join(x['text'] for x in batch)}


from prepare_pair_rag import patient_anchors

SYSTEM='''Locate expressions in TEXT that clearly refer to one of the supplied target patients.
Use the supplied original identity anchors to resolve patient numbers, pedigree labels, relatives, and unambiguous pronouns. Target patients can be relatives; preserve their individual IDs. Do not merge different patients because they share a family or diagnosis. Do not map unrelated cited cases or general disease populations to a supplied patient.
Return only a JSON array of objects {"patient_id":"P1","text":"exact substring","occurrence":1}. occurrence is the 1-based occurrence of that exact case-sensitive string inside TEXT, counting left to right. Include only unambiguous references and list each supported occurrence separately. A phrase may identify several target patients only if it explicitly names the corresponding group. Do not invent IDs, text or offsets. Return [] when no identity can be confidently resolved.'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--chunk-chars',type=int,default=5000)
    args=ap.parse_args();inputs=read_jsonl(args.input);assert not any(x.get('entities') or x.get('association') for x in inputs);rows=[]
    for doc in inputs:
        anchors=patient_anchors(doc,5000)
        for chunk in windows(doc,args.chunk_chars,300):
            rows.append(dict(chunk,row_id=f"{doc['pmc_id']}:coreference:{chunk['chunk_index']}",patient_ids=[str(x['patient_id']) for x in doc.get('patient',[])],messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'SUPPLIED PATIENT IDENTITY ANCHORS:\n'+anchors+'\n\nTEXT:\n'+chunk['text']}]))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(args),'documents':len(inputs),'chunks':len(rows),'code_sha256':sha256(__file__),'patient_anchor_source_sha256':sha256(Path(__file__).with_name('prepare_pair_rag.py')),'base_source_sha256':sha256(Path(__file__).with_name('run.py')),'input_sha256':sha256(args.input),'output_sha256':sha256(out)},indent=2)+'\n');print(json.dumps({'chunks':len(rows),'bytes':out.stat().st_size}))
if __name__=='__main__':main()
