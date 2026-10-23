#!/usr/bin/env python3
"""Train-only nearest passage demonstrations for exact phenotype quote extraction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from run import read_jsonl, sha256

SYSTEM = '''Extract every explicitly mentioned human clinical phenotype from the supplied biomedical text.
Return a JSON array of exact quoted text strings, and nothing else. Each string must be a contiguous substring of the supplied TEXT, with exactly the same spelling and case.
Include signs, symptoms, abnormal morphology, abnormal development and abnormal laboratory or imaging findings. Include general phenotype mentions in discussion as well as findings of individual patients. Exclude disease/diagnosis names, gene names, treatments, normal findings, explicitly absent/negated findings, and isolated anatomical structure names. Do not infer a phenotype from a disease.
Prefer the complete specific phenotype phrase without unrelated modifiers; preserve modifiers that define the abnormality. A phrase may contain several jointly expressed phenotypes. List each distinct quoted phrase once; do not explain and do not emit HPO IDs.'''


def windows(doc, max_chars=4000, overlap=250):
    """Keep full short passages and split long passages at sentence boundaries."""
    for pi, passage in enumerate(doc['full_text']):
        body = passage['text'];start = 0
        if not body.strip():continue
        while start < len(body):
            end = min(start + max_chars, len(body))
            if end < len(body):
                ends = [m.end() for m in re.finditer(r'[.!?;]\s+', body[start:end])]
                if ends and ends[-1] > max_chars//2:end=start+ends[-1]
            yield {'pmc_id':str(doc['pmc_id']),'passage_index':pi,'start':start,'end':end,
                   'offset':int(passage['offset'])+start,'text':body[start:end],
                   'section':str(passage.get('section_type',''))+' '+str(passage.get('type',''))}
            if end == len(body):break
            start=max(start+1,end-overlap)


def gold_quotes(doc, window):
    begin=window['offset'];end=begin+len(window['text']);found=[]
    for e in doc.get('entities',[]):
        a=int(e['offset']);b=a+int(e['length'])
        if str(e.get('note','')).upper()=='NO' or not(begin<=a<b<=end):continue
        if window['text'][a-begin:b-begin]!=e['text']:continue
        if e['text'] not in found:found.append(e['text'])
    return found


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--chunk-chars',type=int,default=4000);ap.add_argument('--overlap',type=int,default=250)
    ap.add_argument('--fewshot',type=int,default=2);ap.add_argument('--limit-documents',type=int,default=0)
    args=ap.parse_args();train=read_jsonl(args.train);inputs=read_jsonl(args.input)
    assert not ({str(x['pmc_id']) for x in train}&{str(x['pmc_id']) for x in inputs})
    assert not any(x.get('entities') or x.get('association') for x in inputs)
    if args.limit_documents:inputs=inputs[:args.limit_documents]
    examples=[]
    for doc in train:
        for w in windows(doc,1800,0):
            if len(w['text'])<100:continue
            quotes=gold_quotes(doc,w)
            # Negative examples are valid, but positive exemplars teach uncommon phenotype forms.
            if quotes:examples.append(dict(w,quotes=quotes))
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=35000,stop_words='english',sublinear_tf=True,dtype=np.float32)
    matrix=vec.fit_transform([x['text'] for x in examples]);out=[]
    for doc in inputs:
        for w in windows(doc,args.chunk_chars,args.overlap):
            scores=(vec.transform([w['text']])@matrix.T).toarray()[0];chosen=[];seen=set()
            for i in np.argsort(scores)[::-1]:
                if examples[i]['pmc_id'] in seen:continue
                chosen.append(int(i));seen.add(examples[i]['pmc_id'])
                if len(chosen)>=args.fewshot:break
            messages=[{'role':'system','content':SYSTEM}]
            if args.fewshot:
                for i in chosen:
                    x=examples[i];messages += [{'role':'user','content':'TEXT:\n'+x['text']},{'role':'assistant','content':json.dumps(x['quotes'],ensure_ascii=False)}]
            messages.append({'role':'user','content':'TEXT:\n'+w['text']})
            row_id=f"{doc['pmc_id']}:{w['passage_index']}:{w['start']}:{w['end']}"
            out.append(dict(w,row_id=row_id,messages=messages,retrieved_training_documents=[examples[i]['pmc_id'] for i in chosen] if args.fewshot else []))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
    manifest={'stage':'exact phenotype extraction','configuration':vars(args),'documents':len(inputs),'chunks':len(out),'training_examples':len(examples),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['train','input','output']}}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':main()
