#!/usr/bin/env python3
"""Combine chunked phenotype-patient decisions, preserving explicit failure accounting."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
# Adapted from sibling case_retrieval at 5937b80; identical fallback semantics.
from run import read_jsonl,sha as sha256
import re

def parse_json(text):
    text=text.strip()
    if text.startswith('<think>') and '</think>' in text:text=text.split('</think>',1)[1].strip()
    text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    try:return json.loads(text)
    except json.JSONDecodeError:
        for opener,closer in [('[',']'),('{','}')]:
            a=text.find(opener);b=text.rfind(closer)
            if 0<=a<b:
                try:return json.loads(text[a:b+1])
                except json.JSONDecodeError:pass
        return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['input','entities','prompts','responses','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--fallback',choices=['none','baseline'],default='baseline')
    args=ap.parse_args();inputs=read_jsonl(args.input);pred={str(x['pmc_id']):x for x in read_jsonl(args.entities)}
    prompts=[json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()];responses=[json.loads(x) for x in Path(args.responses).read_text().splitlines() if x.strip()]
    by_id={x['row_id']:x for x in responses};assert len(by_id)==len(responses)
    assert set(by_id)=={x['row_id'] for x in prompts},'Require complete immutable prompt results'
    assert all('entities' not in d and 'association' not in d for d in inputs)
    relations=defaultdict(lambda:defaultdict(set));stats=Counter()
    for prompt in prompts:
        response=by_id.get(prompt['row_id']);parsed=parse_json(response['response']) if response else None
        pmc=str(prompt['pmc_id']);patients=set(prompt['patient_ids']);base=pred[pmc]
        for code,concept in prompt['code_map'].items():
            value=parsed.get(code) if isinstance(parsed,dict) else None
            if not isinstance(value,list) or any(str(p) not in patients for p in value):
                stats['unresolved_candidate_decisions']+=1
                if args.fallback=='baseline':
                    for old in base['association']:
                        if concept in old.get('phenotype',[]):relations[pmc][str(old['patient_id'])].add(concept)
                continue
            stats['valid_candidate_decisions']+=1
            for pid in value:relations[pmc][str(pid)].add(concept)
    outputs=[]
    for doc in inputs:
        pmc=str(doc['pmc_id']);outputs.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':pred[pmc]['entities'],'association':[{'patient_id':p['patient_id'],'phenotype':sorted(relations[pmc][str(p['patient_id'])])} for p in doc.get('patient',[])]})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in outputs))
    manifest={'configuration':vars(args),'stats':dict(stats),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['input','entities','prompts','responses','output']}}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
