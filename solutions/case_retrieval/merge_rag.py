#!/usr/bin/env python3
"""Convert generated candidate-code selections into strict submission associations."""
import argparse
import json
from pathlib import Path
import re

from run import read_jsonl,sha256


def parse_object(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S).strip()
    decoder=json.JSONDecoder()
    for match in re.finditer(r'\{',text):
        try:
            value,_=decoder.raw_decode(text[match.start():])
            if isinstance(value,dict):return value
        except json.JSONDecodeError:pass
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates',required=True);ap.add_argument('--prompts',required=True)
    ap.add_argument('--responses',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    candidates=read_jsonl(args.candidates);prompts={str(d['pmc_id']):d for d in read_jsonl(args.prompts)}
    responses={str(d['pmc_id']):d for d in read_jsonl(args.responses)}
    assert set(prompts)==set(responses)=={str(d['pmc_id']) for d in candidates}
    issues=[];results=[]
    for doc in candidates:
        key=str(doc['pmc_id']);meta=prompts[key];parsed=parse_object(responses[key]['response'])
        if parsed is None:
            issues.append({'pmc_id':doc['pmc_id'],'issue':'invalid_json','fallback':'CPU kNN associations'})
            results.append(doc);continue
        associations=[]
        for old in doc['association']:
            patient=str(old['patient_id']);codes=parsed.get(patient)
            if not isinstance(codes,list):
                issues.append({'pmc_id':doc['pmc_id'],'patient_id':patient,'issue':'missing_patient','fallback':'CPU kNN association'})
                associations.append(old);continue
            valid_codes=[code for code in codes if isinstance(code,str) and code in meta['code_map']]
            if len(valid_codes)!=len(codes):
                issues.append({'pmc_id':doc['pmc_id'],'patient_id':patient,'issue':'invalid_codes_removed','count':len(codes)-len(valid_codes)})
            associations.append({'patient_id':old['patient_id'],'phenotype':sorted(set(meta['code_map'][code] for code in valid_codes))})
        doc['association']=associations;results.append(doc)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in results))
    metadata={'method':'Qwen3-8B train-case retrieved association over lexical candidates','configuration':vars(args),
              'documents':len(results),'issues':issues,'fallback_document_count':sum(i['issue']=='invalid_json' for i in issues),
              'files':{name:{'sha256':sha256(getattr(args,name)),'path':getattr(args,name)} for name in ['candidates','prompts','responses','output']}}
    out.with_suffix('.merge.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'documents':len(results),'issues':len(issues),'output_sha256':sha256(out)}))


if __name__=='__main__':main()
