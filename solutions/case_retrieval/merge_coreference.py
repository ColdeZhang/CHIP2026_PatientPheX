#!/usr/bin/env python3
"""Ground generated patient mentions to original passage offsets without changing patient IDs."""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import re
from run import read_jsonl,sha256
from normalize_extraction import parse_json

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ['input','prompts','responses','output']:ap.add_argument('--'+k,required=True)
    args=ap.parse_args();inputs=read_jsonl(args.input);assert not any(x.get('entities') or x.get('association') for x in inputs)
    prompts={x['row_id']:x for x in [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]};responses=[json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()]
    found=defaultdict(lambda:defaultdict(dict));stats=Counter()
    for response in responses:
        prompt=prompts[response['row_id']];parsed=parse_json(response['response'])
        if not isinstance(parsed,list):stats['invalid_responses']+=1;continue
        for item in parsed:
            if not isinstance(item,dict):stats['invalid_item']+=1;continue
            pid=str(item.get('patient_id'));text=item.get('text');number=item.get('occurrence')
            if pid not in prompt['patient_ids'] or not isinstance(text,str) or not text or not isinstance(number,int) or number<1:
                stats['invalid_item']+=1;continue
            matches=[]
            for segment in prompt['segments']:
                for match in re.finditer(re.escape(text),segment['text']):matches.append(segment['offset']+match.start())
            if len(matches)<number:stats['unmatched_quote']+=1;continue
            offset=matches[number-1];found[str(prompt['pmc_id'])][pid][(offset,len(text))]={'offset':offset,'length':len(text),'text':text}
            stats['grounded_items']+=1
    outputs=copy.deepcopy(inputs)
    for doc in outputs:
        for patient in doc.get('patient',[]):
            old={(int(x['offset']),int(x['length'])):x for x in patient.get('mention',[])};new=found[str(doc['pmc_id'])][str(patient['patient_id'])]
            stats['added_mentions']+=len(set(new)-set(old));old.update(new);patient['mention']=[old[k] for k in sorted(old)]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in outputs))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(args),'stats':dict(stats),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['input','prompts','responses','output']}},indent=2)+'\n');print(json.dumps(stats))
if __name__=='__main__':main()
