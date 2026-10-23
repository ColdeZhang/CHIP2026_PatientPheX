#!/usr/bin/env python3
"""Resolve linked quotes and combine entities without losing input patient coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from run import read_jsonl,load_hpo,normalize_id,sha256
from normalize_extraction import parse_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--hpo',required=True)
    ap.add_argument('--items',required=True);ap.add_argument('--link-responses')
    ap.add_argument('--baseline',required=True);ap.add_argument('--mode',choices=['new-only','union','prefer-new'],default='union')
    ap.add_argument('--output',required=True)
    args=ap.parse_args();inputs=read_jsonl(args.input);baseline=read_jsonl(args.baseline);base={str(x['pmc_id']):x for x in baseline}
    assert set(base)=={str(x['pmc_id']) for x in inputs}
    assert not any(d.get('entities') or d.get('association') for d in inputs)
    data=json.loads(Path(args.items).read_text());items={x['code']:x for x in data['items']};decisions={};stats=Counter()
    if args.link_responses:
        for line in Path(args.link_responses).read_text().splitlines():
            response=json.loads(line);parsed=parse_json(response['response'])
            if not isinstance(parsed,dict):stats['invalid_link_responses']+=1;continue
            for key,value in parsed.items():
                if key not in items:stats['unknown_item_codes']+=1;continue
                if isinstance(value,list):value=';'.join(str(x) for x in value)
                if not isinstance(value,str):stats['invalid_link_values']+=1;continue
                candidates={part for c in items[key]['candidates'] for part in c['identifier'].split(';')}
                if value!='-1' and (not re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})*',value) or not set(value.split(';'))<=candidates):
                    stats['out_of_candidate_ids']+=1;continue
                decisions[key]=value
    _,_,canonical,_=load_hpo(args.hpo);outputs=[]
    for doc in inputs:
        pmc=str(doc['pmc_id']);old=base[pmc]['entities'];new=[]
        for occurrence in data['occurrences'].get(pmc,[]):
            item=items[occurrence['item_code']]
            if item['exact']:identifier=item['identifier'];stats['exact_links']+=1
            elif occurrence['item_code'] in decisions:identifier=decisions[occurrence['item_code']];stats['llm_links']+=1
            else:stats['unresolved_occurrences']+=1;continue
            identifier=normalize_id(identifier,canonical)
            if identifier is None:stats['invalid_hpo_links']+=1;continue
            new.append({'identifier':identifier,'type':'Phenotype','offset':occurrence['offset'],'length':occurrence['length'],'text':occurrence['text'],'note':None})
        if args.mode=='new-only':entities=new
        elif args.mode=='prefer-new':
            entities=new+[e for e in old if not any(e['offset']<q['offset']+q['length'] and q['offset']<e['offset']+e['length'] for q in new)]
        else:
            # Preserve trusted baseline spans; add genuinely new non-overlapping spans.
            entities=old+[e for e in new if not any(e['offset']<q['offset']+q['length'] and q['offset']<e['offset']+e['length'] for q in old)]
        unique={(e['offset'],e['length'],e['identifier']):e for e in entities}
        outputs.append({'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':[unique[k] for k in sorted(unique)],'association':base[pmc]['association']})
        stats['documents']+=1;stats['entities']+=len(unique);stats['extracted_entities']+=len(new)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in outputs))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(args),'stats':dict(stats),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['input','hpo','items','baseline','output']+(['link_responses'] if args.link_responses else [])}},indent=2)+'\n');print(json.dumps(stats))
if __name__=='__main__':main()
