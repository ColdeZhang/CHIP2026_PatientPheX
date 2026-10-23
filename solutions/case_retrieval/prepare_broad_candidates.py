#!/usr/bin/env python3
"""Create a separately recorded high-recall relation candidate pool from text evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
from run import read_jsonl,load_hpo,LexicalMemory,sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--input',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();train=read_jsonl(args.train);inputs=read_jsonl(args.input)
    assert not ({str(x['pmc_id']) for x in train}&{str(x['pmc_id']) for x in inputs})
    assert not any(x.get('entities') or x.get('association') for x in inputs)
    terms,valid,canonical,_=load_hpo(args.hpo);lexical=LexicalMemory(terms,valid,canonical,train);stats=Counter();out=[]
    for doc in inputs:
        entities={}
        for passage in doc['full_text']:
            for begin,end,key,identifiers in lexical.match(passage['text']):
                text=passage['text'][begin:end]
                if len(key)==1 and len(key[0])<=3 and not text.isupper():continue
                # Preserve broad candidate IDs and negated/nested mentions for subsequent evidence review.
                for identifier in identifiers:
                    k=(int(passage['offset'])+begin,end-begin,identifier)
                    entities[k]={'identifier':identifier,'type':'Phenotype','offset':k[0],'length':k[1],'text':text,'note':None}
        pred={'pmc_id':doc['pmc_id'],'pmid':doc.get('pmid'),'entities':[entities[k] for k in sorted(entities)],'association':[{'patient_id':p['patient_id'],'phenotype':[]} for p in doc.get('patient',[])]}
        out.append(pred);stats['documents']+=1;stats['entity_candidates']+=len(entities)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
    path.with_suffix('.manifest.json').write_text(json.dumps({'purpose':'broad relation evidence pool, not final entity predictions','configuration':vars(args),'stats':dict(stats),'code_sha256':sha256(__file__),'files':{k:sha256(getattr(args,k)) for k in ['train','input','hpo','output']}},indent=2)+'\n')
    print(json.dumps(stats))
if __name__=='__main__':main()
