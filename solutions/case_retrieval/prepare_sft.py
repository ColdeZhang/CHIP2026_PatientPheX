#!/usr/bin/env python3
"""Build bounded supervised extraction examples from one explicitly supplied training fold."""
import argparse
import json
from pathlib import Path
from run import read_jsonl,sha256
from prepare_extraction import windows,gold_quotes,SYSTEM

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--excluded-input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--chunk-chars',type=int,default=3500)
    args=ap.parse_args();train=read_jsonl(args.train);excluded=read_jsonl(args.excluded_input)
    assert not({str(x['pmc_id']) for x in train}&{str(x['pmc_id']) for x in excluded})
    assert not any(x.get('entities') or x.get('association') for x in excluded)
    rows=[]
    for doc in train:
        for chunk in windows(doc,args.chunk_chars,0):
            quotes=gold_quotes(doc,chunk)
            rows.append({'row_id':f"{doc['pmc_id']}:sft-extract:{chunk['chunk_index']}",'pmc_id':str(doc['pmc_id']),'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'TEXT:\n'+chunk['text']}],
                         'answer':json.dumps(quotes,ensure_ascii=False),'segments':chunk['segments']})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    out.with_suffix('.manifest.json').write_text(json.dumps({'purpose':'supervised extraction, training fold only; not an evaluation output','configuration':vars(args),'training_documents':len(train),'examples':len(rows),'negative_examples':sum(x['answer']=='[]' for x in rows),'code_sha256':sha256(__file__),'train_sha256':sha256(args.train),'excluded_input_sha256':sha256(args.excluded_input),'output_sha256':sha256(out)},indent=2)+'\n');print(json.dumps({'examples':len(rows),'documents':len(train)}))
if __name__=='__main__':main()
