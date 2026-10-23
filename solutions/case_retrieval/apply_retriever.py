#!/usr/bin/env python3
"""Apply the frozen training-only case retriever to another grounded entity pool."""
import argparse
import json
from pathlib import Path
import time
from run import CaseRetriever,load_hpo,read_jsonl,sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['train','input','candidates','hpo','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--neighbors',type=int,default=21);parser.add_argument('--threshold',type=float,default=.35)
    args=parser.parse_args();started=time.time()
    train=read_jsonl(args.train);inputs=read_jsonl(args.input);candidates=read_jsonl(args.candidates)
    assert not ({str(x['pmc_id']) for x in train}&{str(x['pmc_id']) for x in inputs})
    assert not any(x.get('entities') or x.get('association') for x in inputs)
    by_id={str(x['pmc_id']):x for x in candidates}
    assert len(by_id)==len(candidates) and set(by_id)=={str(x['pmc_id']) for x in inputs}
    _,_,canonical,_=load_hpo(args.hpo);retriever=CaseRetriever(train,canonical,args.neighbors,args.threshold)
    output=[]
    for doc in inputs:
        old=by_id[str(doc['pmc_id'])]
        output.append(dict(old,association=retriever.predict(doc,old['entities'])))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
    manifest={'purpose':'same entity pool with training-only case retrieval associations','configuration':vars(args),
              'source_sha256':sha256(__file__),'retriever_source_sha256':sha256(Path(__file__).with_name('run.py')),
              'files':{name:sha256(getattr(args,name)) for name in ['train','input','candidates','hpo','output']},
              'elapsed_seconds':time.time()-started,'retriever':retriever.stats,'documents':len(output)}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
