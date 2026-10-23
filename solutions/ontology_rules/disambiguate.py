#!/usr/bin/env python3
"""Document-local acronym meaning correction; no input annotations consumed."""
import argparse,json
from pathlib import Path
import run as baseline
import enhance
import merge_semantic


def correct(doc,pred,lexicon,suppress_unknown=True):
    text='\n'.join(p.get('text','') for p in doc['full_text'])
    known,other=lexicon.local_abbreviations(text)
    entities=[];changes={'removed':0,'relinked':0}
    for e in pred['entities']:
        short=e['text']
        if short in known:
            identifier=';'.join(known[short][0])
            if identifier!=e['identifier']:changes['relinked']+=1
            entities.append(dict(e,identifier=identifier))
        elif short in other and suppress_unknown:changes['removed']+=1
        else:entities.append(e)
    return dict(pred,entities=entities,association=merge_semantic.assign(doc,entities)),changes


def main():
    ap=argparse.ArgumentParser()
    for k in ['train','input','hpo','pred','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--keep-unknown',action='store_true')
    args=ap.parse_args();train=baseline.read_jsonl(args.train);inputs=baseline.read_jsonl(args.input)
    assert not {d['pmc_id'] for d in train}&{d['pmc_id'] for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    lexicon=enhance.OntologyLexicon(args.hpo,train,mode='morph',synonyms='related')
    preds={d['pmc_id']:d for d in baseline.read_jsonl(args.pred)}
    out=[];stats={}
    for d in inputs:
        p,s=correct(d,preds[d['pmc_id']],lexicon,not args.keep_unknown);out.append(p);stats[d['pmc_id']]=s
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in out))
    target.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(args),'stats':stats,'source_sha256':baseline.sha256(__file__),'enhance_source_sha256':baseline.sha256(enhance.__file__),'inputs':{k:baseline.sha256(getattr(args,k)) for k in ['train','input','hpo','pred']},'output_sha256':baseline.sha256(target)},indent=2)+'\n')
if __name__=='__main__':main()
