#!/usr/bin/env python3
"""Independent relation hypotheses; final entity predictions remain separate."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import run as base


def main():
    p=argparse.ArgumentParser()
    for k in ['input','base','output','extra-output']:p.add_argument('--'+k,required=True)
    p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--votes',type=int,default=2)
    a=p.parse_args();docs=base.read_jsonl(a.input);bp={str(d['pmc_id']):d for d in base.read_jsonl(a.base)};cs=[{str(d['pmc_id']):d for d in base.read_jsonl(path)} for path in a.candidates]
    assert all('entities' not in d and 'association' not in d for d in docs);assert all(set(c)==set(bp) for c in cs);assert set(bp)=={str(d['pmc_id']) for d in docs}
    out=Path(a.output);extra_out=Path(a.extra_output);out.parent.mkdir(parents=True,exist_ok=True);extra_out.parent.mkdir(parents=True,exist_ok=True);stats=Counter();pools=[];extras=[]
    for d in docs:
        pmc=str(d['pmc_id']);pr=bp[pmc];existing={x for e in pr['entities'] for x in base.association_keys(e)};votes=Counter()
        for c in cs:votes.update({x for e in c[pmc]['entities'] for x in base.association_keys(e) if x.startswith('HP:')})
        selected={x for x,n in votes.items() if n>=a.votes and x not in existing};added=[];seen=set()
        for c in cs:
            for e in c[pmc]['entities']:
                ids=selected.intersection(base.association_keys(e))
                if not ids:continue
                key=(e['offset'],e['length'],';'.join(sorted(ids)))
                if key in seen:continue
                seen.add(key);added.append(dict(e,identifier=key[2]))
        pools.append(dict(pr,entities=sorted(pr['entities']+added,key=lambda e:(e['offset'],e['length'],e['identifier']))))
        extras.append(dict(pr,entities=sorted(added,key=lambda e:(e['offset'],e['length'],e['identifier']))))
        stats['new_document_concepts']+=len(selected);stats['supporting_spans']+=len(added);stats['documents_with_new_concepts']+=bool(selected)
    out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pools));extra_out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in extras))
    out.with_suffix('.manifest.json').write_text(json.dumps({'configuration':vars(a),'stats':dict(stats),'source_sha256':base.sha(__file__),'base_source_sha256':base.sha(base.__file__),'inputs':{k:base.sha(getattr(a,k)) for k in ['input','base']},'candidate_sources':{x:base.sha(x) for x in a.candidates},'output_sha256':base.sha(out),'extra_output_sha256':base.sha(extra_out),'purpose':'Relations only: do not substitute these hypotheses for final entity predictions','input_annotations_used':False},indent=2)+'\n');print(json.dumps(dict(stats)))
if __name__=='__main__':main()
