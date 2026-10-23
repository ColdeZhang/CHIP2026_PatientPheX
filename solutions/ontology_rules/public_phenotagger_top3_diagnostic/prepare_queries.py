"""Freeze diagnostic queries only; never produce candidate/prediction records."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fn-plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    p=json.loads(Path(a.fn_plan).read_text());out=Path(a.output);assert not out.exists()
    for ref in p['references'].values():assert sha(ref['path'])==ref['sha256']
    e=Path(p['references']['evaluator']['path']);sys.path.insert(0,str(e.parent))
    spec=importlib.util.spec_from_file_location('ev',e);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    gold={str(d['pmc_id']):d for d in rows(p['references']['train64']['path'])}
    queries=[]
    for f in p['folds']:
        scores=rows(p['references'][f['scores_key']]['path']);preds=rows(p['references'][f['prediction_key']]['path'])
        assert [str(x['pmc_id']) for x in scores]==[str(x['pmc_id']) for x in preds]
        for sr,pr in zip(scores,preds):
            did=str(sr['pmc_id']);g=gold[did];gu=ev.positive_units(g,g,'all');pu=ev.positive_units(pr,g,'all')
            for span,ids in sorted(gu.items()):
                same=[c for c in sr['candidates'] if (c['offset'],c['length'])==span]
                poolids={v for c in same for v in c['identifier'].split(';')}
                for target in sorted(ids-pu.get(span,set())):
                    if target=='-1' or not same or target in poolids:continue
                    offset,length=span
                    passages=[(i,t) for i,t in enumerate(g['full_text']) if t['offset']<=offset and offset+length<=t['offset']+len(t['text'])]
                    assert len(passages)==1,(did,span)
                    pi,passage=passages[0];local=offset-passage['offset']
                    queries.append(dict(query_index=len(queries),fold=f['fold'],pmc_id=did,offset=offset,length=length,
                        passage_index=pi,local_start=local,local_end=local+length,text=passage['text'][local:local+length],
                        gold_literal_id=target,gold_span_atomic_ids=sorted(ids)))
    assert len(queries)==298
    assert len({(q['pmc_id'],q['offset'],q['length'],q['gold_literal_id']) for q in queries})==298
    out.write_text(json.dumps({'scope':'diagnostic queries, not predictions or candidates','source_sha256':sha(__file__),
        'fn_plan_sha256':sha(a.fn_plan),'count':len(queries),'queries':queries},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(queries=len(queries),spans=len({(q['pmc_id'],q['offset'],q['length']) for q in queries}),
        folds=Counter(q['fold'] for q in queries),sha256=sha(out),bytes=out.stat().st_size)))


if __name__=='__main__':main()
