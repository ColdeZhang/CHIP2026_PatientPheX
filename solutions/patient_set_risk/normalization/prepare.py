"""Decode all8 immutable responses and collect every necessary uncached quote."""
import argparse
from pathlib import Path
from common import *

def prepare(responses_path,output,assets_path=HERE/'assets.json'):
    rows=load(responses_path);validate_responses(rows)
    assets=load(assets_path)
    for item in assets.values():verify(item)
    # Fixed original train identities cannot be replaced through a path override.
    assert {k:v['sha256'] for k,v in assets.items()}=={k:v['sha256'] for k,v in load(HERE/'assets.json').items()}
    interpret,units,StrictNormalizer,policy,precheck,_=original()
    docs={str(d['pmc_id']):d for d in map(__import__('json').loads,Path(assets['train32']['path']).read_text().splitlines())}
    targets=list(map(__import__('json').loads,Path(assets['targets87']['path']).read_text().splitlines()))
    assert len(docs)==32 and len(targets)==87
    truth={(str(d['pmc_id']),a['patient_id']):atoms(a['phenotype']) for d in docs.values() for a in d['association']}
    assert sum(map(len,truth.values()))==818,'Keep all4 nonconstructible literal gold atoms'
    strict=StrictNormalizer(assets['hpo']['path'],assets['train32']['path'])
    cached_rows=load(assets['cached624']['path']);cached={x['quote']:x for x in cached_rows};assert len(cached)==len(cached_rows)==624
    targets_by={x['row_id']:x for x in targets}
    for index,row_id in SELECTED:assert targets[index]['row_id']==row_id
    decoded=[];needed=set();used_cache=set()
    for _,row_id in SELECTED:
        target=targets_by[row_id];doc=docs[str(target['pmc_id'])]
        assert units(doc)==target['units']
        gold=truth[str(target['pmc_id']),target['target_patient_id']]
        for arm in ('greedy','sample'):
            raw=next(x for x in rows if (x['row_id'],x['arm'])==(row_id,arm))
            result=interpret(raw['response'],doc,raw['EOS'],raw['cap'])
            valid=result['status']=='complete' and not result['rejected']
            anchors=[]
            # Invalid rows are already fixed reward0; their partial anchors are
            # preserved in decoder result, but are not necessary reward queries.
            if valid:
                for anchor in result['anchors']:
                    q=anchor['quote'];s=strict.normalize(q)
                    if s['reason']=='unmatched' and precheck(q,policy) is None:
                        (used_cache if q in cached else needed).add(q)
                    anchors.append(dict(anchor=anchor,strict=s))
            binding=dict(response_utf8_sha256=hashlib.sha256(raw['response'].encode()).hexdigest(),
                         generated_ids_sha256=hashlib.sha256(__import__('json').dumps(raw['generated_ids'],separators=(',',':')).encode()).hexdigest(),
                         generated_count=len(raw['generated_ids']),EOS=raw['EOS'],cap=raw['cap'])
            decoded.append(dict(row_id=row_id,arm=arm,valid=valid,gold=sorted(gold),gold_count=len(gold),decode=result,anchors=anchors,response_binding=binding))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Copy raw bytes first; full original response remains the recovery anchor.
    raw=Path(responses_path).read_bytes()
    with (out/'responses.json').open('xb') as f:f.write(raw);f.flush();__import__('os').fsync(f.fileno())
    queries=save(out/'queries.json',[{'quote':q} for q in sorted(needed)])
    cache=save(out/'used-cache.json',[cached[q] for q in sorted(used_cache)])
    report=dict(schema='scst-normalization-preparation-v1',status='ALL8_PREPARED',source=ref(__file__),common=ref(HERE/'common.py'),
                original_sources=ref(HERE/'reuse-manifest.json'),assets=assets,raw=ref(out/'responses.json'),
                original_raw=ref(responses_path),rows=decoded,query_count=len(needed),queries=queries,used_cache=cache,
                training_gold_atoms=818,all4_bad_literal_gold_preserved=True,no_model_calls=True,no_quality=True)
    save(out/'preparation.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--responses',required=True);p.add_argument('--output',required=True);p.add_argument('--assets',default=str(HERE/'assets.json'));a=p.parse_args()
    r=prepare(a.responses,a.output,a.assets);print(__import__('json').dumps(dict(status=r['status'],query_count=r['query_count'])))
