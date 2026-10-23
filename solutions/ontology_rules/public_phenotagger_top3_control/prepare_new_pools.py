"""Uniform existing-boundary top3 ID union; preserve every old candidate field."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

FEATURES=[m+'_top3_'+f for m in ['pubmed','bioformer'] for f in ['p','rank','observed']]
OLD_PUBLIC=[m+'_'+f for m in ['public','bioformer'] for f in ['present','score','neural','dictionary','abbreviation','origin_unknown']]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def rows(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f]
def key(c):return c['offset'],c['length'],c['identifier']
def save(p,values):
    with Path(p).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf8') as f:
                for x in values:f.write(json.dumps(x,ensure_ascii=False)+'\n')


def augment(document,lookup):
    old=document['candidates'];pool={key(c):dict(c) for c in old};assert len(pool)==len(old)
    assert all(not set(FEATURES)&set(c) for c in old)
    spans=defaultdict(list)
    for c in old:spans[c['offset'],c['length']].append(c)
    observed={};added=[]
    for (offset,length),cs in sorted(spans.items()):
        item=lookup[str(document['pmc_id']),offset,length]
        byid=defaultdict(dict)
        for model in ['pubmed','bioformer']:
            for obs in item[model]:
                h=obs['hpo_id'];p=obs['p'];rank=obs['rank']
                assert h.startswith('HP:') and h!='HP:None' and math.isfinite(p) and 0<=p<=1 and rank in [1,2,3]
                assert model not in byid[h]
                byid[h][model]=(p,rank)
        assert len({c['text'] for c in cs})==1
        for h in sorted(byid):
            k=(offset,length,h);observed[k]=byid[h]
            if k not in pool:
                pool[k]=dict(offset=offset,length=length,identifier=h,text=cs[0]['text'],
                    probability=.5,similarity=0.,exact=False,is_base=False,crf=False,
                    crf_span=any(c['crf_span'] for c in cs),semantic_source=False,**{name:False for name in OLD_PUBLIC})
                added.append(k)
    result=[pool[key(c)] for c in old]+[pool[k] for k in sorted(added)]
    for c in result:
        for model in ['pubmed','bioformer']:
            obs=observed.get(key(c),{}).get(model)
            c[model+'_top3_observed']=obs is not None
            c[model+'_top3_p']=obs[0] if obs is not None else 0.
            c[model+'_top3_rank']=obs[1] if obs is not None else 0
    assert [{k:v for k,v in c.items() if k not in FEATURES} for c in result[:len(old)]]==old
    assert {(c['offset'],c['length']) for c in result}==set(spans)
    assert len(result)==len({key(c) for c in result}) and len(result)==len(old)+len(added)
    stats=Counter(old_rows=len(old),new_rows=len(result),added_ID_rows=len(added),old_spans=len(spans))
    byspan=defaultdict(list);newkeys=set(added)
    for c in result:
        byspan[c['offset'],c['length']].append(c)
        for m in ['pubmed','bioformer']:stats[m+'_observed_rows']+=c[m+'_top3_observed']
    for cs in byspan.values():
        if not any(key(c) in newkeys for c in cs):continue
        stats['spans_with_added_IDs']+=1
        vectors=defaultdict(list)
        for c in cs:vectors[tuple(c[f] for f in FEATURES)].append(c)
        collisions=[group for group in vectors.values() if len(group)>1 and any(key(c) in newkeys for c in group)]
        stats['spans_with_new_ID_feature_collision']+=bool(collisions)
        stats['new_ID_rows_in_feature_collision']+=sum(key(c) in newkeys for group in collisions for c in group)
    return dict(document,candidates=result),stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();start=time.monotonic()
    plan=json.loads(Path(a.plan).read_text());assert sha(__file__)==plan['source_sha256']
    for field in ['old_manifest','lookup']:
        info=plan[field];assert sha(info['path'])==info['sha256']
    old=json.loads(Path(plan['old_manifest']['path']).read_text());lookup={}
    for x in rows(plan['lookup']['path']):
        k=(str(x['pmc_id']),x['offset'],x['length']);assert k not in lookup;lookup[k]=x
    whitelist=old['span_whitelist'];assert sha(whitelist['path'])==whitelist['sha256']
    expected={(str(x['pmc_id']),x['offset'],x['length']) for x in map(json.loads,Path(whitelist['path']).read_text().splitlines())}
    assert set(lookup)==expected and len(expected)==old['unique_spans']
    out=Path(plan['output_directory']);out.mkdir(exist_ok=False);folds=[];total=Counter()
    for oldfold in old['folds']:
        target=out/('fold'+str(oldfold['fold']));target.mkdir();stats={};files={}
        for role in ['train','input']:
            source=oldfold[role];assert sha(source['path'])==source['sha256']
            docs=rows(source['path']);new=[];one=Counter()
            for d in docs:
                row,count=augment(d,lookup);new.append(row);one.update(count)
            assert one['old_rows']==oldfold['old_'+role+'_rows']
            assert [d['pmc_id'] for d in new]==oldfold[role+'_ids']
            path=target/(role+'.pool.jsonl.gz');save(path,new);files[role]=ref(path);stats[role]=one;total.update(one)
        ratio=stats['train']['new_rows']/stats['train']['old_rows']
        folds.append(dict(fold=oldfold['fold'],**files,old_train_rows=stats['train']['old_rows'],new_train_rows=stats['train']['new_rows'],
            old_input_rows=stats['input']['old_rows'],new_input_rows=stats['input']['new_rows'],training_row_ratio=ratio,
            train_ids=oldfold['train_ids'],input_ids=oldfold['input_ids'],nested_input_ids=oldfold['nested_input_ids'],stats=stats,
            all_old_key_values_preserved=True,no_new_spans=True,old_candidate_order_kept_as_prefix=True))
    # A resource estimate, not measured new training time. Include six extra
    # dimensions, 25% buffer, and 15s fixed import/cache overhead.
    estimate=15.
    for fold,oldcost in zip(folds,plan['old_costs']):
        assert fold['fold']==oldcost['fold']
        estimate+=1.25*oldcost['seconds']*fold['training_row_ratio']*(oldcost['features']+6)/oldcost['features']
    gate=all(f['training_row_ratio']<=3 for f in folds) and estimate<=240
    manifest=dict(status='label-free uniform top3-ID pools prepared',source_sha256=sha(__file__),plan_sha256=sha(a.plan),
        old_manifest=plan['old_manifest'],lookup=plan['lookup'],folds=folds,total=total,new_features=FEATURES,
        raw_missing='observed=false with p/rank numeric placeholders only, never an observed zero posterior',
        estimated_four_fit_infer_seconds=estimate,estimate_method='15s + 1.25 * sum(old end-to-end seconds * actual row ratio * (oldfeatures+6)/oldfeatures)',
        gate_all_training_rows_at_most_3x=all(f['training_row_ratio']<=3 for f in folds),gate_estimate_at_most_240s=estimate<=240,
        authorized_fit_size_cost_gate_passed=gate,no_gold_read=True,no_new_fit=True,no_new_predictions=True,
        seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(folds=[{k:f[k] for k in ['fold','old_train_rows','new_train_rows','training_row_ratio']} for f in folds],
        estimated_seconds=estimate,gate=gate,total=total,seconds=manifest['seconds'])))


if __name__=='__main__':main()
