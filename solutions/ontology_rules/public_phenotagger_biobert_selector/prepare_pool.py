"""Input-only exact-key BioBERT increment over the frozen eight old pools."""
import argparse
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

FEATURES = ['biobert_'+x for x in ('present','score','neural','dictionary','abbreviation','origin_unknown')]
OLD_FEATURES = [p+x for p in ('public_','bioformer_') for x in ('present','score','neural','dictionary','abbreviation','origin_unknown')]
WEIGHT = '1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def ref(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def checked(r):
    p=Path(r['path']);assert p.is_absolute() and sha(p)==r['sha256'],str(p)
    if 'bytes' in r: assert p.stat().st_size==r['bytes']
    return p

def read(p): return json.loads(Path(p).read_text())
def rows(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def key(c): return c['offset'],c['length'],c['identifier']
def fullkey(c): return c['pmc_id'],c['offset'],c['length'],c.get('hpo_id',c.get('identifier'))
def defaults(): return {n:(0. if n=='biobert_score' else False) for n in FEATURES}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def sources(plan):
    for p,h in plan['sources'].items():assert sha(p)==h,p

def same_old(current, previous):
    assert len(current)>=len(previous)
    for new,old in zip(current,previous):
        assert set(new)==set(old)|set(FEATURES)
        projected={k:v for k,v in new.items() if k not in FEATURES}
        assert projected==old and all(type(projected[k]) is type(v) for k,v in old.items())

def merge(previous, supplied, doc):
    old=previous['candidates'];out=[dict(c,**defaults()) for c in old]
    assert all(not set(FEATURES)&set(c) for c in old)
    indexed={key(c):c for c in out};assert len(indexed)==len(out)
    crf_spans={key(c)[:2] for c in old if c['crf_span']}
    for c in supplied:
        k=key(c);offset,length=k[:2]
        passages=[p for p in doc['full_text'] if p['offset']<=offset and offset+length<=p['offset']+len(p['text'])]
        assert len(passages)==1
        p=passages[0];left=offset-p['offset'];assert p['text'][left:left+length]==c['text']
        if k not in indexed:
            target={n:c[n] for n in ('offset','length','identifier','text')}
            target.update(probability=.5,similarity=0.,exact=False,is_base=False,crf=False,
                          crf_span=k[:2] in crf_spans,semantic_source=False,
                          **{n:False for n in OLD_FEATURES},**defaults())
            indexed[k]=target;out.append(target)
        target=indexed[k];assert target['text']==c['text']
        target.update({n:c[n] for n in FEATURES})
    same_old(out,old);assert len({key(c) for c in out})==len(out)
    return dict(previous,candidates=out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic()
    plan=read(a.plan);assert plan['execution_ready'] is True;sources(plan)
    assert plan['maximum_training_row_ratio']==1.5
    old=read(checked(plan['old_prepared_manifest']))
    cm=read(checked(plan['cache']['manifest']))
    assert cm['model_weight_sha256']==WEIGHT and cm['public_model_training_on_task'] is False
    assert cm['model_type']=='biobert' and cm['counts']['documents']==64
    assert cm['input_sha256']=='ee2523843ff61f524db313fb2f9c37707a858e40f252870a7a05f4e7c1b3c246'
    assert cm['fixed_policy']['ngram_per_sentence_min']==2 and cm['fixed_policy']['ngram_per_sentence_max']==8
    assert cm['fixed_policy']['threshold_strictly_greater_than']==.95
    canonical=rows(checked(plan['cache']['canonical']));provenance=rows(checked(plan['cache']['provenance']))
    for role,name in [('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]:
        assert plan['cache'][role]['sha256']==cm['outputs'][name]['sha256']
    evidence={fullkey(x):x for x in provenance};assert len(evidence)==len(provenance)==5903
    grouped={};actual_scores=defaultdict(set)
    for c in canonical:
        k=fullkey(c);assert k in evidence
        assert math.isfinite(c['score']) and 0<=c['score']<=1
        actual_scores[k].add(c['score'])
        if k not in grouped:grouped[k]=dict(pmc_id=k[0],offset=k[1],length=k[2],identifier=k[3],text=c['text'])
        assert grouped[k]['text']==c['text']
    assert grouped.keys()==evidence.keys()
    supplied=defaultdict(list)
    for k,c in grouped.items():
        e=evidence[k];assert e['cache_origin']=='biobert_train64_public_protocol'
        assert sorted(actual_scores[k])==e['actual_hybrid_scores']
        assert all(type(e[n]) is bool for n in ('raw_neural_match','raw_dictionary_match','abbrev_inferred'))
        assert e['raw_neural_match'] or e['raw_dictionary_match'] or e['abbrev_inferred']
        c.update(biobert_present=True,biobert_score=float(max(actual_scores[k])),biobert_neural=e['raw_neural_match'],
                 biobert_dictionary=e['raw_dictionary_match'],biobert_abbreviation=e['abbrev_inferred'],biobert_origin_unknown=False)
        supplied[k[0]].append(c)
    out=Path(plan['output_dir']);out.mkdir(parents=True,exist_ok=False);folds=[];total_old=total_new=0
    for prior in old['folds']:
        fold=copy.deepcopy(prior);fd=out/('fold'+str(prior['fold']));fd.mkdir()
        for role in ('train','input'):
            before=rows(checked(prior[role]));after=[];nold=nnew=0
            assert [x['pmc_id'] for x in before]==prior[role+'_ids']
            for row in before:
                assert set(row['input'])=={'pmc_id','pmid','patient','full_text'} and row['pmc_id']==row['input']['pmc_id']
                result=merge(row,supplied[row['pmc_id']],row['input']);after.append(result)
                nold+=len(row['candidates']);nnew+=len(result['candidates'])
            assert nold==prior['old_'+role+'_rows']
            if role=='train':assert nold<=nnew<=1.5*nold,(prior['fold'],nold,nnew)
            target=fd/(role+'.pool.jsonl.gz');target.write_bytes(gzip.compress(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in after).encode(),mtime=0))
            fold[role]=ref(target);fold['new_'+role+'_rows']=nnew;fold[role+'_row_ratio']=nnew/nold
            total_old+=nold;total_new+=nnew
        folds.append(fold)
    assert total_old==244794
    historical_seconds=max(read(checked(r))['fit_infer_elapsed_seconds'] for r in plan['historical_cost_states'])
    ratio=max(f['train_row_ratio'] for f in folds)
    estimate=historical_seconds*ratio*1.25+10.
    result=dict(status='prepared_input_only_no_fit_or_quality',source_sha256=sha(__file__),plan_sha256=sha(a.plan),old_prepared_manifest=plan['old_prepared_manifest'],
                cache=plan['cache'],folds=folds,new_features=FEATURES,old_rows=total_old,new_rows=total_new,added_rows=total_new-total_old,
                old_candidate_projection_values_types_order_exact=True,new_key_order='first appearance in frozen native canonical cache',
                score_semantics='actual native hybrid maximum on exact key, not raw classifier posterior',
                estimated_four_fit_infer_seconds=estimate,historical_four_fit_infer_seconds=historical_seconds,
                estimate_basis='max completed oldSap/prehybrid four-fit wall seconds * maximum train row ratio *1.25 +10s; conservative estimate, not measured BioBERT selector speed',
                no_new_NN=True,no_labels_read=True,seconds=time.monotonic()-start)
    sources(plan);write(out/'manifest.json',result);print(json.dumps({k:result[k] for k in ['old_rows','new_rows','added_rows','seconds']}))

if __name__=='__main__':main()
