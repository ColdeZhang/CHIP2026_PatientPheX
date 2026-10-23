"""Separate training-side nine-cut diagnostic; no new model fits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def checked(ref):
    p=Path(ref['path']);assert p.is_absolute() and sha(p)==ref['sha256']
    return p

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]

def ref(path):
    return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def decode(candidates, probabilities, cut):
    assert len(candidates)==len(probabilities)
    selected=[]
    for c,p in sorted(zip(candidates,probabilities),key=lambda cp:(-cp[1],-cp[0]['length'],cp[0]['offset'])):
        assert math.isfinite(float(p)) and 0<=p<=1
        if p<cut or any(c['offset']<e['offset']+e['length'] and e['offset']<c['offset']+c['length'] for e in selected):
            continue
        selected.append({k:c[k] for k in ['identifier','offset','length','text']}|{'type':'Phenotype','note':None})
    return sorted(selected,key=lambda e:e['offset'])

def entity_counts(truth,guess):
    # Exactly the existing v2 all/text entity primitives; no relation scoring.
    mention=evaluate.mention_counts(evaluate.positive_units(truth,truth,'all'),evaluate.positive_units(guess,truth,'all'))
    document=evaluate.compare_sets(evaluate.document_concepts(truth,truth,'all','text'),evaluate.document_concepts(guess,truth,'all','text'))
    return {'mention':dict(zip(['tp','fp','fn'],mention)),'document':dict(zip(['tp','fp','fn'],document))}

def summarize(counts, records):
    details={name:evaluate.prf(**values) for name,values in counts.items()}
    men,doc=details['mention']['f1'],details['document']['f1']
    return dict(menF1=men,docF1=doc,objective=(men+doc)/2,counts=details,entity_records=records)

def main():
    global evaluate
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic()
    plan=json.loads(Path(a.plan).read_text());root=Path(plan['output_dir'])
    assert not root.exists();root.mkdir()
    assert plan['cuts']==[i/10 for i in range(1,10)] and plan['CPU_cap_seconds']==30
    for path,digest in plan['sources'].items():assert sha(path)==digest,path
    sys.path.insert(0,plan['evaluation_source']);evaluate=importlib.import_module('evaluate')
    assert sha(evaluate.__file__)==plan['sources'][str(Path(evaluate.__file__).resolve())]
    assert evaluate.VERSION=='patientphex-public-approx-v2'
    assert [s['fold'] for s in plan['folds']]==[0,1,2,3]
    pools=[];seen=set()
    for spec in plan['folds']:
        for field in ['scores','manifest','baseline_prediction','baseline_metrics','old_template']:checked(spec[field])
        scores=rows(spec['scores']['path']);preds=rows(spec['baseline_prediction']['path']);old=rows(spec['old_template']['path'])
        ids=[d['pmc_id'] for d in preds];assert len(ids)==len(set(ids))==16 and not set(ids)&seen;seen.update(ids)
        assert ids==[d['pmc_id'] for d in old]==[d['pmc_id'] for d in scores]
        manifest=json.loads(Path(spec['manifest']['path']).read_text())
        assert manifest['outputs']['selector-scores.jsonl']==spec['scores']['sha256']
        assert manifest['outputs']['augmented.jsonl']==spec['baseline_prediction']['sha256']
        for scored,pred,prior in zip(scores,preds,old):
            cs=scored['candidates'];assert decode(cs,[c['selector_probability'] for c in cs],.3)==pred['entities']
            assert {k:v for k,v in pred.items() if k!='entities'}=={k:v for k,v in prior.items() if k!='entities'}
        pools.append((spec,scores,preds))
    assert len(seen)==64
    decoded={};outputs=[]
    for index,cut in enumerate(plan['cuts'],1):
        target=root/f'cut-{index}';target.mkdir();one=[]
        for spec,scores,old in pools:
            preds=[dict(doc,entities=decode(row['candidates'],[c['selector_probability'] for c in row['candidates']],cut)) for row,doc in zip(scores,old)]
            p=target/f'fold{spec["fold"]}.jsonl'
            with p.open('x') as f:
                for row in preds:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index==3:assert p.read_bytes()==Path(spec['baseline_prediction']['path']).read_bytes()
            decoded[index,spec['fold']]=preds;one.append(dict(fold=spec['fold'],**ref(p)))
        outputs.append(dict(cut=cut,cut_index=index,predictions=one))
    freeze=dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),plan_sha256=sha(a.plan),all36=outputs,new_calibration_labels_read=False,anchor_cut03_byteexact=True,all_nonentity_fields_oldcut03_exact=True,seconds_before_labels=time.monotonic()-start)
    write(root/'decoded-freeze.json',freeze)
    truth={d['pmc_id']:d for d in rows(checked(plan['train64']))};assert set(truth)==seen
    curves=[]
    for index,cut in enumerate(plan['cuts'],1):
        total={n:Counter(tp=0,fp=0,fn=0) for n in ['mention','document']};folds=[];n=0
        for spec,_,_ in pools:
            counts={name:Counter(tp=0,fp=0,fn=0) for name in total};preds=decoded[index,spec['fold']]
            for pred in preds:
                one=entity_counts(truth[pred['pmc_id']],pred)
                for name in counts:counts[name].update(one[name])
            for name in total:total[name].update(counts[name])
            records=sum(len(d['entities']) for d in preds);n+=records;fold=dict(fold=spec['fold'],**summarize(counts,records));folds.append(fold)
            if index==3:
                prior=json.loads(Path(spec['baseline_metrics']['path']).read_text())['metrics']
                assert fold['menF1']==prior['menF1'] and fold['docF1']==prior['docF1']
        curves.append(dict(cut=cut,cut_index=index,folds=folds,pooled=summarize(total,n)))
    chosen=max(curves,key=lambda c:(c['pooled']['objective'],-abs(c['cut_index']-4),c['cut_index']))
    old=json.loads(checked(plan['old_calibration_results']).read_text())
    oldchosen=max(old['curves'],key=lambda c:(c['pooled']['objective'],-abs(c['cut_index']-4),c['cut_index']))
    assert oldchosen['cut']==old['selected_cut'] and oldchosen['pooled']==old['selected_pooled']
    compare=dict(old_selected_cut=oldchosen['cut'],old_selected_pooled=oldchosen['pooled'],new_selected_cut=chosen['cut'],new_selected_pooled=chosen['pooled'],objective_delta=chosen['pooled']['objective']-oldchosen['pooled']['objective'],fold_deltas=[dict(fold=a['fold'],objective_delta=b['objective']-a['objective'],menF1_delta=b['menF1']-a['menF1'],docF1_delta=b['docF1']-a['docF1']) for a,b in zip(oldchosen['folds'],chosen['folds'])])
    result=dict(status='separate fixed nine-cut diagnostic complete',source_sha256=sha(__file__),plan_sha256=sha(a.plan),freeze=ref(root/'decoded-freeze.json'),curves=curves,comparison=compare,selected_cut=chosen['cut'],selected_pooled=chosen['pooled'],anchor_cut03=curves[2]['pooled'],finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,new_fit=False,originaldev_read=False,original_fixed03_negative_unchanged=True,limitations=['All train64 labels participate in calibration; selected maximum is not independent generalization or competition Score.','Same prespecified nine cuts as prior baseline; no new ranking, feature, overlap or model search.','Nonentities retain oldcut.3 fields and .4-origin associations; only two Task1 entity metrics enter objective.'])
    assert result['seconds']<=30
    write(root/'results.json',result);print(json.dumps(dict(comparison=compare,seconds=result['seconds'])))

if __name__=='__main__':main()
