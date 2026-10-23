#!/usr/bin/env python3
"""One declared global cut calibration on strict selector-OOF train64 scores."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time
import evaluate_public_selector as shared
import select_public_bioformer as selector
base,evaluate=shared.base,shared.evaluate


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
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();r=Path(a.directory);started=time.monotonic()
    plan=json.loads((r/'plan.json').read_text());assert plan['cuts']==[i/10 for i in range(1,10)]
    for path,digest in plan['sources'].items():assert base.sha(path)==digest
    assert base.sha(plan['train64']['path'])==plan['train64']['sha256']
    assert not (r/'decoded-freeze.json').exists() and not (r/'results.json').exists()
    pools=[]
    for spec in plan['folds']:
        for field in ['scores','manifest','baseline_prediction','baseline_metrics']:
            assert base.sha(spec[field]['path'])==spec[field]['sha256']
        scores=base.read_jsonl(spec['scores']['path']);old=base.read_jsonl(spec['baseline_prediction']['path'])
        assert len(scores)==len(old)==16 and [d['pmc_id'] for d in scores]==[d['pmc_id'] for d in old]
        for scored,doc in zip(scores,old):
            cs=scored['candidates'];ps=[c['selector_probability'] for c in cs]
            assert decode(cs,ps,.4)==selector.select(cs,ps)==doc['entities']
        pools.append((spec,scores,old))
    # All nine decodes are frozen before the first annotation-label read below.
    outputs=[];decoded={}
    for i,cut in enumerate(plan['cuts'],start=1):
        target=r/f'cut-{i}';target.mkdir(exist_ok=False);perfold=[]
        for spec,scores,old in pools:
            predictions=[dict(doc,entities=decode(row['candidates'],[c['selector_probability'] for c in row['candidates']],cut)) for row,doc in zip(scores,old)]
            out=target/f'fold{spec["fold"]}.jsonl';out.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in predictions))
            if i==4:assert out.read_bytes()==Path(spec['baseline_prediction']['path']).read_bytes()
            perfold.append({'fold':spec['fold'],'path':str(out),'sha256':base.sha(out),'entities':sum(len(d['entities']) for d in predictions)})
            decoded[(i,spec['fold'])]=predictions
        outputs.append({'cut':cut,'cut_index':i,'predictions':perfold})
    freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=base.sha(r/'plan.json'),source_sha256=base.sha(__file__),new_calibration_labels_read=False,
        all_nine_decodes=outputs,fixed_point_four_full_prediction_byteexact=True,seconds_before_labels=time.monotonic()-started)
    (r/'decoded-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    # Calibration labels are the original train64 only. They are not an
    # unbiased estimate of the decoder selected using these same labels.
    truth={d['pmc_id']:d for d in base.read_jsonl(plan['train64']['path'])};assert len(truth)==64
    curves=[]
    for i,cut in enumerate(plan['cuts'],start=1):
        total={name:Counter(tp=0,fp=0,fn=0) for name in ['mention','document']};folds=[];n=0;seen=set()
        for spec,_,_ in pools:
            counts={name:Counter(tp=0,fp=0,fn=0) for name in total};preds=decoded[(i,spec['fold'])]
            for pred in preds:
                did=pred['pmc_id'];assert did not in seen;seen.add(did)
                one=entity_counts(truth[did],pred)
                for name in counts:counts[name].update(one[name])
            for name in total:total[name].update(counts[name])
            records=sum(len(d['entities']) for d in preds);n+=records
            fold=dict(fold=spec['fold'],**summarize(counts,records));folds.append(fold)
            if i==4:
                prior=json.loads(Path(spec['baseline_metrics']['path']).read_text())['metrics']
                assert fold['menF1']==prior['menF1'] and fold['docF1']==prior['docF1']
        assert seen==set(truth)
        curves.append(dict(cut=cut,cut_index=i,folds=folds,pooled=summarize(total,n)))
    chosen=max(curves,key=lambda x:(x['pooled']['objective'],-abs(x['cut_index']-4),x['cut_index']))
    assert base.sha(plan['baseline_pooled_metrics']['path'])==plan['baseline_pooled_metrics']['sha256']
    prior=json.loads(Path(plan['baseline_pooled_metrics']['path']).read_text())['metrics']
    assert curves[3]['pooled']['menF1']==prior['menF1'] and curves[3]['pooled']['docF1']==prior['docF1']
    result=dict(status='one declared calibration completed',source_sha256=base.sha(__file__),plan_sha256=base.sha(r/'plan.json'),freeze_sha256=base.sha(r/'decoded-freeze.json'),
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),curves=curves,selected_cut=chosen['cut'],selected_pooled=chosen['pooled'],fixed_point_four=curves[3]['pooled'],
        objective='pooled (menF1 + docF1)/2; no relation metric used',tie_break='nearest .4 then higher cut; distance computed on integer grid indices',
        seconds=time.monotonic()-started,originaldev_read=False,new_fit=False,
        limitations=['Scores are strictly held-out final-selector predictions, with nested task-fit candidate generators.',
        'Selection uses all train64 labels; same64 best objective is a calibration fit result, not a fresh independent estimate.',
        'Existing train64 folds were already used for source/method development; no originaldev labels accessed here.',
        'No extra cut/feature/ranking/overlap rule search; fixed .4 predictions and entity metrics exactly replayed.',
        'No assumptions of calibrated probabilities; this is an empirical global operating point for the stated Task1 objective.',
        'Rule relationship fields copied only to preserve record schema; they do not enter this calibration objective.'])
    assert result['seconds']<=120
    (r/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cut':result['selected_cut'],'pooled':result['selected_pooled'],'baseline':result['fixed_point_four'],'seconds':result['seconds']}))


if __name__=='__main__':main()
