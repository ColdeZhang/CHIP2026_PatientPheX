"""Fixed nine-cut training calibration of prehybrid raw-source scores; no new fits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def checked(item):
    path=Path(item['path'])
    assert path.is_absolute() and sha(path)==item['sha256'],str(path)
    if 'bytes' in item:assert path.stat().st_size==item['bytes'],str(path)
    return path


def rows(path):
    result=[json.loads(x) for x in Path(path).read_text().splitlines()]
    ids=[row['pmc_id'] for row in result]
    assert len(ids)==len(set(ids)),str(path)
    return result


def ref(path):
    return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def nonentities(document):
    return {key:value for key,value in document.items() if key!='entities'}


def same_ref(left,right):
    assert left['sha256']==right['sha256']
    assert Path(left['path']).resolve()==Path(right['path']).resolve()


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


def choose(curves):
    return max(curves,key=lambda c:(c['pooled']['objective'],-abs(c['cut_index']-4),c['cut_index']))


def numeric_deltas(after,before):
    assert after.keys()==before.keys()
    return {key:numeric_deltas(value,before[key]) if isinstance(value,dict)
            else value-before[key] for key,value in after.items()}


def entity_changes(truth,before,after):
    count=Counter({kind+'_'+role+'_'+outcome:0 for kind in ['mention','document']
                   for role in ['added','removed'] for outcome in ['tp','fp']})
    assert before['pmc_id']==after['pmc_id']==truth['pmc_id']
    gu,au,bu=[evaluate.positive_units(row,truth,'all') for row in [truth,before,after]]
    for role,left,right in [('added',bu,au),('removed',au,bu)]:
        diff={span:ids-right.get(span,set()) for span,ids in left.items()
              if ids-right.get(span,set())}
        tp,fp,_=evaluate.mention_counts(gu,diff)
        count['mention_'+role+'_tp']+=tp;count['mention_'+role+'_fp']+=fp
    gt,at,bt=[evaluate.document_concepts(row,truth,'all','text') for row in [truth,before,after]]
    for role,diff in [('added',bt-at),('removed',at-bt)]:
        tp,fp,_=evaluate.compare_sets(gt,diff)
        count['document_'+role+'_tp']+=tp;count['document_'+role+'_fp']+=fp
    return count


def verify_sources(plan):
    assert str(Path(__file__).resolve()) in plan['sources']
    for path,digest in plan['sources'].items():assert sha(path)==digest,path


def timeout_handler(signum,frame):
    raise TimeoutError('Fixed decode/calibration 30-second budget exceeded')


def main():
    global evaluate
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    start=time.monotonic();plan_path=Path(a.plan).resolve();plan_sha=sha(plan_path)
    plan=json.loads(plan_path.read_text());root=Path(plan['output_dir'])
    assert root.is_absolute() and not root.exists()
    assert plan['cuts']==[i/10 for i in range(1,10)] and plan['CPU_cap_seconds']==30
    old_handler=signal.signal(signal.SIGALRM,timeout_handler)
    signal.setitimer(signal.ITIMER_REAL,30)
    try:
        run(plan,plan_path,plan_sha,root,start)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)


def run(plan,plan_path,plan_sha,root,start):
    global evaluate
    verify_sources(plan)
    if 'python' in plan:assert Path(sys.executable).resolve()==Path(plan['python']).resolve()
    fit_plan_path=checked(plan['fit_plan']);fit_plan=json.loads(fit_plan_path.read_text())
    complete_path=checked(plan['complete_freeze']);complete=json.loads(complete_path.read_text())
    assert complete['plan_sha256']==plan['fit_plan']['sha256']
    assert complete['all_four_predictions_complete'] is True
    assert complete['new_quality_evaluation_started'] is False
    fit_source=Path(__file__).resolve().with_name('fit_control.py')
    run_source=Path(__file__).resolve().with_name('run_control.py')
    for path,digest in fit_plan['sources'].items():assert sha(path)==digest,path
    assert complete['source_sha256']==fit_plan['sources'][str(run_source)]
    assert sha(fit_source)==fit_plan['sources'][str(fit_source)]
    assert [s['fold'] for s in complete['folds']]==[0,1,2,3]
    assert [s['fold'] for s in fit_plan['folds']]==[0,1,2,3]
    assert [s['fold'] for s in plan['folds']]==[0,1,2,3]
    checked(plan['old_calibration_results'])
    sys.path.insert(0,plan['evaluation_source']);evaluate=importlib.import_module('evaluate')
    assert Path(evaluate.__file__).resolve().parent==Path(plan['evaluation_source']).resolve()
    assert sha(evaluate.__file__)==plan['sources'][str(Path(evaluate.__file__).resolve())]
    assert evaluate.VERSION=='patientphex-public-approx-v2'
    pools=[];seen=set()
    for spec,frozen,fit_spec in zip(plan['folds'],complete['folds'],fit_plan['folds']):
        for field in ['scores','manifest','baseline_prediction','old_template']:checked(spec[field])
        for local,frozen_role in [('scores','scores'),('manifest','new_manifest'),
                                 ('baseline_prediction','new_prediction'),('old_template','baseline_prediction')]:
            same_ref(spec[local],frozen[frozen_role])
        same_ref(spec['old_template'],fit_spec['baseline_prediction'])
        scores=rows(spec['scores']['path']);preds=rows(spec['baseline_prediction']['path'])
        old=rows(spec['old_template']['path'])
        ids=[d['pmc_id'] for d in preds]
        assert len(ids)==16 and not set(ids)&seen;seen.update(ids)
        assert ids==[d['pmc_id'] for d in old]==[d['pmc_id'] for d in scores]
        manifest=json.loads(Path(spec['manifest']['path']).read_text())
        assert manifest['source_sha256']==fit_plan['sources'][str(fit_source)]
        assert manifest['plan_sha256']==plan['fit_plan']['sha256']
        assert manifest['prepared_manifest_sha256']==fit_plan['prepared_manifest']['sha256']
        assert manifest['actual_iterations']==160 and manifest['threshold']==.3
        assert manifest['early_stopping_performed'] is False
        old_fit=json.loads(checked(fit_spec['old_fit_manifest']).read_text())
        assert manifest['features']==sorted(old_fit['features']+[f'raw_{m}_{f}' for m in ('pubmed','bioformer') for f in ('observed','p')])
        assert manifest['new_features']==[f'raw_{m}_{f}' for m in ('pubmed','bioformer') for f in ('observed','p')]
        assert manifest['assertions']['old_pool_fields_keys_order_exact'] is True
        assert manifest['old_training_examples']==old_fit['training_examples'] and manifest['training_examples']>=old_fit['training_examples']
        assert manifest['old_training_positive']==old_fit['training_positive']
        assert manifest['outputs']['selector-scores.jsonl']==spec['scores']['sha256']
        assert manifest['outputs']['augmented.jsonl']==spec['baseline_prediction']['sha256']
        for scored,pred,prior in zip(scores,preds,old):
            cs=scored['candidates']
            assert decode(cs,[c['selector_probability'] for c in cs],.3)==pred['entities']
            assert nonentities(pred)==nonentities(prior)
        pools.append((spec,scores,preds,old))
    assert len(seen)==64
    root.mkdir()
    decoded={};outputs=[]
    for index,cut in enumerate(plan['cuts'],1):
        target=root/f'cut-{index}';target.mkdir();one=[]
        for spec,scores,old,_ in pools:
            preds=[dict(doc,entities=decode(row['candidates'],[c['selector_probability'] for c in row['candidates']],cut)) for row,doc in zip(scores,old)]
            assert all(nonentities(a)==nonentities(b) for a,b in zip(preds,old))
            p=target/f'fold{spec["fold"]}.jsonl'
            with p.open('x') as f:
                for row in preds:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index==3:assert p.read_bytes()==Path(spec['baseline_prediction']['path']).read_bytes()
            decoded[index,spec['fold']]=preds;one.append(dict(fold=spec['fold'],**ref(p)))
        outputs.append(dict(cut=cut,cut_index=index,predictions=one))
    assert len(outputs)==9 and all(len(item['predictions'])==4 for item in outputs)
    verify_sources(plan);assert sha(plan_path)==plan_sha
    for item in outputs:
        for prediction in item['predictions']:checked(prediction)
    freeze=dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
                plan_sha256=plan_sha,fit_complete_freeze=plan['complete_freeze'],all36=outputs,
                new_calibration_labels_read=False,anchor_cut03_byteexact=True,
                all_nonentity_fields_oldcut03_exact=True,seconds_before_labels=time.monotonic()-start)
    write(root/'decoded-freeze.json',freeze)
    write(root/'calibration-start.json',dict(started_at=datetime.now(timezone.utc).isoformat(),
          plan_sha256=plan_sha,decoded_freeze=ref(root/'decoded-freeze.json'),
          all36_predictions_verified_before_labels=True))
    gold=rows(checked(plan['train64']));truth={d['pmc_id']:d for d in gold}
    assert len(gold)==64 and set(truth)==seen
    curves=[];fixed_changes=Counter();fixed_fold_changes={}
    for index,cut in enumerate(plan['cuts'],1):
        total={n:Counter(tp=0,fp=0,fn=0) for n in ['mention','document']};folds=[];n=0
        for spec,_,_,old_template in pools:
            counts={name:Counter(tp=0,fp=0,fn=0) for name in total}
            preds=decoded[index,spec['fold']]
            for pred in preds:
                one=entity_counts(truth[pred['pmc_id']],pred)
                for name in counts:counts[name].update(one[name])
            for name in total:total[name].update(counts[name])
            records=sum(len(d['entities']) for d in preds);n+=records
            fold=dict(fold=spec['fold'],**summarize(counts,records));folds.append(fold)
            if index==3:
                changes=Counter()
                for before,after in zip(old_template,preds):changes.update(entity_changes(truth[after['pmc_id']],before,after))
                fixed_changes.update(changes)
                fixed_fold_changes[spec['fold']]=dict(changes)
        curves.append(dict(cut=cut,cut_index=index,folds=folds,pooled=summarize(total,n)))
    chosen=choose(curves)
    old=json.loads(checked(plan['old_calibration_results']).read_text());oldchosen=choose(old['curves'])
    assert oldchosen['cut']==old['selected_cut'] and oldchosen['pooled']==old['selected_pooled']
    assert oldchosen['pooled']['objective']==.7801925961542502
    assert [f['fold'] for f in oldchosen['folds']]==[0,1,2,3]
    compare=dict(old_selected_cut=oldchosen['cut'],old_selected_pooled=oldchosen['pooled'],
                 new_selected_cut=chosen['cut'],new_selected_pooled=chosen['pooled'],
                 deltas=numeric_deltas(chosen['pooled'],oldchosen['pooled']),
                 objective_delta=chosen['pooled']['objective']-oldchosen['pooled']['objective'],
                 fold_deltas=[dict(fold=a['fold'],**numeric_deltas(
                     {k:v for k,v in b.items() if k!='fold'},{k:v for k,v in a.items() if k!='fold'}))
                     for a,b in zip(oldchosen['folds'],chosen['folds'])])
    old03=next(curve for curve in old['curves'] if curve['cut']==.3)
    assert [fold['fold'] for fold in old03['folds']]==[0,1,2,3]
    fixed_folds=[]
    for previous,current in zip(old03['folds'],curves[2]['folds']):
        prior={k:v for k,v in previous.items() if k!='fold'}
        new={k:v for k,v in current.items() if k!='fold'}
        fixed_folds.append(dict(fold=current['fold'],original_bioformer=prior,prehybrid=new,
                               deltas=numeric_deltas(new,prior),changes=fixed_fold_changes[current['fold']]))
    fixed=dict(cut=.3,folds=fixed_folds,pooled=dict(original_bioformer=old03['pooled'],prehybrid=curves[2]['pooled'],
               deltas=numeric_deltas(curves[2]['pooled'],old03['pooled']),changes=dict(fixed_changes)),
               versus_old_selected=dict(old_selected_cut=oldchosen['cut'],
                   old_selected_pooled=oldchosen['pooled'],prehybrid=curves[2]['pooled'],
                   deltas=numeric_deltas(curves[2]['pooled'],oldchosen['pooled'])))
    result=dict(status='fixed prehybrid raw-source nine-cut training calibration complete',source_sha256=sha(__file__),
                plan_sha256=plan_sha,freeze=ref(root/'decoded-freeze.json'),curves=curves,
                comparison=compare,fixed03_comparison=fixed,selected_cut=chosen['cut'],
                selected_pooled=chosen['pooled'],anchor_cut03=curves[2]['pooled'],
                finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,
                runtime=dict(python=platform.python_version(),executable=str(Path(sys.executable).resolve())),
                new_fit=False,originaldev_read=False,original_bioformer_baseline_preserved=True,
                relation_scored=False,all_nonentity_fields_oldcut03_exact=True,
                limitations=['All train64 labels participate in calibration; selected maximum is not independent generalization or competition Score.',
                             'The same original Bioformer pools, prespecified nine cuts and tie rule are used; the only fit increment is new original-gate prehybrid raw candidates and four real source features on additions only.',
                             'Nonentities retain oldcut.3 fields and .4-origin associations; only Task1 entity metrics enter the objective.'])
    assert result['seconds']<=30
    write(root/'results.json',result)
    print(json.dumps(dict(comparison=compare,fixed03_comparison=fixed,seconds=result['seconds'])))


if __name__=='__main__':main()
