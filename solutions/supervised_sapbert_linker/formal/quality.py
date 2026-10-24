"""One protected full-F1 ID replacement; freeze before the sole publicv2 call."""
import collections
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(s):return ' '.join(s.lower().split())
def put(path,obj):
    with open(path,'x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def replace(base,eligible,results):
    new=copy.deepcopy(base);bydoc={str(d['pmc_id']):d for d in new};old={str(d['pmc_id']):d for d in base}
    q={x['text']:x for x in results};assert len(q)==len(results)
    assert set(q)=={norm(x['text']) for x in eligible}
    changed=set();ambiguous=0
    for x in eligible:
        pmc=x['pmc_id'];i=x['entity_index'];e=bydoc[pmc]['entities'][i]
        assert all(e[k]==x[k] for k in ['text','identifier','offset','length'])
        r=q[norm(x['text'])];assert __import__('math').isfinite(r['score'])
        if len(r['identifiers'])!=1:ambiguous+=1;continue
        hid=r['identifiers'][0];assert __import__('re').fullmatch(r'HP:\d{7}',hid)
        if hid!=e['identifier']:e['identifier']=hid;changed.add((pmc,i))
    # Simultaneous replacements may collide or expose a collision after rollback.
    # Revert changed members only; no baseline entity is deleted or reordered.
    rolled_back=set()
    while True:
        bad=set()
        for pmc,d in bydoc.items():
            groups=collections.defaultdict(list)
            for i,e in enumerate(d['entities']):groups[(e['offset'],e['length'],e['identifier'],e.get('note'))].append(i)
            for indices in groups.values():
                if len(indices)>1:bad.update((pmc,i) for i in indices if (pmc,i) in changed)
        if not bad:break
        for pmc,i in bad:bydoc[pmc]['entities'][i]['identifier']=old[pmc]['entities'][i]['identifier']
        changed-=bad;rolled_back|=bad
    changes=[]
    for before,after in zip(base,new):
        assert {k:v for k,v in before.items() if k!='entities'}=={k:v for k,v in after.items() if k!='entities'}
        assert len(before['entities'])==len(after['entities'])
        for i,(a,b) in enumerate(zip(before['entities'],after['entities'])):
            assert {k:v for k,v in a.items() if k!='identifier'}=={k:v for k,v in b.items() if k!='identifier'}
            if a['identifier']!=b['identifier']:
                assert (str(before['pmc_id']),i) in changed
                changes.append(dict(pmc_id=str(before['pmc_id']),entity_index=i,offset=a['offset'],length=a['length'],old=a['identifier'],new=b['identifier']))
    return new,changes,dict(ambiguous_queries_applied_to_entities=ambiguous,collision_rollbacks=len(rolled_back))


def contracts():
    base=[dict(pmc_id='x',association=[dict(patient_id='p',phenotype=['HP:0000001'])],entities=[
        dict(offset=0,length=3,text='abc',identifier='HP:0000001',note='YES'),
        dict(offset=0,length=3,text='abc',identifier='HP:0000002',note='YES'),
        dict(offset=8,length=3,text='nil',identifier='-1',note='YES')])]
    eligible=[dict(pmc_id='x',entity_index=0,**{k:base[0]['entities'][0][k] for k in ['offset','length','text','identifier']})]
    got,changes,counts=replace(base,eligible,[dict(text='abc',score=.7,identifiers=['HP:0000002'])]);assert got==base and not changes and counts['collision_rollbacks']==1
    got,changes,_=replace(base,eligible,[dict(text='abc',score=.7,identifiers=['HP:0000003'])]);assert len(changes)==1 and got[0]['association']==base[0]['association'] and got[0]['entities'][2]==base[0]['entities'][2]
    got,changes,_=replace(base,eligible,[dict(text='abc',score=.7,identifiers=['HP:0000003','HP:0000004'])]);assert got==base and not changes
    print(json.dumps(dict(status='PASS_DISTINCT_COLLISION_PROTECTION_AND_REPLACEMENT_CONTRACTS',tests=3,models=0,gold_reads=0,quality_calls=0)))


def main(plan_path):
    plan=json.loads(Path(plan_path).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    refs=plan['refs']
    for k in ['baseline','baseline_metrics','input','eligible']:
        assert sha(refs[k]['path'])==refs[k]['sha256']
    ev=Path(plan['evaluator_dir'])
    assert sha(ev/'evaluate.py')=='ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    assert sha(ev/'validate_predictions.py')=='b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f'
    sys.path.insert(0,str(ev));from validate_predictions import load_jsonl,read_hpo,validate_predictions
    from evaluate import evaluate,VERSION,positive_units
    assert VERSION=='patientphex-public-approx-v2'
    actual=Path(plan['collected'])/'formal-output';complete=json.loads((actual/'complete.json').read_text())
    assert complete['status']=='COMPLETE_FRESH3EPOCH219_AND_ALL303_NO_QUALITY' and complete['updates']==219 and complete['held_rows']==303 and complete['bank_refresh_count']==4
    assert sha(actual/'held303-results.json')==complete['held_results']['sha256']
    results=json.loads((actual/'held303-results.json').read_text());assert len(results)==303
    base=load_jsonl(refs['baseline']['path']);inputs=load_jsonl(refs['input']['path']);eligible=json.loads(Path(refs['eligible']['path']).read_text())
    assert len(base)==len(inputs)==16 and len(eligible)==422 and sum(len(x['entities']) for x in base)==1313
    rows,changes,counts=replace(base,eligible,results)
    report=validate_predictions(inputs,rows,hpo=read_hpo(plan['hpo']));assert report['valid']
    put(out/'format.json',report)
    pred=out/'prediction.jsonl'
    if changes:pred.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    else:pred.write_bytes(Path(refs['baseline']['path']).read_bytes())
    freeze=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_sha256=sha(pred),baseline_sha256=refs['baseline']['sha256'],
        formal_complete_sha256=sha(actual/'complete.json'),plan_sha256=sha(plan_path),documents=16,patients=39,entities=1313,
        eligible=422,changed=len(changes),all_nonentity_objects_exact=True,all_span_text_order_unchanged=True,gold_not_read=True,old_v2_calls=0,planned_new_v2_calls=int(bool(changes)))
    put(out/'changes.json',dict(changes=changes,counts=counts));put(out/'freeze.json',freeze)
    baseline=json.loads(Path(refs['baseline_metrics']['path']).read_text());assert baseline['metrics']['Score']==.7286975364690524
    if not changes:
        put(out/'result.json',dict(status='BYTE_EXACT_NO_CHANGE',metrics=baseline['metrics'],changed=0,new_v2_calls=0,old_v2_calls=0,strictly_better=False));return
    # Only this block opens held annotations, after the complete immutable endpoint.
    assert sha(refs['gold']['path'])==refs['gold']['sha256'];gold=load_jsonl(refs['gold']['path'])
    put(out/'quality-start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=sha(out/'freeze.json'),new_v2_calls_planned=1))
    metric=evaluate(gold,rows,body_scope='all',no_id_doc='text',wildcard_no_id=True,validate=False)
    assert metric['metrics']['micF1']==baseline['metrics']['micF1'] and metric['metrics']['macF1']==baseline['metrics']['macF1']
    put(out/'metrics.json',metric)
    by_gold={str(x['pmc_id']):positive_units(x,x,'all') for x in gold};diagnostic=collections.Counter()
    for c in changes:
        ids=by_gold[c['pmc_id']].get((c['offset'],c['length']),set())
        diagnostic[('old_exact' if c['old'] in ids else 'old_not_exact')+'__'+('new_exact' if c['new'] in ids else 'new_not_exact')]+=1
        if '-1' in ids:diagnostic['gold_NIL_wildcard_not_counted_as_exact_HPO']+=1
    result=dict(status='ONE_COMPLETE_F1_ENDPOINT_EVALUATED',baseline_metrics=baseline['metrics'],metrics=metric['metrics'],
        deltas={k:v-baseline['metrics'][k] for k,v in metric['metrics'].items()},changed=len(changes),
        mention_tp_delta=metric['counts']['mention']['tp']-baseline['counts']['mention']['tp'],mention_fp_delta=metric['counts']['mention']['fp']-baseline['counts']['mention']['fp'],
        changed_exact_HPO_membership_diagnostic=dict(diagnostic),strictly_better=metric['metrics']['Score']>baseline['metrics']['Score'],
        new_v2_calls=1,old_v2_calls=0,originaldev_calls=0,official_calls=0,freeze_sha256=sha(out/'freeze.json'))
    put(out/'result.json',result);print(json.dumps(result))


if __name__=='__main__':
    if sys.argv[1:] == ['contracts']:contracts()
    else:main(sys.argv[1])
