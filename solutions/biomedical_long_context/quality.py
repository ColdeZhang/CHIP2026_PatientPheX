"""Original GP protected union, one full F1 freeze then at most one public v2."""
import copy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'normalization'))
from common import read_jsonl,write_json,write_jsonl,ref,sha


def entity_key(e):return (int(e['offset']),int(e['length']),e['identifier'])


def compatible(candidate,existing):
    # Exact reuse of original GP protected-union policy.
    a,b=candidate['offset'],candidate['offset']+candidate['length']
    for old in existing:
        c,d=old['offset'],old['offset']+old['length']
        if max(a,c)>=min(b,d):continue
        if a==c and b==d:return False
        contained=a<=c and d<=b or c<=a and b<=d
        if not contained or candidate['identifier']==old['identifier']:return False
    return True


def main():
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'quality-plan.json').read_text());out=Path(cfg['output'])
    assert not (out/'freeze.json').exists(),'Recover original endpoint; never repeat quality'
    for key in ('baseline','baseline_metrics','input','evaluator'):
        assert sha(cfg[key]['path'])==cfg[key]['sha256'],key
    formal=Path(cfg['formal_collected']);norm=Path(cfg['normalization_collected'])
    fe=json.loads((formal/'job/ended.json').read_text());fc=json.loads((formal/'output/complete.json').read_text())
    ne=json.loads((norm/'job/ended.json').read_text());nc=json.loads((norm/'linked/complete.json').read_text())
    assert fe['returncode']==ne['returncode']==0 and not fe['timed_out'] and not ne['timed_out']
    assert fc['status']=='COMPLETE_FRESH8EP256_AND_ALL17_HELD_RAW' and fc['actual_adam_updates']==256 and fc['held_windows']==17
    assert nc['status']=='COMPLETE_ALL_DECLARED_CANDIDATES_NORMALIZED' and not nc['held_gold_read']
    assert nc['sources']['span_pool']['sha256']==fc['span_pool']['sha256']
    assert nc['sources']['input']['sha256']==cfg['input']['sha256']
    poolpath=norm/'linked/normalized-pool.jsonl';assert sha(poolpath)==nc['normalized_pool']['sha256']
    spec=importlib.util.spec_from_file_location('longcontext_public_v2',cfg['evaluator']['path']);ev=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(Path(cfg['evaluator']['path']).parent));spec.loader.exec_module(ev)
    inputs=read_jsonl(cfg['input']['path']);baseline=read_jsonl(cfg['baseline']['path']);pool=read_jsonl(poolpath)
    bydoc={str(x['pmc_id']):[] for x in inputs}
    for row in pool:
        assert row['pmc_id'] in bydoc
        if row['gp_decision']:bydoc[row['pmc_id']].append(row)
    assert len(inputs)==len(baseline)==16 and sum(len(d['patient']) for d in inputs)==39
    result=[];added=[]
    for old in baseline:
        new=copy.deepcopy(old);pmc=str(old['pmc_id']);keys={entity_key(e) for e in old['entities']}
        for row in sorted(bydoc[pmc],key=lambda x:(-x['gp_logit'],-x['length'],x['offset'],x['identifier'])):
            if row['identifier']=='-1':continue
            e={k:row[k] for k in ('identifier','offset','length','text')};e.update(type='Phenotype',note=None)
            if entity_key(e) in keys or not compatible(e,new['entities']):continue
            new['entities'].append(e);keys.add(entity_key(e));added.append({'pmc_id':pmc,**e,'gp_logit':row['gp_logit']})
        assert new['entities'][:len(old['entities'])]==old['entities']
        assert {k:v for k,v in new.items() if k!='entities'}=={k:v for k,v in old.items() if k!='entities'}
        result.append(new)
    report=ev.validate_predictions(inputs,result);assert report['valid'],report
    write_json(out/'format.json',report)
    if added:write_jsonl(out/'prediction.jsonl',result)
    else:(out/'prediction.jsonl').write_bytes(Path(cfg['baseline']['path']).read_bytes())
    write_jsonl(out/'added-entities.jsonl',added)
    write_json(out/'freeze.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'source':ref(__file__),'plan':ref(root/'quality-plan.json'),
       'prediction':ref(out/'prediction.jsonl'),'formal_complete':ref(formal/'output/complete.json'),'normalization_complete':ref(norm/'linked/complete.json'),
       'input':cfg['input'],'baseline':cfg['baseline'],'gold_opened_before_freeze':False,'relation_objects_exact':True,'existing_entity_prefix_exact':True})
    old=json.loads(Path(cfg['baseline_metrics']['path']).read_text())
    if added:
        write_json(out/'quality-start.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'new_v2':1,'old_v2':0})
        assert sha(cfg['gold']['path'])==cfg['gold']['sha256']
        metric=ev.evaluate(read_jsonl(cfg['gold']['path']),result);write_json(out/'metrics.json',metric);calls=1
    else:
        assert sha(out/'prediction.jsonl')==cfg['baseline']['sha256'];metric=old;write_json(out/'metrics.json',metric);calls=0
    write_json(out/'result.json',{'status':'COMPLETE_FIXED_PRIMARY_ONLY','baseline':old['metrics'],'new':metric['metrics'],
       'delta':{k:metric['metrics'][k]-old['metrics'][k] for k in old['metrics']},'retain':metric['metrics']['Score']>old['metrics']['Score'],
       'added_entities':len(added),'entities':sum(len(x['entities']) for x in result),'documents':16,'patients':39,
       'new_v2':calls,'old_v2':0,'relations_exact':True,'old_entities_exact_prefix':True,
       'interpretation':'Fixed fresh32 joint backbone/tokenizer/longcontext/passage-loss control; oldstrong selector trained48 excludes F1. No causal isolation of context, no originaldev result.'})
    print((out/'result.json').read_text())


if __name__=='__main__':main()
