#!/usr/bin/env python3
"""Freeze both fixed-cut complete predictions before the only heldout evaluation."""
import argparse,copy,datetime,gzip,hashlib,importlib.util,json,math,signal,sys,time
from pathlib import Path

def ref(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):return json.loads(Path(p).read_bytes())
def rows(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;tick=time.monotonic();p=load(b/'endpoint-plan.json');plan=load(b/'formal-plan.json');term=b/'terminal'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU endpoint30')));signal.alarm(30)
    assert not (b/'prediction-freeze.json').exists() and not (b/'evaluation-started.json').exists()
    assert ref(__file__)['sha256']==p['source_sha256']
    for r in p['references'].values():assert ref(r['path'])==r
    assert ref(b/'formal-plan.json')==p['formal_plan']
    complete=load(term/'training-complete.json');assert complete['plan_sha256']==ref(b/'formal-plan.json')['sha256'] and not complete['quality_evaluation']
    assert load(term/'gpu-release.json')['target_empty'] and complete['seconds']<=1800
    with gzip.open(b/'F1-input-documents.json.gz','rt') as f:held=json.load(f)
    expected={(d['pmc_id'],pid,cid) for d in held for pid,cid in d['pairs']};assert len(expected)==1557
    inputs=rows(p['references']['input']['path']);old=rows(p['references']['old_sft_pred']['path']);oldmap={str(x['pmc_id']):x for x in old}
    assert {str(x['pmc_id']) for x in inputs}==set(oldmap)==set(plan['heldout_ids']) and sum(len(x['entities']) for x in old)==1189
    predictions={};freezes={}
    for arm in plan['arms']:
        receipt=next(x for x in complete['arms'] if x['mode']==arm);assert receipt['steps']==256 and len(receipt['epochs'])==8
        probability=term/arm/'probabilities.json';assert ref(probability)['sha256']==receipt['probabilities']['sha256']
        values=load(probability);keys=[(x['pmc_id'],x['patient_id'],x['concept']) for x in values];assert len(keys)==len(set(keys))==1557 and set(keys)==expected
        decisions={k:float(v['probability']) for k,v in zip(keys,values)};assert all(math.isfinite(x) and 0<=x<=1 for x in decisions.values())
        pred=[]
        for doc in inputs:
            pmc=str(doc['pmc_id']);row=copy.deepcopy(oldmap[pmc]);row['association']=[{'patient_id':x['patient_id'],'phenotype':sorted(cid for (d,pid,cid),pr in decisions.items() if d==pmc and pid==str(x['patient_id']) and pr>=.5)} for x in doc['patient']]
            assert {k:v for k,v in row.items() if k!='association'}=={k:v for k,v in oldmap[pmc].items() if k!='association'};pred.append(row)
        dest=b/(arm+'.predictions.jsonl')
        with dest.open('x') as f:
            for row in pred:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        predictions[arm]=pred;freezes[arm]=ref(dest)
    save(b/'prediction-freeze.json',{'at_utc':stamp(),'plan_sha256':ref(b/'endpoint-plan.json')['sha256'],'predictions':freezes,'both_models_and_probabilities_frozen_before_gold':True,'entities':1189,'old_nonassociation_exact':True})
    sys.path.insert(0,str(Path(p['references']['evaluator']['path']).parent));evaluator=module('pair_endpoint_evaluator',p['references']['evaluator']['path']);validator=module('pair_endpoint_validator',p['references']['validator']['path'])
    hpo=validator.read_hpo(p['references']['hpo']['path']);formats={arm:validator.validate_predictions(inputs,pred,hpo) for arm,pred in predictions.items()}
    assert all(v['valid'] and v['error_count']==0 for v in formats.values())
    save(b/'evaluation-started.json',{'at_utc':stamp(),'freeze':ref(b/'prediction-freeze.json'),'gold_not_yet_read':True})
    assert ref(p['gold_path'])['sha256']==p['gold_sha256'];gold=rows(p['gold_path']);results={}
    for arm,pred in predictions.items():
        metric=evaluator.evaluate_report(gold,pred);metric.update(gold_sha256=p['gold_sha256'],prediction_sha256=freezes[arm]['sha256']);save(b/(arm+'.metrics.json'),metric);save(b/(arm+'.format.json'),formats[arm]);results[arm]=metric
    baseline={n:load(p['references'][n+'_metrics']['path']) for n in ['old_sft','old_rules']}
    assert baseline['old_sft']['prediction_sha256']==p['references']['old_sft_pred']['sha256'] and baseline['old_rules']['prediction_sha256']==p['references']['old_rules_pred']['sha256']
    comparator={**{k:v['metrics'] for k,v in baseline.items()},'uniform':results['uniform']['metrics']};pair=results['pair']['metrics']
    keep=all(pair['Score']>m['Score'] and pair['macF1']>=m['macF1'] for m in comparator.values())
    report={'status':'COMPLETE_TWO_MATCHED_FINAL_ARMS_UNIQUE_F1','at_utc':stamp(),'seconds':time.monotonic()-tick,'new_metrics':{k:v['metrics'] for k,v in results.items()},'counts':{k:v['counts'] for k,v in results.items()},'comparators':comparator,'pair_delta_vs':{k:{n:pair[n]-m[n] for n in pair} for k,m in comparator.items()},'keep_pair':keep,'no_threshold_scan':True,'originaldev_accessed':False,'predictions':freezes,'formal_plan':ref(b/'formal-plan.json'),'endpoint_plan':ref(b/'endpoint-plan.json'),'training_complete':ref(term/'training-complete.json')}
    save(b/'results.json',report);signal.alarm(0);print(json.dumps(report))

if __name__=='__main__':main()
