"""One fixed LightGBM refit: unchanged strong pool plus strict OOF GP span features."""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'1')
from common import read_jsonl, write_json, write_jsonl, ref, sha


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj); return obj


def checked(r):
    assert sha(r['path']) == r['sha256']; return Path(r['path'])


def key(doc, c): return (str(doc), int(c['offset']), int(c['length']))


def main():
    ap=argparse.ArgumentParser()
    for name in ('plan','actual','output','hpo','gold'): ap.add_argument('--'+name,required=True)
    a=ap.parse_args();start=time.monotonic();signal.alarm(120);out=Path(a.output)
    assert not (out/'fit-started.json').exists(), 'Existing fit intent: recover, do not retrain'
    plan=json.loads(Path(a.plan).read_text());oldplan=json.loads(checked(plan['old_formal_plan']).read_text())
    for p,h in oldplan['sources'].items(): assert sha(p)==h
    for p,h in plan['sources'].items(): assert sha(p)==h
    assert plan['threshold']==.3 and plan['same_old_features']==101
    manifest=json.loads(checked(plan['strong_fit_manifest']).read_text());reference=json.loads(checked(plan['old_fit']).read_text())
    legacyfit=json.loads(Path(oldplan['folds'][1]['old_fit_manifest']['path']).read_text())
    dependencies=Path(oldplan['ontology_source']).parents[1]
    sys.path.insert(0,str(dependencies))
    helpers=module('gp_fixed_helpers',dependencies/'fit_helpers.py')
    legacy,multi,modules=helpers.load_modules(oldplan,legacyfit,reference);single=multi.single;np=single.np
    prepfile=checked(plan['prepared_manifest']);prepared=json.loads(prepfile.read_text())['folds'][1]
    pools={}
    for role in ('train','input'):
        with gzip.open(checked(prepared[role]),'rt') as f:pools[role]=[json.loads(x) for x in f]
    truth={str(x['pmc_id']):x for x in read_jsonl(reference['config']['train'])}
    assert len(truth)==48 and set(truth).isdisjoint(plan['outer1_held_ids'])
    # Four complete target passes came from model groups excluding their target IDs.
    scores={};reports=[];actual=Path(a.actual)
    terminal=json.loads((actual/'run/terminal.json').read_text())
    assert terminal['status']=='COMPLETE' and len(terminal['steps'])==7
    assert all(s['returncode']==0 and not s['timed_out'] for s in terminal['steps'])
    for job in plan['jobs']:
        target=actual/'run'/job['name']/'scores';report=json.loads((target/'complete.json').read_text())
        data=target/'target-scores.jsonl';assert sha(data)==report['output']['sha256']
        assert report['status']=='COMPLETE_ALL_TARGET_BOUNDARIES' and report['no_score_floor'] and not report['held_gold_read']
        expected=read_jsonl(checked(job['targets']));values=read_jsonl(data)
        assert len(values)==len(expected)==job['target_spans']
        assert [{k:v[k] for k in ('pmc_id','offset','length','text')} for v in values]==expected
        assert set(job['train_ids']).isdisjoint(job['input_ids']) and set(job['train_ids']).isdisjoint(plan['outer1_held_ids'])
        for v in values:
            k=key(v['pmc_id'],v);assert k not in scores;scores[k]=v
            assert type(v['gp_span_valid']) is bool and type(v['gp_span_logit']) is float
            assert v['gp_span_valid'] or v['gp_span_logit']==0. and v['missing_reason'] in ('unaligned_endpoint','no_common_window')
        reports.append(ref(target/'complete.json'))
    old_fields=legacy.PUBLIC_FEATURES+legacy.BIOFORMER_FEATURES+['biobert_present','biobert_score','biobert_neural','biobert_dictionary','biobert_abbreviation','biobert_origin_unknown']
    def features(row,c,new):
        f=multi.features(row['input'],c,row['base'])|{n:c[n] for n in old_fields}
        if new:
            s=scores[key(row['pmc_id'],c)];f.update({n:s[n] for n in plan['new_features']})
        return f
    old=pickle.loads(checked(manifest['outputs']['model-vectorizer.pkl']).read_bytes())
    old_scores={r['pmc_id']:r['candidates'] for r in read_jsonl(checked(manifest['outputs']['selector-scores.jsonl']))}
    for row in pools['input']:
        fs=[features(row,c,False) for c in row['candidates']]
        actual_old=old['model'].predict_proba(old['vectorizer'].transform(fs))[:,1]
        expected_old=np.array([c['selector_probability'] for c in old_scores[row['pmc_id']]])
        assert np.array_equal(actual_old,expected_old)
        assert [c for c in row['candidates']]==[{k:v for k,v in c.items() if k!='selector_probability'} for c in old_scores[row['pmc_id']]]
    xs=[];ys=[]
    for row in pools['train']:
        assert row['input']==helpers.strip_labels(truth[row['pmc_id']])
        ys.extend(helpers.labels(truth[row['pmc_id']],row['candidates']))
        xs.extend(features(row,c,True) for c in row['candidates'])
    assert len(ys)==45403 and sum(ys)==2978
    vec=single.DictVectorizer(sparse=False);matrix=vec.fit_transform(xs)
    assert len(vec.feature_names_)==103 and set(vec.feature_names_)==set(old['features'])|set(plan['new_features'])
    configpath=next(p for p in oldplan['sources'] if p.endswith('/lightgbm_selector_control/configuration.py'))
    configuration=module('gp_fixed_lgb_configuration',configpath);model=configuration.estimator()
    assert model.get_params()==manifest['estimator']
    write_json(out/'fit-started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'plan':ref(a.plan),'source':ref(__file__),
                                    'old_all_candidate_probabilities_exact':True,'new_fit_count':1,'input_labels_used':False})
    model.fit(matrix,ys);assert model.n_iter_==160 and np.array_equal(model.classes_,[0,1])
    modelpath=out/'model-vectorizer.pkl';modelpath.write_bytes(pickle.dumps({'model':model,'vectorizer':vec,'features':vec.feature_names_},protocol=5))
    with modelpath.open('rb') as f:os.fsync(f.fileno())
    reloaded=pickle.loads(modelpath.read_bytes());scored=[]
    for row in pools['input']:
        fs=[features(row,c,True) for c in row['candidates']];mat=vec.transform(fs);p=model.predict_proba(mat)[:,1]
        assert np.array_equal(p,reloaded['model'].predict_proba(reloaded['vectorizer'].transform(fs))[:,1])
        scored.append({'pmc_id':row['pmc_id'],'candidates':[dict(c,**{n:scores[key(row['pmc_id'],c)][n] for n in plan['new_features']},selector_probability=float(v)) for c,v in zip(row['candidates'],p)]})
    write_jsonl(out/'selector-scores.jsonl',scored)
    sys.path.insert(0,oldplan['nested_source']);import decode_nested_selector as decoder
    _,allowed,_,_=decoder.prior.base.parse_hpo(a.hpo);allowed.discard('HP:0000118')
    base={x['pmc_id']:x for x in read_jsonl(checked(plan['strong_baseline']))};new=[]
    for row,s in zip(pools['input'],scored):
        doc=row['input'];pmc=row['pmc_id']
        old_greedy=decoder.greedy(old_scores[pmc],.3)
        old_nested,_=decoder.decode(old_scores[pmc],old_greedy,doc,allowed)
        assert old_nested==base[pmc]['entities']
        greedy=decoder.greedy(s['candidates'],.3);entities,_=decoder.decode(s['candidates'],greedy,doc,allowed)
        result=copy.deepcopy(base[pmc]);result['entities']=entities
        assert all(result[k]==base[pmc][k] for k in result if k!='entities');new.append(result)
    sys.path.insert(0,oldplan['evaluation_source']);import evaluate as ev
    assert sha(ev.__file__)=='ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    fmt=ev.validate_predictions([r['input'] for r in pools['input']],new);assert fmt['valid'];write_json(out/'format.json',fmt)
    write_jsonl(out/'prediction.jsonl',new)
    write_json(out/'complete-freeze.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'source':ref(__file__),'plan':ref(a.plan),
        'prediction':ref(out/'prediction.jsonl'),'scores':ref(out/'selector-scores.jsonl'),'model':ref(modelpath),
        'all_gp_passes':reports,'features':vec.feature_names_,'old_candidate_rows_exact':True,'old_probabilities_and_decoded_entities_exact':True,
        'same_nonentity_objects':True,'input_gold_opened':False,'old_metrics_not_reevaluated':True})
    write_json(out/'evaluation-intent.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'new_v2_calls':1})
    metrics=ev.evaluate(read_jsonl(a.gold),new);write_json(out/'metrics.json',metrics)
    oldmetrics=json.loads(checked(plan['strong_metrics']).read_text())['metrics']
    assert metrics['metrics']['micF1']==oldmetrics['micF1'] and metrics['metrics']['macF1']==oldmetrics['macF1']
    result={'status':'COMPLETE_FIXED_OOF_GP_FEATURE_CONTROL','baseline':oldmetrics,'metrics':metrics['metrics'],
        'delta':{k:metrics['metrics'][k]-oldmetrics[k] for k in oldmetrics},'retain':metrics['metrics']['Score']>oldmetrics['Score'],
        'source':ref(__file__),'freeze':ref(out/'complete-freeze.json'),'same_old_features':101,'new_features':plan['new_features'],
        'training_rows':len(ys),'positive_rows':sum(ys),'input_rows':sum(len(x['candidates']) for x in scored),
        'new_v2_calls':1,'old_v2_calls':0,'fit_count':1,'seconds':time.monotonic()-start,'original_dev_read':False}
    write_json(out/'result.json',result);signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
