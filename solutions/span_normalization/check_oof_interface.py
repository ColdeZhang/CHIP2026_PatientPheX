"""Load-only check of the exact strong feature matrix/decoder; zero new fits or v2."""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'1')
from common import write_json, read_jsonl, ref, sha
from fit_oof_features import module, checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);ap.add_argument('--hpo',required=True);a=ap.parse_args()
    start=time.monotonic();signal.alarm(60);plan=json.loads(Path(a.plan).read_text())
    p=json.loads(checked(plan['old_formal_plan']).read_text());m=json.loads(checked(plan['strong_fit_manifest']).read_text())
    for path,h in p['sources'].items():assert sha(path)==h
    reference=json.loads(checked(plan['old_fit']).read_text());lf=json.loads(checked(p['folds'][1]['old_fit_manifest']).read_text())
    deps=Path(p['ontology_source']).parents[1];sys.path.insert(0,str(deps));helpers=module('gp_interface_helpers',deps/'fit_helpers.py')
    legacy,multi,_=helpers.load_modules(p,lf,reference);np=multi.single.np
    fields=legacy.PUBLIC_FEATURES+legacy.BIOFORMER_FEATURES+['biobert_present','biobert_score','biobert_neural','biobert_dictionary','biobert_abbreviation','biobert_origin_unknown']
    def feature(r,c):return multi.features(r['input'],c,r['base'])|{n:c[n] for n in fields}
    prepared=json.loads(checked(plan['prepared_manifest']).read_text())['folds'][1]
    with gzip.open(checked(prepared['train']),'rt') as f:train=[json.loads(x) for x in f]
    with gzip.open(checked(prepared['input']),'rt') as f:held=[json.loads(x) for x in f]
    truth={x['pmc_id']:x for x in read_jsonl(reference['config']['train'])}
    xs=[];ys=[]
    for row in train:
        assert row['input']==helpers.strip_labels(truth[row['pmc_id']])
        xs.extend(feature(row,c) for c in row['candidates']);ys.extend(helpers.labels(truth[row['pmc_id']],row['candidates']))
    model=pickle.loads(checked(m['outputs']['model-vectorizer.pkl']).read_bytes())
    oldmat=model['vectorizer'].transform(xs)
    newvec=multi.single.DictVectorizer(sparse=False)
    newmat=newvec.fit_transform([x|{'gp_span_valid':False,'gp_span_logit':0.} for x in xs])
    columns=[newvec.feature_names_.index(n) for n in model['features']]
    assert len(model['features'])==101 and len(newvec.feature_names_)==103 and np.array_equal(oldmat,newmat[:,columns])
    assert len(ys)==45403 and sum(ys)==2978
    scores={x['pmc_id']:x['candidates'] for x in read_jsonl(checked(m['outputs']['selector-scores.jsonl']))}
    base={x['pmc_id']:x for x in read_jsonl(checked(plan['strong_baseline']))}
    sys.path.insert(0,p['nested_source']);import decode_nested_selector as decoder
    _,allowed,_,_=decoder.prior.base.parse_hpo(a.hpo);allowed.discard('HP:0000118')
    for row in held:
        c=scores[row['pmc_id']];expected=np.array([x['selector_probability'] for x in c])
        actual=model['model'].predict_proba(model['vectorizer'].transform([feature(row,x) for x in row['candidates']]))[:,1]
        assert np.array_equal(expected,actual)
        greedy=decoder.greedy(c,.3);entities,_=decoder.decode(c,greedy,row['input'],allowed)
        assert entities==base[row['pmc_id']]['entities']
    result={'status':'PASS_EXISTING_STRONG_INTERFACE','source':ref(__file__),'plan':ref(a.plan),
            'old_101_training_feature_columns_exact_in_103':True,'rows':45403,'positive_rows':2978,
            'all_16595_old_input_probabilities_exact':True,'all_16_old_nested_entity_lists_exact':True,
            'new_classifier_fits':0,'new_v2_calls':0,'evaluation_gold_read':False,'seconds':time.monotonic()-start}
    write_json(a.output,result);signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
