"""Same fixed LightGBM fold with eighteen input-only relative public-source features."""
import argparse
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
sys.path.insert(0,'/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector')
from prepare_pool import FEATURES,OLD_FEATURES,sha,ref,checked,read,rows,write,sources,same_old
import configuration
from features import relative_features, FEATURES as RELATIVE_FEATURES
import lightgbm
sys.path.insert(0,'/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies')
import fit_helpers as prior


def keyed(path):
    data=rows(path);out={r['pmc_id']:r for r in data};assert len(out)==len(data);return out

def feature(multi,row,c):
    # Original audited feature recipe; only the six new numeric/bool fields are appended.
    return multi.features(row['input'],c,row['base'])|{n:c[n] for n in OLD_FEATURES+FEATURES}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--fold',type=int,choices=range(4),required=True);a=ap.parse_args()
    started=time.monotonic();plan=read(a.plan);assert plan['execution_ready'] is True;sources(plan);plan_sha=sha(a.plan)
    assert plan['maximum_training_row_ratio']==1.5 and plan['cut']==.3
    fold=plan['folds'][a.fold];assert fold['fold']==a.fold
    prep=read(checked(plan['prepared_manifest']));old=read(checked(plan['old_prepared_manifest']))
    assert prep['old_prepared_manifest']==plan['old_prepared_manifest'] and prep['new_features']==FEATURES
    prepared=prep['folds'][a.fold];original=old['folds'][a.fold]
    for n in ('old_train_rows','old_input_rows','train_ids','input_ids','nested_input_ids'):assert prepared[n]==original[n]
    old_fit=read(checked(fold['old_fit_manifest']));reference=read(checked(fold['reference']))
    assert old_fit['old_reference_sha256']==fold['reference']['sha256']
    legacy,multi,modules=prior.load_modules(plan,old_fit,reference);single=multi.single
    assert legacy.PUBLIC_FEATURES+legacy.BIOFORMER_FEATURES==OLD_FEATURES
    config=reference['config'];assert checked(fold['train_gold']).resolve()==Path(config['train']).resolve()
    assert sha(config['train'])==reference['inputs']['train'] and sha(config['input'])==reference['inputs']['input']
    truth=keyed(config['train']);inputs=keyed(config['input'])
    assert len(truth)==48 and len(inputs)==16 and not truth.keys()&inputs.keys()
    assert set(truth)==set(prepared['train_ids']) and list(inputs)==prepared['input_ids']
    assert all(not {'entities','association'}&set(x) for x in inputs.values())
    specs=read(config['training_manifest']);assert sha(config['training_manifest'])==reference['inputs']['training_manifest']
    assert len(specs)==len(old_fit['training_origins'])==len(prepared['nested_input_ids'])==3
    origins=[];seen=set()
    for i,spec in enumerate(specs):
        actual={k:sha(p) for k,p in spec.items()};assert actual==old_fit['training_origins'][i];origins.append(actual)
        nested_gold=keyed(spec['gold']);generator=keyed(spec['generator_train']);nested_input=keyed(spec['input'])
        ids=set(prepared['nested_input_ids'][i]);assert set(nested_gold)==set(nested_input)==ids and len(ids)==16
        assert set(generator)==set(truth)-ids and len(generator)==32 and not seen&ids
        assert all(doc==truth[did] for did,doc in generator.items())
        for did,doc in nested_gold.items():assert doc==truth[did] and nested_input[did]==prior.strip_labels(doc)
        seen|=ids
    assert seen==set(truth)
    pools={};nold={};nnew={};old_positives=0;xs=[];ys=[]
    for role in ('train','input'):
        before=rows(checked(original[role]));after=rows(checked(prepared[role]));assert len(before)==len(after)==(48 if role=='train' else 16)
        assert [d['pmc_id'] for d in before]==[d['pmc_id'] for d in after]==prepared[role+'_ids']
        nold[role]=nnew[role]=0
        for previous,row in zip(before,after):
            assert {k:v for k,v in previous.items() if k!='candidates'}=={k:v for k,v in row.items() if k!='candidates'}
            did=row['pmc_id'];same_old(row['candidates'],previous['candidates'])
            assert row['input']==(prior.strip_labels(truth[did]) if role=='train' else inputs[did])
            assert row['origin_index']==(next(i for i,x in enumerate(prepared['nested_input_ids']) if did in x) if role=='train' else None)
            nold[role]+=len(previous['candidates']);nnew[role]+=len(row['candidates'])
            if role=='train':
                old_positives+=sum(prior.labels(truth[did],previous['candidates']))
                ys.extend(prior.labels(truth[did],row['candidates']))
                extras=relative_features(row['candidates'])
                xs.extend(feature(multi,row,c)|extra for c,extra in zip(row['candidates'],extras))
        assert nold[role]==original['old_'+role+'_rows'] and nnew[role]==prepared['new_'+role+'_rows']
        pools[role]=after
    assert nold['train']==old_fit['training_examples'] and old_positives==old_fit['training_positive']
    assert len(ys)==nnew['train'] and nold['train']<=len(ys)<=1.5*nold['train']
    vectorizer=single.DictVectorizer(sparse=False);matrix=vectorizer.fit_transform(xs)
    assert set(old_fit['features'])|set(FEATURES)<=set(vectorizer.feature_names_)
    strong=read(checked(plan['strong_biobert_fit_manifests'][a.fold]))
    assert [n for n in vectorizer.feature_names_ if n not in RELATIVE_FEATURES]==strong['features']
    assert set(RELATIVE_FEATURES)<=set(vectorizer.feature_names_) and len(vectorizer.feature_names_)==len(strong['features'])+18
    baseline=read(checked(plan['baseline_lightgbm_fit_manifests'][a.fold]))
    assert baseline['features']==strong['features']
    assert len(ys)==strong['training_examples']==baseline['training_examples'] and sum(ys)==strong['training_positive']==baseline['training_positive']
    assert old_fit['estimator']==reference['estimator']==strong['estimator']
    model=configuration.estimator()
    assert all(model.get_params()[k]==v for k,v in configuration.PARAMETERS.items())
    assert model.get_params()==baseline['estimator']
    sources(plan);assert sha(a.plan)==plan_sha
    target=Path(plan['output_dir'])/('fold'+str(a.fold))/'fitted';target.mkdir(parents=True,exist_ok=False)
    model.fit(matrix,ys)
    assert model.n_iter_==model.booster_.current_iteration()==160 and single.np.array_equal(model.classes_,[0,1])
    assert model.n_features_in_==len(vectorizer.feature_names_)
    with (target/'model-vectorizer.pkl').open('xb') as f:pickle.dump({'model':model,'vectorizer':vectorizer,'features':vectorizer.feature_names_},f,protocol=5)
    reloaded=pickle.loads((target/'model-vectorizer.pkl').read_bytes())
    assert reloaded['features']==vectorizer.feature_names_ and single.np.array_equal(reloaded['model'].classes_,[0,1])
    scored=[]
    for row in pools['input']:
        cs=row['candidates']
        if cs:
            extras=relative_features(cs)
            features=[feature(multi,row,c)|extra for c,extra in zip(cs,extras)];values=vectorizer.transform(features)
            all_probabilities=model.predict_proba(values)
            assert all_probabilities.shape==(len(cs),2) and single.np.isfinite(all_probabilities).all()
            assert single.np.array_equal(all_probabilities,reloaded['model'].predict_proba(reloaded['vectorizer'].transform(features)))
            assert single.np.array_equal(all_probabilities[:,1],model.booster_.predict(values,num_threads=1))
            assert single.np.allclose(all_probabilities.sum(axis=1),1.,rtol=0,atol=1e-15)
            probabilities=all_probabilities[:,1]
        else:probabilities=[]
        scored.append({'pmc_id':row['pmc_id'],'candidates':[dict(c,selector_probability=float(p)) for c,p in zip(cs,probabilities)]})
    with (target/'selector-scores.jsonl').open('x',encoding='utf-8') as f:
        for row in scored:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    sources(plan);assert sha(a.plan)==plan_sha
    manifest=dict(status='fit_and_input_scores_complete_no_decode_or_quality',fold=a.fold,source_sha256=sha(__file__),plan_sha256=plan_sha,
                  prepared_manifest=plan['prepared_manifest'],old_prepared_manifest=plan['old_prepared_manifest'],old_fit_manifest=fold['old_fit_manifest'],reference=fold['reference'],
                  training_origins=origins,training_documents=48,input_documents=16,old_training_examples=nold['train'],training_examples=len(ys),old_training_positive=old_positives,training_positive=sum(ys),
                  old_input_candidates=nold['input'],input_candidates=nnew['input'],features=vectorizer.feature_names_,new_features=list(RELATIVE_FEATURES),relative_feature_source=ref(Path(__file__).with_name('features.py')),
                  old_feature_count=len(strong['features']),actual_feature_count=len(vectorizer.feature_names_),
                  actual_new_feature_names=sorted(set(vectorizer.feature_names_)-set(strong['features'])),
                  estimator=model.get_params(),actual_iterations=int(model.n_iter_),early_stopping_performed=False,threshold=.3,
                  dependencies={m.__name__:sha(m.__file__) for m in modules},label_function_sha256=sha(prior.__file__),
                  outputs={n:ref(target/n) for n in ('selector-scores.jsonl','model-vectorizer.pkl')},
                  runtime={'python':platform.python_version(),'sklearn':single.sklearn.__version__,'numpy':single.np.__version__,'lightgbm':lightgbm.__version__},
                  positive_class_probability_column=1,saved_reload_all_input_probabilities_exact=True,native_probability_column_exact=True,
                  strong_biobert_reference=plan['strong_biobert_fit_manifests'][a.fold],baseline_lightgbm_reference=plan['baseline_lightgbm_fit_manifests'][a.fold],
                  original_dev_read=False,evaluation_gold_read=False,seconds=time.monotonic()-started)
    write(target/'manifest.json',manifest);print(json.dumps({'fold':a.fold,'seconds':manifest['seconds'],'training_rows':len(ys),'actual_feature_count':len(vectorizer.feature_names_)}))

if __name__=='__main__':main()
