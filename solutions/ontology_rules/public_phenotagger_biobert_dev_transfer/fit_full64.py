"""One full64 HGB160 fit; no dev inputs, inference, gold or threshold selection."""
import argparse,importlib,json,os,pickle,platform,sys,time
from pathlib import Path
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(n,'1')
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector';sys.path.insert(0,str(COMMON))
from prepare_pool import sha,ref,checked,read,rows,write,sources,FEATURES,OLD_FEATURES
from fit_control import feature,prior

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();p=read(a.plan);sources(p)
 prep=read(checked(p['prepared_train']));old=read(checked(p['old_full64_fit']));reference=read(checked(p['reference']));cfg=reference['config']
 assert prep['old_full64_fit']==p['old_full64_fit'] and prep['reference']==p['reference'] and old['old_reference_sha256']==p['reference']['sha256']
 sys.path.insert(0,p['ontology_source']);multi=importlib.import_module('calibrate_multi');single=multi.single
 modules=[multi,single,single.baseline,single.enhance,single.merge_semantic]
 assert all(sha(m.__file__)==old['dependencies'][m.__name__] for m in modules)
 assert sha(cfg['train'])==reference['inputs']['train'] and checked(p['training_gold'])==Path(cfg['train'])
 truth={x['pmc_id']:x for x in rows(cfg['train'])};pool=rows(checked(prep['training_pool']));assert len(truth)==len(pool)==64 and set(truth)==set(prep['training_ids'])
 specs=read(cfg['training_manifest']);assert sha(cfg['training_manifest'])==reference['inputs']['training_manifest']
 for i,spec in enumerate(specs):
  assert {k:sha(v) for k,v in spec.items()}==old['training_origins'][i]==prep['training_origins'][i]['actual_sources']
  pure={x['pmc_id']:x for x in rows(spec['input'])};gold={x['pmc_id']:x for x in rows(spec['gold'])};generators={x['pmc_id']:x for x in rows(spec['generator_train'])}
  assert set(gold)==set(pure)==set(prep['training_origins'][i]['input_ids']) and len(gold)==16
  assert set(generators)==set(truth)-set(gold) and len(generators)==48
  assert all(d==truth[k] for k,d in gold.items()) and all(d==truth[k] for k,d in generators.items())
  assert all(prior.strip_labels(gold[k])==d for k,d in pure.items())
 xs=[];ys=[]
 for row in pool:
  assert row['input']==prior.strip_labels(truth[row['pmc_id']]);ys.extend(prior.labels(truth[row['pmc_id']],row['candidates']));xs.extend(feature(multi,row,c) for c in row['candidates'])
 assert len(xs)==len(ys)==62281
 vec=single.DictVectorizer(sparse=False);matrix=vec.fit_transform(xs);assert set(old['features'])|set(FEATURES)<=set(vec.feature_names_)
 model=single.HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=5.,learning_rate=.06,random_state=20261023,early_stopping=False)
 assert model.get_params()==old['estimator']==reference['estimator'];out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
 model.fit(matrix,ys);assert model.n_iter_==160 and not model.do_early_stopping_
 with (out/'model-vectorizer.pkl').open('xb') as f:pickle.dump({'model':model,'vectorizer':vec,'features':vec.feature_names_},f,protocol=5)
 sources(p);result={'status':'full64_model_frozen_before_any_new_dev_inference_or_gold','plan_sha256':sha(a.plan),'source_sha256':sha(__file__),'prepared_train':p['prepared_train'],'old_full64_fit':p['old_full64_fit'],'reference':p['reference'],'training_ids':prep['training_ids'],'training_origins':prep['training_origins'],'training_examples':len(ys),'training_positive':sum(ys),'old_features':old['features'],'features':vec.feature_names_,'actual_feature_count':len(vec.feature_names_),'new_features':FEATURES,'estimator':model.get_params(),'actual_iterations':int(model.n_iter_),'model':ref(out/'model-vectorizer.pkl'),'runtime':{'python':platform.python_version(),'sklearn':single.sklearn.__version__,'numpy':single.np.__version__},'seconds':time.monotonic()-start,'original_dev_input_or_gold_read':False}
 write(out/'manifest.json',result);print(json.dumps({k:result[k] for k in ('status','training_examples','training_positive','actual_feature_count','seconds')}))
if __name__=='__main__':main()
