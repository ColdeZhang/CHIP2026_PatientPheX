#!/usr/bin/env python3
"""Single F1 meta fold, identical context+LR basis, only ± four frozen SFT facts.

Prepare/produce never opens outer gold. Both prediction files freeze together;
evaluate is once-only and reads gold only after verifying that freeze.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,datetime,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(r):assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],r['path']
def save(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_plan(b):
    plan=read(b/'fusion-plan.json');assert plan['source_sha256']==sha(__file__)
    for r in plan['references'].values():check(r)
    spec=importlib.util.spec_from_file_location('frozen_tree',plan['references']['tree_helper']['path']);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    assert plan['tree_parameters']==h.TREE_PARAMETERS and plan['threshold']==h.THRESHOLD==.5
    h.verify_parser(plan['references']['parser']['path'])
    return plan,h

def verify_sft_response(h,meta,raw,path,prompts_path,adapter,model_source):
    fp=meta['fingerprint'];config=fp['configuration']
    assert fp['adapter_files']==adapter and config['adapter']
    assert fp['source_sha256']==h.RUNNER_SHA and fp['source_manifest_sha256']==h.PINS['model_source']
    assert fp['prompts_sha256']==sha(prompts_path) and meta['output_sha256']==sha(path)
    assert meta['expected_rows']==len(raw) and meta['base_parameters']==8953803264 and meta['total_parameters']==8997081600
    assert config['max_input_tokens']==14000 and config['max_new_tokens']==1600
    assert fp['batch_size']==1 and fp['thinking'] is False and fp['precision']=='text_only_bfloat16_sdpa_no_quantization'
    assert meta['eos_token_ids']==[248044,248046]
    assert all(meta['load_information'].get(k)==[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    expected={f['name']:dict(bytes=f['size'],sha256=f['sha256']) for f in model_source['files']}
    assert len(expected)==15 and fp['model_files']==expected
    for name,version in dict(transformers='5.20.2',peft='0.21.2',tokenizers='0.23.1').items():assert fp['runtime'][name]==version
    for r in raw:
        assert r['removed_demonstrations']==0 and isinstance(r['hit_token_limit'],bool)
        assert 0<r['input_tokens']<=14000 and 0<=r['output_tokens']<=1600

def common(b):
    plan,h=load_plan(b);r=plan['references'];mp=rows(r['meta_prompts']['path']);op=rows(r['outer_prompts']['path'])
    pairs=h.pair_rows(mp,rows(r['meta_labels']['path']));outer=h.pair_rows(op)
    assert len(mp)==80 and len(op)==58 and len(pairs)==1557 and len(outer)==1166
    assert len({p['pmc_id'] for p in pairs})==len({p['pmc_id'] for p in outer})==16
    assert not {p['pmc_id'] for p in pairs}&{p['pmc_id'] for p in outer}
    y=np.asarray([p['target'] for p in pairs],dtype=np.int64);assert int(y.sum())==365
    train=h.join_features(pairs,rows(r['meta_features']['path']),True);test=h.join_features(outer,rows(r['outer_features']['path']),False)
    for f in train+test:
        assert 0<=f['relation_lr_probability']<=1
        assert f['relation_lr_selected']==int(f['relation_lr_probability']>=.55)
    vectorizer,x,z=h.base_matrices(train,test)
    assert all(name in vectorizer.feature_names_ for name in ('relation_lr_probability','relation_lr_selected'))
    raw=rows(r['outer_raw']['path']);manifest=read(r['outer_manifest']['path']);oldcp=read(r['old48_checkpoint']['path']);oldactual=read(r['old48_actual_checkpoint_hashes']['path'])
    expected={k.split('/')[-1]:v for k,v in oldcp['files'].items() if k in ('adapter/adapter_config.json','adapter/adapter_model.safetensors')}
    assert oldcp['completed_epochs']==2 and oldcp['global_step']==52
    # The retained old endpoint's source-side actual SHA audit is independently frozen.
    assert oldactual['epoch-002']['files']==oldcp['files']
    verify_sft_response(h,manifest,raw,r['outer_raw']['path'],r['outer_prompts']['path'],expected,read(r['model_source']['path']))
    flags,stats=h.llm_features(op,raw,outer)
    inputs=rows(r['input']['path']);entities=rows(r['entities']['path'])
    assert sum(len(d['entities']) for d in entities)==947 and len(inputs)==len(entities)==16
    by={str(d['pmc_id']):d for d in inputs};assert set(by)=={p['pmc_id'] for p in outer}
    for p in op:assert set(p['patient_ids'])=={str(pat['patient_id']) for pat in by[str(p['pmc_id'])]['patient']}
    return plan,h,mp,op,pairs,outer,y,vectorizer,x,z,flags,stats,inputs,entities

def prepare(b):
    plan,h,mp,op,pairs,outer,y,v,x,z,flags,stats,inputs,entities=common(b)
    output=dict(status='CPU_prepared_no_meta_fit_no_outergold',at_utc=now(),source=ref(__file__),plan=ref(b/'fusion-plan.json'),
                meta_rows=len(pairs),outer_rows=len(outer),positive_labels=int(y.sum()),base_columns=v.feature_names_,
                train_matrix_sha256=h.matrix_sha(x),outer_matrix_sha256=h.matrix_sha(z),target_sha256=h.matrix_sha(y),
                outer_sft_facts_sha256=h.matrix_sha(flags),outer_sft_parse=stats,common_LR_columns=True,
                feature_vocabulary_fit_only_metaF1=True,raw_HPO_label_convention=True,not_full3fold_OOF=True)
    save(b/'fusion-cpu-prepared.json',output);return output

def produce(b):
    plan,h,mp,op,pairs,outer,y,v,x,z,outerflags,outerstats,inputs,entities=common(b)
    prepared=read(b/'fusion-cpu-prepared.json');assert prepared['plan']==ref(b/'fusion-plan.json')
    assert prepared['train_matrix_sha256']==h.matrix_sha(x) and prepared['outer_matrix_sha256']==h.matrix_sha(z) and prepared['target_sha256']==h.matrix_sha(y)
    f=read(b/'meta-responses-freeze.json');assert f['status']=='complete_F1_SFT32_responses_frozen_before_meta_fit'
    assert f['plan_sha256']==sha(b/'run-plan.json') and f['raw_sha256']==sha(b/'meta.responses.jsonl') and f['manifest_sha256']==sha(b/'meta.responses.manifest.json')
    assert f['actual_inference_plan_sha256']==sha(b/'actual-inference-plan.json')
    inference=read(b/'actual-inference-plan.json');complete=read(b/'formal2ep/training_complete.json');cp=read(b/'formal2ep/epoch-002/checkpoint.json');actual=read(b/'checkpoint-actual-hashes.json')
    assert inference['training_complete_sha256']==sha(b/'formal2ep/training_complete.json') and inference['checkpoint_sha256']==sha(b/'formal2ep/epoch-002/checkpoint.json')
    assert inference['actual_checkpoint_hashes_sha256']==sha(b/'checkpoint-actual-hashes.json')
    assert complete['epochs']==2 and complete['global_step']==32 and complete['source_sha256']==read(b/'run-plan.json')['trainer_sha256']
    assert cp['completed_epochs']==2 and cp['global_step']==32 and cp['files']==actual['epoch-002']['files']
    expected={k.split('/')[-1]:v for k,v in cp['files'].items() if k in ('adapter/adapter_config.json','adapter/adapter_model.safetensors')};assert expected==inference['adapter_files']
    raw=rows(b/'meta.responses.jsonl');meta=read(b/'meta.responses.manifest.json')
    verify_sft_response(h,meta,raw,b/'meta.responses.jsonl',plan['references']['meta_prompts']['path'],expected,read(plan['references']['model_source']['path']))
    flags,stats=h.llm_features(mp,raw,pairs);xx=h.append_llm(x,flags);zz=h.append_llm(z,outerflags)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import joblib
    output=b/'fusion-results';output.mkdir();predictions={};models={};times={};probrefs={}
    with threadpool_limits(limits=1):
        for arm,tx,tz in [('without_llm',x,z),('with_llm',xx,zz)]:
            start=time.monotonic();model=HistGradientBoostingClassifier(**h.TREE_PARAMETERS).fit(tx,y);prob=model.predict_proba(tz)[:,1]
            predictions[arm]=h.construct_predictions(inputs,entities,outer,prob);models[arm]=model;times[arm]=time.monotonic()-start
            probpath=output/(arm+'.probabilities.npy');np.save(probpath,prob);probrefs[arm]=ref(probpath)
    freeze=h.freeze_predictions(output,predictions)
    joblib.dump(dict(models=models,vectorizer=v,llm_columns=h.LLM_COLUMNS,threshold=.5),output/'models.joblib')
    details=dict(status='two_fresh_F1_meta_trees_complete',at_utc=now(),plan=ref(b/'fusion-plan.json'),run_plan=ref(b/'run-plan.json'),
                 cpu_preparation=ref(b/'fusion-cpu-prepared.json'),meta_response_freeze=ref(b/'meta-responses-freeze.json'),
                 tree_source=ref(plan['references']['tree_helper']['path']),source=ref(__file__),columns=v.feature_names_,
                 train_matrix_sha256=h.matrix_sha(x),outer_matrix_sha256=h.matrix_sha(z),target_sha256=h.matrix_sha(y),
                 train_sft_flags_sha256=h.matrix_sha(flags),outer_sft_flags_sha256=h.matrix_sha(outerflags),
                 augmented_train_sha256=h.matrix_sha(xx),augmented_outer_sha256=h.matrix_sha(zz),
                 train_parse=stats,outer_parse=outerstats,shared_prefix_exact=True,threshold=.5,parameters=h.TREE_PARAMETERS,
                 times=times,probabilities=probrefs,models=ref(output/'models.joblib'),original_helper_freeze=ref(output/'predictions-freeze.json'),
                 heldout_gold_read=False,originaldev_read=False)
    save(output/'production.json',details)
    save(b/'paired-predictions-freeze.json',dict(status='both_complete_outer_predictions_frozen_before_gold',at_utc=now(),plan=ref(b/'fusion-plan.json'),
        production=ref(output/'production.json'),original_freeze=ref(output/'predictions-freeze.json'),meta_response_freeze=ref(b/'meta-responses-freeze.json'),
        predictions={arm:ref(v['path']) for arm,v in freeze['files'].items()},references=plan['references'],threshold=.5,heldout_gold_read=False,evaluation_performed=False))
    return dict(status=details['status'],freeze=ref(b/'paired-predictions-freeze.json'),times=times)

def evaluate(b):
    plan,h=load_plan(b);f=read(b/'paired-predictions-freeze.json');assert f['status']=='both_complete_outer_predictions_frozen_before_gold'
    assert f['plan']==ref(b/'fusion-plan.json') and not f['heldout_gold_read'] and not f['evaluation_performed'] and f['threshold']==.5
    for r in [f['production'],f['original_freeze'],f['meta_response_freeze'],*f['predictions'].values()]:check(r)
    assert f['references']==plan['references']
    save(b/'evaluation-started.json',dict(at_utc=now(),freeze=ref(b/'paired-predictions-freeze.json'),two_new_arms_only=True,old_controls_not_reevaluated=True))
    # First access to outer labels, after both complete predictions were frozen.
    assert sha(plan['gold_path'])==plan['historical_gold_sha256']
    r=plan['references'];controls={name:read(r[name+'_metrics']['path']) for name in ('rules','lr','sft48')};arms={}
    for arm,pred in f['predictions'].items():
        vp=b/'fusion-results'/(arm+'.validation.json');ep=b/'fusion-results'/(arm+'.metrics.json')
        commands=[[plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',pred['path'],'--hpo',r['hpo']['path'],'--output',str(vp)],
                  [plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',pred['path'],'--body-scope','all','--no-id-doc','text','--output',str(ep)]]
        for i,cmd in enumerate(commands):
            result=subprocess.run(cmd,text=True,capture_output=True);save(b/'fusion-results'/(arm+f'.command{i}.json'),dict(argv=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0
            if i==0:assert read(vp)['valid']
        metrics=read(ep);assert metrics['documents']==16 and metrics['patients']==47
        assert all(metrics['metrics'][k]==old['metrics'][k] for old in controls.values() for k in ('menF1','docF1'))
        check(pred);arms[arm]=dict(metrics=metrics['metrics'],counts=metrics['counts'],prediction=pred,metrics_file=ref(ep),validation_file=ref(vp))
    result=dict(status='complete_single_meta_fold_control',at_utc=now(),scope='F2F3 SFT32 / F1 HGB-meta16 / outer0 eval16; not complete3foldOOF; previously researched outer0',
                arms=arms,controls={k:dict(metrics=v['metrics'],counts=v['counts']) for k,v in controls.items()},
                delta_with_to_without={k:arms['with_llm']['metrics'][k]-arms['without_llm']['metrics'][k] for k in arms['with_llm']['metrics']},
                delta_with_to_controls={k:arms['with_llm']['metrics']['Score']-v['metrics']['Score'] for k,v in controls.items()},
                freeze=ref(b/'paired-predictions-freeze.json'),production=f['production'],threshold=.5,originaldev_read=False,official_submission=False,
                limitations=['One meta fold only; same F1 fitting for both new arms','SFT32 F1 candidate generators train16; cachedSFT48 outer generated32','Both add LR probabilities to old context basis; old299 tree is not this baseline','No threshold/epoch search'])
    save(b/'paired-summary.json',result);return dict(status=result['status'],arms={k:v['metrics'] for k,v in arms.items()},summary=ref(b/'paired-summary.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','produce','evaluate']);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    result={'prepare':prepare,'produce':produce,'evaluate':evaluate}[a.phase](a.directory.resolve());print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
