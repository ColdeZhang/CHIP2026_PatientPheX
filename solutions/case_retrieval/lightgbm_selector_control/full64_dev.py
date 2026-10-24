"""One same-configuration full64 fit and a179-only entity replacement on dev16."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def ref(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def checked(spec):
    actual=ref(spec['path']);assert all(actual[k]==spec[k] for k in ('path','sha256','bytes')),spec
    return Path(spec['path'])
def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as stream:
        return [json.loads(line) for line in stream if line.strip()]
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def execute(plan_path):
    start=time.monotonic();signal.alarm(120)
    p=read(plan_path);assert p['total_CPU_cap_seconds']==120 and p['cut']==.3
    for spec in p['sources']:checked(spec)
    checked(p['cv_terminal_review']);runtime=read(checked(p['runtime_contract']));checked(runtime['native_library'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=Path(p['output_directory']);root.mkdir(exist_ok=False)
    state=dict(status='one_fresh_full64_running',pid=os.getpid(),started_utc=now(),plan=ref(plan_path))
    write(root/'execution-state.json',state)
    try:
        sys.path.insert(0,p['module_directories']['configuration']);import configuration
        sys.path.insert(0,p['module_directories']['common']);from fit_control import feature,prior
        sys.path.insert(0,p['module_directories']['ontology']);multi=importlib.import_module('calibrate_multi');single=multi.single
        import lightgbm
        assert lightgbm.__version__=='4.6.0'
        legacy=read(checked(p['lineage']['legacy_full64_manifest']))
        for module in (multi,single,single.baseline,single.enhance,single.merge_semantic):
            assert ref(module.__file__)['sha256']==legacy['dependencies'][module.__name__]
        prep=read(checked(p['lineage']['training_preparation']))
        oldfit=read(checked(p['lineage']['strong_full64_manifest']))
        dev_manifest=read(checked(p['lineage']['dev_pool_manifest']))
        assert oldfit['prepared_train']==p['lineage']['training_preparation']
        assert prep['training_pool']==p['training_pool'] and prep['old_full64_fit']==p['lineage']['legacy_full64_manifest']
        assert dev_manifest['model_manifest']==p['lineage']['strong_full64_manifest']
        assert dev_manifest['output']['input.pool.jsonl.gz']==p['dev_pool'] and dev_manifest['input']==p['dev_input']
        assert oldfit['features']==dev_manifest['features'] and len(oldfit['features'])==103
        bundle=pickle.loads(checked(oldfit['model']).read_bytes())
        vectorizer=bundle['vectorizer']
        assert bundle['features']==vectorizer.feature_names_==oldfit['features']
        assert vectorizer.sparse is False and len(vectorizer.vocabulary_)==103
        assert all(vectorizer.vocabulary_[name]==i for i,name in enumerate(oldfit['features']))
        del bundle
        train=rows(checked(p['training_pool']));truth={x['pmc_id']:x for x in rows(checked(p['training_gold']))}
        assert len(train)==len(truth)==64 and set(truth)==set(oldfit['training_ids'])==set(prep['training_ids'])
        assert len({x['pmc_id'] for x in train})==64
        xs=[];ys=[]
        for row in train:
            assert row['input']==prior.strip_labels(truth[row['pmc_id']])
            ys.extend(prior.labels(truth[row['pmc_id']],row['candidates']))
            xs.extend(feature(multi,row,c) for c in row['candidates'])
        assert len(xs)==len(ys)==oldfit['training_examples']==62281 and sum(ys)==oldfit['training_positive']==4196
        matrix=vectorizer.transform(xs);assert matrix.shape==(62281,103)
        matrix_sha=hashlib.sha256(matrix.tobytes(order='C')).hexdigest()
        model=configuration.estimator();model.fit(matrix,ys)
        assert model.n_iter_==model.booster_.current_iteration()==160 and single.np.array_equal(model.classes_,[0,1])
        assert model.n_features_in_==103
        model_path=root/'model-vectorizer.pkl'
        model_path.write_bytes(pickle.dumps(dict(model=model,vectorizer=vectorizer,features=vectorizer.feature_names_),protocol=5))
        reloaded=pickle.loads(model_path.read_bytes())
        assert reloaded['features']==vectorizer.feature_names_ and single.np.array_equal(reloaded['model'].classes_,[0,1])
        fit_seconds=time.monotonic()-start
        write(root/'model-manifest.json',dict(status='fresh_full64_160_iterations_saved_before_dev_inference',plan=ref(plan_path),
              model=ref(model_path),configuration=ref(configuration.__file__),features=vectorizer.feature_names_,
              old_vectorizer=oldfit['model'],vectorizer_refit=False,training_examples=62281,training_positive=4196,
              training_matrix_shape=list(matrix.shape),training_matrix_dtype=str(matrix.dtype),training_matrix_sha256=matrix_sha,
              historical_matrix_byte_comparison=False,actual_iterations=model.n_iter_,training_ids=oldfit['training_ids'],
              estimator=model.get_params(),seconds=fit_seconds))
        del matrix,xs,ys,train
        dev=rows(checked(p['dev_pool']));inputs=rows(checked(p['dev_input']));byid={x['pmc_id']:x for x in inputs}
        assert len(dev)==len(byid)==len(inputs)==16 and not set(byid)&set(truth)
        assert [x['pmc_id'] for x in dev]==dev_manifest['input_ids']
        assert all(set(x)=={'pmc_id','pmid','patient','full_text'} for x in inputs)
        scores=[];total=0
        for row in dev:
            assert row['input']==byid[row['pmc_id']]
            cs=row['candidates'];values=vectorizer.transform([feature(multi,row,c) for c in cs])
            probabilities=model.predict_proba(values)
            assert values.shape==(len(cs),103) and probabilities.shape==(len(cs),2) and single.np.isfinite(probabilities).all()
            assert single.np.array_equal(probabilities,reloaded['model'].predict_proba(reloaded['vectorizer'].transform([feature(multi,row,c) for c in cs])))
            assert single.np.array_equal(probabilities[:,1],model.booster_.predict(values,num_threads=1))
            assert single.np.allclose(probabilities.sum(axis=1),1.,rtol=0,atol=1e-15)
            scores.append(dict(pmc_id=row['pmc_id'],candidates=[dict(c,selector_probability=float(prob)) for c,prob in zip(cs,probabilities[:,1])]))
            total+=len(cs)
        assert total==dev_manifest['new_rows']==13246
        score_path=root/'selector-scores.jsonl'
        score_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in scores))
        fit_predict_seconds=time.monotonic()-start;assert fit_predict_seconds<120
        write(root/'inference-manifest.json',dict(status='complete16_fixed_pool_probabilities',model=ref(model_path),
              input_pool=p['dev_pool'],rows=total,documents=16,features=vectorizer.feature_names_,scores=ref(score_path),
              pickle_reload_all_probabilities_exact=True,native_positive_column_exact=True,classes=[0,1],
              fit_load_save_predict_seconds=fit_predict_seconds,dev_gold_read=False,actual_classes=model.classes_.tolist()))
        # The reused ontology feature modules already hold their own run object.
        # Bind sequence's distinct run.py explicitly for the decoder imports.
        ontology_run=single.baseline
        sequence_path=Path(p['module_directories']['nested'])/'run.py'
        spec=importlib.util.spec_from_file_location('run',sequence_path)
        sequence_run=importlib.util.module_from_spec(spec);sys.modules['run']=sequence_run;spec.loader.exec_module(sequence_run)
        sys.path.insert(0,p['module_directories']['nested']);import decode_nested_selector as decoder
        assert decoder.prior.base is sequence_run and single.baseline is ontology_run
        sys.path.insert(0,p['module_directories']['literal']);import control as literal
        _,allowed,_,_=decoder.prior.base.parse_hpo(checked(p['hpo']));allowed.discard('HP:0000118')
        original=checked(p['baseline_prediction']).read_bytes().decode('utf-8').splitlines(keepends=True)
        baseline=[json.loads(x) for x in original]
        old_entity_docs=rows(checked(p['lineage']['strong_entity_prediction']))
        old_entities={x['pmc_id']:x['entities'] for x in old_entity_docs}
        assert len(baseline)==16 and set(x['pmc_id'] for x in baseline)==set(byid)
        assert sum(len(x['entities']) for x in baseline)==1072
        assert sum(len(a['phenotype']) for x in baseline for a in x['association'])==391
        scorebyid={x['pmc_id']:x for x in scores};output=[];diagnostics=[]
        for line,before in zip(original,baseline):
            pmc=before['pmc_id'];assert before['entities']==old_entities[pmc]
            cs=scorebyid[pmc]['candidates'];selected,diag=decoder.decode(cs,decoder.greedy(cs,.3),byid[pmc],allowed,.3)
            output.append(literal.replace_entities(line,selected))
            oldkeys={decoder.key(x) for x in before['entities']};newkeys={decoder.key(x) for x in selected}
            diagnostics.append(dict(pmc_id=pmc,changed=before['entities']!=selected,entity_added=len(newkeys-oldkeys),entity_removed=len(oldkeys-newkeys),**diag))
        pred_path=root/'lightgbm.a179-entities-only.jsonl';pred_path.write_bytes(''.join(output).encode('utf-8'))
        write(root/'input-only-diagnostics.json',diagnostics)
        write(root/'complete-freeze.json',dict(frozen_at_utc=now(),plan=ref(plan_path),model=ref(model_path),scores=ref(score_path),
              baseline=p['baseline_prediction'],prediction=ref(pred_path),diagnostics=ref(root/'input-only-diagnostics.json'),
              all16_complete=True,nonentity_literal_bytes_preserved=True,old391_relations_preserved=True,no_support_recompute=True,
              changed_documents=sum(x['changed'] for x in diagnostics),entity_added=sum(x['entity_added'] for x in diagnostics),
              entity_removed=sum(x['entity_removed'] for x in diagnostics),dev_gold_read=False,seconds=time.monotonic()-start))
        checked(read(root/'complete-freeze.json')['prediction'])
        write(root/'evaluation-start.json',dict(started_utc=now(),freeze=ref(root/'complete-freeze.json')))
        sys.path.insert(0,p['module_directories']['evaluation']);import evaluate;import validate_predictions as validator
        assert evaluate.VERSION=='patientphex-public-approx-v2'
        gold=rows(checked(p['dev_gold']));goldbyid={x['pmc_id']:x for x in gold}
        assert len(goldbyid)==len(gold)==16 and set(goldbyid)==set(byid)
        assert all(prior.strip_labels(x)==byid[x['pmc_id']] for x in gold)
        predictions=rows(pred_path);report=evaluate.evaluate_report(gold,predictions,'all','text')
        validity=validator.validate_predictions(gold,predictions,validator.read_hpo(checked(p['hpo'])))
        assert validity['valid'] and validity['error_count']==validity['warning_count']==0
        old_metrics=read(checked(p['baseline_metrics']))
        assert report['counts']['relation']==old_metrics['counts']['relation']
        assert report['metrics']['micF1']==old_metrics['metrics']['micF1'] and report['metrics']['macF1']==old_metrics['metrics']['macF1']
        write(root/'metrics.json',report);write(root/'validation.json',validity)
        delta={k:report['metrics'][k]-old_metrics['metrics'][k] for k in old_metrics['metrics']}
        result=dict(status='complete_unique_originaldev_control',finished_utc=now(),plan=ref(plan_path),source=ref(__file__),
              baseline_metrics=old_metrics['metrics'],lightgbm_metrics=report['metrics'],deltas=delta,
              keep_as_candidate_best=delta['Score']>0,meets_0_8_score=report['metrics']['Score']>=.8,
              changed_documents=sum(x['changed'] for x in diagnostics),entity_added=sum(x['entity_added'] for x in diagnostics),
              entity_removed=sum(x['entity_removed'] for x in diagnostics),model=ref(model_path),prediction=ref(pred_path),
              freeze=ref(root/'complete-freeze.json'),full64_fit_calls=1,new_dev_v2_calls=1,old_dev_v2_calls=0,
              relations391_literal_preserved=True,all_nonentity_literal_bytes_preserved=True,
              no_sequence_baseline_modification=True,no_official_actions=True,originaldev_is_development_not_blind=True,
              fit_load_save_predict_seconds=fit_predict_seconds,total_seconds=time.monotonic()-start)
        assert result['total_seconds']<120
        write(root/'result.json',result)
        state.update(status='complete',finished_utc=now(),result=ref(root/'result.json'))
        print(json.dumps({k:result[k] for k in ('lightgbm_metrics','deltas','keep_as_candidate_best','changed_documents','entity_added','entity_removed','fit_load_save_predict_seconds','total_seconds')}))
    except BaseException as exc:
        state.update(status='failed_preserved',finished_utc=now(),failure=repr(exc));raise
    finally:
        (root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n');signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    execute(Path(parser.parse_args().plan).resolve())
