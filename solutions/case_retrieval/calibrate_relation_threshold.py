#!/usr/bin/env python3
"""Strict train16->train32->meta16 calibration, without LLM features.

Features, score freezing, threshold selection and outer application are separate
phases. Only the selection phase accepts meta-validation labels. The outer
application freezes predictions but never reads outer or original-dev gold.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed(p):
    rows = read(p); result = {str(d['pmc_id']): d for d in rows}
    assert len(rows) == len(result), 'Duplicate document'
    return result


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_rows(p, rows):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(row):
    return str(row['pmc_id']), str(row['patient_id']), row['concept']


def allowed(entities, inputs):
    rows = []
    for doc in inputs:
        assert 'entities' not in doc and 'association' not in doc
        did = str(doc['pmc_id'])
        concepts = sorted({value for e in entities[did]['entities'] for value in
                           ([e['text']] if e['identifier'] == '-1' else e['identifier'].split(';'))})
        for pid in [str(p['patient_id']) for p in doc['patient']]:
            rows.extend({'pmc_id': did, 'patient_id': pid, 'concept': c} for c in concepts)
    return rows


def predictions(inputs, entities, rows, probabilities, threshold):
    assert len(rows) == len(probabilities)
    by = {str(d['pmc_id']): {str(p['patient_id']): set() for p in d['patient']} for d in inputs}
    for row, p in zip(rows, probabilities):
        if p >= threshold:
            by[str(row['pmc_id'])][str(row['patient_id'])].add(row['concept'])
    output = []
    for d in inputs:
        did = str(d['pmc_id']); item = copy.deepcopy(entities[did])
        item['association'] = [{'patient_id': p['patient_id'], 'phenotype': sorted(by[did][str(p['patient_id'])])}
                               for p in d['patient']]
        assert item['entities'] == entities[did]['entities']
        output.append(item)
    return output


def choose_threshold(grid_results):
    # Integer grid indices avoid floating-point .45/.55 distance tie accidents.
    return max(grid_results, key=lambda r: (r['metrics']['Score'], -abs(r['step'] - 10), r['step']))


def validate_plan(plan):
    assert plan['grid_steps'] == list(range(2, 19))
    assert plan['grid_denominator'] == 20 and plan['meta_validation_fold'] == 1
    assert [(p['generator_fold'], p['target_fold']) for p in plan['pools']] == [(2, 3), (3, 2)]
    for role, item in plan['pins'].items():
        assert sha(item['path']) == item['sha256'], role


def features(plan, out):
    assert not (out/'features-complete.json').exists()
    start = time.monotonic()
    sys.path.insert(0, plan['ontology_source_dir'])
    agg = load('frozen_relation_aggregate', plan['aggregate_source'])
    sequence = agg.load_sequence(plan['sequence_source'])
    meta_ids = set(keyed(plan['meta_input']))
    outer_ids = set(keyed(plan['outer_input']))
    training_ids = set().union(*(set(keyed(p['generator_train'])) for p in plan['pools']))
    assert len(training_ids) == 32 and len(meta_ids) == 16 and len(outer_ids) == 16
    assert not (training_ids & meta_ids or training_ids & outer_ids or meta_ids & outer_ids)
    train_rows, seen, origins, run_times = [], set(), [], {}
    for spec in plan['pools']:
        g, j = spec['generator_fold'], spec['target_fold']
        destination = out/f'pool{g}-to{j}'; destination.mkdir()
        train = keyed(spec['generator_train']); inp = keyed(spec['input'])
        assert len(train) == len(inp) == 16 and not set(train)&set(inp)
        assert set(train)|set(inp) == training_ids and not seen&set(inp)
        seen |= set(inp)
        source_manifest = json.loads(Path(spec['candidate_manifest']).read_text())
        assert source_manifest['inputs']['train'] == sha(spec['generator_train'])
        assert source_manifest['inputs']['input'] == sha(spec['input'])
        assert source_manifest['output']['sha256'] == sha(spec['candidates'])
        assert source_manifest['parent_actual_manifest_sha256'] == sha(spec['parent_manifest'])
        assert source_manifest['parent_actual_pool_sha256'] == sha(spec['parent_pool'])
        assert all('entities' not in d and 'association' not in d for d in inp.values())
        crf = destination/'crf.jsonl'; work = destination/'crf-work'
        command = [plan['crf_python'], plan['sequence_source'], '--train', spec['generator_train'],
                   '--input', spec['input'], '--hpo', plan['hpo'], '--output', str(crf), '--work-dir', str(work)]
        t = time.monotonic()
        with (destination/'crf.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        run_times[f'crf{g}to{j}'] = time.monotonic() - t
        crf_manifest = work/'crf-manifest.json'
        crfs = agg.multi.checked_crf(str(crf), str(crf_manifest), spec['generator_train'], spec['input'], training_ids)
        candidates = destination/'candidates.jsonl'
        augment = [plan['crf_python'], plan['augment_source'], '--train', spec['generator_train'],
                   '--input', spec['input'], '--hpo', plan['hpo'], '--base', str(crf),
                   '--candidates', spec['candidates'], '--candidate-manifest', spec['candidate_manifest'],
                   '--output', str(candidates), '--relation-mode', 'rules', '--probability', '.65', '--similarity', '.92']
        subprocess.run(augment, capture_output=True, text=True, check=True)
        bases = agg.multi.keyed(spec['base']); raw = agg.multi.single.read_candidates(spec['candidates'])
        es = keyed(candidates); wanted = {identity(r) for r in allowed(es, list(inp.values()))}
        # Only the other training group supplies labels; meta F1 is not read.
        gold = keyed(spec['gold']); assert set(gold) == set(inp)
        for did, doc in inp.items():
            rows, _, _ = agg.document_rows(doc, bases[did], raw[did], crfs[did], sequence)
            truth = agg.gold_links(gold[did])
            for r in rows:
                r['pmc_id'] = str(r['pmc_id']); r['patient_id'] = str(r['patient_id'])
                if identity(r) in wanted:
                    r['target'] = (r['patient_id'], r['concept']) in truth
                    train_rows.append(r)
        assert {identity(r) for r in train_rows if r['pmc_id'] in inp} == wanted
        origins.append({'generator_fold': g, 'target_fold': j, 'command': command, 'augment_command': augment,
                        'crf_manifest_sha256': sha(crf_manifest), 'candidates_sha256': sha(candidates),
                        'semantic_manifest_sha256': sha(spec['candidate_manifest']),
                        'parent_manifest_sha256': sha(spec['parent_manifest']), 'rows': len(wanted)})
    assert seen == training_ids
    # Recompute meta features from stripped input and its train32 predictions.
    # Do not open the old aggregate training rows containing meta labels.
    m = plan['meta_sources']; docs = keyed(plan['meta_input']); bases = agg.multi.keyed(m['base'])
    raw = agg.multi.single.read_candidates(m['candidates'])
    crfs = agg.multi.checked_crf(m['crf'], m['crf_manifest'], m['generator_train'], plan['meta_input'], training_ids)
    assert set(keyed(m['generator_train'])) == training_ids
    mm = json.loads(Path(m['candidate_manifest']).read_text())
    assert mm['inputs']['train'] == sha(m['generator_train']) and mm['inputs']['input'] == sha(plan['meta_input'])
    assert mm['output']['sha256'] == sha(m['candidates'])
    entities = keyed(plan['meta_entities']); wanted = {identity(r) for r in allowed(entities, list(docs.values()))}
    meta_rows = []
    for did, doc in docs.items():
        rows, _, _ = agg.document_rows(doc, bases[did], raw[did], crfs[did], sequence)
        meta_rows += [dict(r, pmc_id=str(r['pmc_id']), patient_id=str(r['patient_id'])) for r in rows if identity(r) in wanted]
    assert len(meta_rows) == len(wanted) == 1557 and all('target' not in r for r in meta_rows)
    train_rows.sort(key=identity); meta_rows.sort(key=identity)
    write_rows(out/'train-rows.jsonl', train_rows); write_rows(out/'meta-rows.jsonl', meta_rows)
    save(out/'features-complete.json', {'source_sha256': sha(__file__), 'origins': origins,
         'train_rows': len(train_rows), 'positive': sum(r['target'] for r in train_rows), 'meta_rows': len(meta_rows),
         'train_rows_sha256': sha(out/'train-rows.jsonl'), 'meta_rows_sha256': sha(out/'meta-rows.jsonl'),
         'meta_labels_read': False, 'outer_gold_read': False, 'seconds': time.monotonic()-start, 'CRF_seconds': run_times})
    print(json.dumps({'features_complete': True, 'train_rows': len(train_rows), 'meta_rows': len(meta_rows)}), flush=True)


def fit_meta(plan, out):
    import numpy as np
    import joblib
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    f = json.loads((out/'features-complete.json').read_text())
    assert f['train_rows_sha256'] == sha(out/'train-rows.jsonl') and f['meta_rows_sha256'] == sha(out/'meta-rows.jsonl')
    train, meta = read(out/'train-rows.jsonl'), read(out/'meta-rows.jsonl')
    assert not {r['pmc_id'] for r in train}&{r['pmc_id'] for r in meta} and all('target' not in r for r in meta)
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform([r['features'] for r in train]); z = vectorizer.transform([r['features'] for r in meta])
    y = np.asarray([r['target'] for r in train], dtype=np.int64)
    with threadpool_limits(limits=1):
        model = HistGradientBoostingClassifier(**plan['estimator']).fit(x, y)
        scores = model.predict_proba(z)[:, 1]
    assert np.isfinite(scores).all()
    if (out/'meta-scores.jsonl').exists(): raise ValueError('Refusing to overwrite scores')
    write_rows(out/'meta-scores.jsonl', [{'pmc_id': r['pmc_id'], 'patient_id': r['patient_id'], 'concept': r['concept'], 'probability': float(p)} for r,p in zip(meta,scores)])
    joblib.dump({'model': model, 'vectorizer': vectorizer}, out/'meta-model.joblib')
    save(out/'meta-scores-freeze.json', {'source_sha256': sha(__file__), 'plan_sha256': sha(plan['_path']),
         'scores_sha256': sha(out/'meta-scores.jsonl'), 'model_sha256': sha(out/'meta-model.joblib'),
         'train_matrix_sha256': hashlib.sha256(x.tobytes()).hexdigest(), 'columns': vectorizer.feature_names_,
         'meta_labels_read': False, 'outer_gold_read': False, 'parameter_sweep': False})
    print(json.dumps({'meta_scores_frozen': True, 'features': x.shape[1], 'train_rows': x.shape[0]}), flush=True)


def select(plan, out):
    assert not (out/'threshold-selection.json').exists()
    f = json.loads((out/'meta-scores-freeze.json').read_text())
    assert f['scores_sha256'] == sha(out/'meta-scores.jsonl') and not f['meta_labels_read']
    rows = read(out/'meta-scores.jsonl'); scores = [r['probability'] for r in rows]
    inputs = read(plan['meta_input']); entities = keyed(plan['meta_entities'])
    sys.path.insert(0, str(Path(plan['evaluator']).parent))
    evaluator = load('public_evaluator_meta_calibration', plan['evaluator'])
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    gold = read(plan['meta_gold'])  # First and only meta label read occurs after score freeze.
    grid = []
    for step in plan['grid_steps']:
        threshold = step/plan['grid_denominator']
        pred = predictions(inputs, entities, rows, scores, threshold)
        result = evaluator.evaluate(gold, pred)
        grid.append({'step': step, 'threshold': threshold, 'metrics': result['metrics'], 'counts': result['counts']})
    selected = choose_threshold(grid)
    save(out/'threshold-selection.json', {'source_sha256': sha(__file__), 'score_freeze_sha256': sha(out/'meta-scores-freeze.json'),
         'meta_gold_sha256_after_freeze': sha(plan['meta_gold']), 'grid': grid, 'selected': selected,
         'tie_policy': 'Higher exact Score, then integer-grid distance to0.5 nearer, then larger threshold.',
         'outer_gold_read': False, 'originaldev_gold_read': False})
    print(json.dumps({'selected_threshold': selected['threshold'], 'meta_metrics': selected['metrics']}), flush=True)


def apply_outer(plan, out):
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    old = load('old_fixed_llm_feature_tree', plan['old_tree_source'])
    selected = json.loads((out/'threshold-selection.json').read_text())['selected']['threshold']
    assert selected in [step/20 for step in plan['grid_steps']]
    assert sha(plan['old_models']) == plan['old_models_sha256']
    bundle = joblib.load(plan['old_models'])  # Own earlier frozen trusted artifact only.
    prompts = old.read_rows(plan['outer_prompts']); pairs = old.pair_rows(prompts)
    raw = old.read_rows(plan['outer_features']); features = old.join_features(pairs, raw, False)
    z = bundle['vectorizer'].transform(features)
    with threadpool_limits(limits=1): scores = bundle['models']['without_llm'].predict_proba(z)[:,1]
    assert np.isfinite(scores).all()
    inputs = read(plan['outer_input']); entities = keyed(plan['outer_entities'])
    baseline = predictions(inputs, entities, pairs, scores, .5)
    baseline_bytes = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in baseline).encode()
    assert hashlib.sha256(baseline_bytes).hexdigest() == sha(plan['old_baseline_prediction'])
    chosen = predictions(inputs, entities, pairs, scores, selected)
    path = out/'outer-calibrated.predictions.jsonl'; assert not path.exists()
    write_rows(path, chosen)
    save(out/'outer-predictions-freeze.json', {'source_sha256': sha(__file__), 'selection_sha256': sha(out/'threshold-selection.json'),
         'selected_threshold': selected, 'old0.5_prediction_byteexact': True, 'old0.5_sha256': sha(plan['old_baseline_prediction']),
         'output_sha256': sha(path), 'old_model_sha256': sha(plan['old_models']), 'entities_unchanged': True,
         'outer_gold_read': False, 'originaldev_gold_read': False, 'evaluation_performed': False})
    print(json.dumps({'outer_prediction_frozen': True, 'threshold': selected, 'sha256': sha(path)}), flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['features','fit','select','apply']);p.add_argument('--plan',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());validate_plan(plan);plan['_path']=a.plan
    out=Path(plan['output_dir']);out.mkdir(parents=True,exist_ok=True)
    {'features':features,'fit':fit_meta,'select':select,'apply':apply_outer}[a.phase](plan,out)


if __name__=='__main__': main()
