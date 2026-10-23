#!/usr/bin/env python3
"""Fixed raw top1 original-gate prehybrid supply on strict original Bioformer pools."""
import argparse
import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

RAW_FEATURES = [f'raw_{model}_{field}' for model in ('pubmed', 'bioformer') for field in ('observed', 'p')]
FROZEN_SELECTOR_SHA = '7cf106a6503cb515dd389b0664cee2bbf2591a61bbbe686d53ab1dccae73cb25'
THRESHOLD = .3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_ref(item):
    path = Path(item['path'])
    assert path.is_absolute(), path
    assert sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes'], str(path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def keyed(path, compressed=False):
    opener = gzip.open if compressed else open
    result = {}
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            did = row['pmc_id']
            assert did not in result, ('duplicate_document', did)
            result[did] = row
    return result


def key(candidate):
    return candidate['offset'], candidate['length'], candidate['identifier']


def strip_labels(doc):
    return {k: v for k, v in doc.items() if k not in ('entities', 'association')}


def nonentities(doc):
    return {k: v for k, v in doc.items() if k != 'entities'}


def labels(doc, candidates):
    """Literal old labels, including composite subset and gold '-1' wildcard."""
    truth = {}
    for entity in doc['entities']:
        if str(entity.get('note', '')).upper() != 'NO':
            truth.setdefault((entity['offset'], entity['length']), set()).update(
                entity['identifier'].split(';'))
    return [bool((ids := truth.get(key(c)[:2], set())) and
                 ('-1' in ids or set(c['identifier'].split(';')) <= ids))
            for c in candidates]


def select(candidates, probabilities):
    """The frozen stable probability/length/offset greedy decoder, at .3."""
    assert len(candidates) == len(probabilities)
    selected = []
    for candidate, probability in sorted(
            zip(candidates, probabilities),
            key=lambda item: (-item[1], -item[0]['length'], item[0]['offset'])):
        assert math.isfinite(float(probability)) and 0 <= probability <= 1
        if probability < THRESHOLD or any(
                candidate['offset'] < entity['offset'] + entity['length'] and
                entity['offset'] < candidate['offset'] + candidate['length']
                for entity in selected):
            continue
        selected.append({k: candidate[k] for k in
                         ('identifier', 'offset', 'length', 'text')} |
                        {'type': 'Phenotype', 'note': None})
    return sorted(selected, key=lambda entity: entity['offset'])


def check_pool(row, doc, base, old, origin_index, old_features):
    """Old objects stay exact; appended raw-source candidates may add spans."""
    assert set(row) == {'pmc_id', 'input', 'base', 'candidates', 'origin_index'}
    assert row['pmc_id'] == doc['pmc_id'] and row['origin_index'] == origin_index
    assert row['input'] == strip_labels(doc) and row['base'] == base
    assert set(row['input']) == {'pmc_id', 'pmid', 'full_text', 'patient'}
    candidates = row['candidates']
    assert candidates[:len(old)] == old
    for previous, current in zip(old, candidates):
        assert set(current) == set(previous) and not set(RAW_FEATURES) & set(previous)
        assert all(type(current[k]) is type(v) for k, v in previous.items())
    assert len({key(c) for c in candidates}) == len(candidates)
    new = candidates[len(old):]
    assert [key(c) for c in new] == sorted(key(c) for c in new)
    crf_spans = {key(c)[:2] for c in old if c['crf_span']}
    for candidate in new:
        span = key(candidate)[:2]
        passages = [p for p in doc['full_text'] if p['offset'] <= span[0] and span[0]+span[1] <= p['offset']+len(p['text'])]
        assert len(passages) == 1
        passage = passages[0];start = span[0]-passage['offset']
        assert passage['text'][start:start+span[1]] == candidate['text']
        assert candidate['raw_pubmed_observed'] or candidate['raw_bioformer_observed']
        for model in ('pubmed', 'bioformer'):
            observed, probability = candidate[f'raw_{model}_observed'], candidate[f'raw_{model}_p']
            assert type(observed) is bool and type(probability) is float and math.isfinite(probability)
            assert .95 < probability <= 1 if observed else probability == 0.
        identifier = candidate['identifier']
        assert len(identifier) == 10 and identifier.startswith('HP:') and identifier[3:].isdigit()
        expected = {k:candidate[k] for k in ('offset','length','identifier','text')}
        expected.update(probability=.5, similarity=0., exact=False, is_base=False, crf=False,
                        crf_span=span in crf_spans, semantic_source=False, **{name:False for name in old_features})
        assert {k:v for k,v in candidate.items() if k not in RAW_FEATURES} == expected
    return candidates


def added_raw_features(candidate):
    # Missing old-row columns mean "not an appended raw source", never p=0
    # as a statement about the public model's unobserved posterior.
    return {name:candidate.get(name, False if name.endswith('_observed') else 0.) for name in RAW_FEATURES}


def checked_sources(plan):
    assert str(Path(__file__).resolve()) in plan['sources']
    for path, digest in plan['sources'].items():
        assert Path(path).is_absolute() and sha(path) == digest, path


def load_modules(plan, old_fit, reference):
    paths = [p for p, digest in plan['sources'].items()
             if Path(p).name == 'select_public_bioformer.py' and digest == FROZEN_SELECTOR_SHA]
    assert len(paths) == 1 and old_fit['source_sha256'] == FROZEN_SELECTOR_SHA
    sys.path.insert(0, str(Path(paths[0]).parent))
    sys.path.insert(0, plan['ontology_source'])
    module_spec = importlib.util.spec_from_file_location('frozen_top3_selector', paths[0])
    legacy = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(legacy)
    assert sha(legacy.legacy.__file__) == old_fit['legacy_source_sha256']
    multi = importlib.import_module('calibrate_multi')
    single = multi.single
    modules = [multi, single, single.baseline, single.enhance, single.merge_semantic]
    for module in modules:
        assert sha(module.__file__) == old_fit['dependencies'][module.__name__]
    assert sha(multi.__file__) == reference['source_sha256']
    assert all(sha(module.__file__) == reference['dependencies'][module.__name__]
               for module in modules[1:])
    return legacy, multi, modules


def public_sources(old_fit, multi, legacy, docs):
    arguments = old_fit['arguments']
    result = []
    for role in ('public', 'bioformer'):
        assert sha(arguments[role]) == old_fit[role + '_sha256']
        assert sha(arguments[role + '_manifest']) == old_fit[role + '_manifest_sha256']
        manifest = read_json(arguments[role + '_manifest'])
        assert manifest['output_sha256'] == old_fit[role + '_sha256']
        assert manifest['public_model_task_fit'] is False
        expected_weight = ('d58af55a68840a24b052a775cf0cb32b8993a1fa4823c76aad10d3e6299d7ec3'
                           if role == 'public' else legacy.BIOFORMER_WEIGHT)
        assert manifest['model_weight_sha256'] == expected_weight
        assert sha(manifest['input_path']) == manifest['input_sha256']
        inputs = keyed(manifest['input_path'])
        assert inputs.keys() == docs.keys() and len(inputs) == 64
        for did, doc in docs.items():
            assert inputs[did] == strip_labels(doc)
            assert not {'entities', 'association'} & inputs[did].keys()
        result.append(legacy.public_rows(arguments[role], inputs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--fold', required=True, type=int, choices=range(4))
    args = parser.parse_args()
    started = time.monotonic()
    plan = read_json(args.plan)
    plan_sha = sha(args.plan)
    checked_sources(plan)
    assert [item['fold'] for item in plan['folds']] == [0, 1, 2, 3]
    fold = plan['folds'][args.fold]
    target = Path(plan['output_dir']) / f'fold{args.fold}' / 'fitted'
    target.mkdir(parents=True, exist_ok=True)
    assert not any((target / name).exists() for name in
                   ('manifest.json', 'augmented.jsonl', 'selector-scores.jsonl'))
    prepared_manifest_path = checked_ref(plan['prepared_manifest'])
    prepared_manifest = read_json(prepared_manifest_path)
    assert [item['fold'] for item in prepared_manifest['folds']] == [0, 1, 2, 3]
    prepared = prepared_manifest['folds'][args.fold]
    prepared_train = keyed(checked_ref(prepared['train']), compressed=True)
    prepared_input = keyed(checked_ref(prepared['input']), compressed=True)
    old_fit = read_json(checked_ref(fold['old_fit_manifest']))
    reference = read_json(checked_ref(fold['reference']))
    assert old_fit['old_reference_sha256'] == fold['reference']['sha256']
    legacy, multi, modules = load_modules(plan, old_fit, reference)
    single = multi.single
    old_features = legacy.PUBLIC_FEATURES + legacy.BIOFORMER_FEATURES
    assert len(old_features) == 12 and not set(old_features) & set(RAW_FEATURES)
    config = reference['config']
    assert config['threshold'] == .4 and config['without_crf'] is False and config['crf_confidence'] is None
    for name, digest in reference['inputs'].items():
        assert sha(config[name]) == digest, name
    assert sha(config['output']) == reference['output_sha256']
    assert checked_ref(fold['train_gold']).resolve() == Path(config['train']).resolve()
    train, inputs = keyed(config['train']), keyed(config['input'])
    assert len(train) == 48 and len(inputs) == 16 and not train.keys() & inputs.keys()
    assert all(not {'entities', 'association'} & doc.keys() for doc in inputs.values())
    assert prepared_train.keys() == train.keys() and prepared_input.keys() == inputs.keys()
    public, bioformer = public_sources(old_fit, multi, legacy, dict(train, **inputs))
    specs = read_json(config['training_manifest'])
    assert len(specs) == len(old_fit['training_origins']) == 3
    xs, ys, seen, origins, ordered_train = [], [], set(), [], []
    old_count = original_count = old_positive = 0
    for index, spec in enumerate(specs):
        origin = {name: sha(path) for name, path in spec.items()}
        assert origin == old_fit['training_origins'][index]
        origins.append(origin)
        gold, inner_input = keyed(spec['gold']), keyed(spec['input'])
        generators, bases = keyed(spec['generator_train']), keyed(spec['base'])
        assert len(generators) == 32 and len(gold) == len(inner_input) == 16
        assert generators.keys() == train.keys() - gold.keys()
        assert not gold.keys() & seen and gold.keys() <= train.keys()
        assert gold.keys() == inner_input.keys() == bases.keys()
        for did, doc in generators.items():
            assert doc == train[did]
        for did, doc in gold.items():
            assert doc == train[did] and inner_input[did] == strip_labels(doc)
        seen.update(gold)
        cm = read_json(spec['candidate_manifest'])
        assert cm['inputs']['input'] == sha(spec['input'])
        assert cm['inputs']['train'] == sha(spec['generator_train'])
        assert cm['output']['sha256'] == sha(spec['candidates'])
        raw = single.read_candidates(spec['candidates'])
        assert raw.keys() == gold.keys()
        crfs = multi.checked_crf(spec['crf'], spec['crf_manifest'],
                                 spec['generator_train'], spec['input'], set(train))
        for did, doc in gold.items():
            old_ontology = multi.pool(doc, bases[did], raw[did], crfs[did])
            old = legacy.merge_bioformer(legacy.merge_pool(old_ontology, public[did]), bioformer[did])
            candidates = check_pool(prepared_train[did], doc, bases[did], old, index, old_features)
            original_count += len(old_ontology)
            old_count += len(old)
            old_positive += sum(labels(doc, old))
            ys.extend(labels(doc, candidates))
            xs.extend(multi.features(doc, c, bases[did]) |
                      {name: c[name] for name in old_features} | added_raw_features(c)
                      for c in candidates)
            ordered_train.append(did)
    assert seen == train.keys() and list(prepared_train) == ordered_train
    assert original_count == reference['training_examples']
    assert old_count == old_fit['training_examples'] == prepared['old_train_rows']
    assert old_positive == old_fit['training_positive']
    assert plan['maximum_training_row_ratio'] == 1.04
    assert len(ys) == prepared['new_train_rows'] and old_count <= len(ys) <= plan['maximum_training_row_ratio'] * old_count
    vectorizer = single.DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(xs)
    assert vectorizer.feature_names_ == sorted(old_fit['features'] + RAW_FEATURES)
    model = single.HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=5., learning_rate=.06,
        random_state=20261023, early_stopping=False)
    assert model.get_params() == old_fit['estimator'] == reference['estimator']
    baseline = keyed(checked_ref(fold['baseline_prediction']))
    old_output_dir = Path(old_fit['arguments']['output_dir'])
    for name in ('augmented.jsonl', 'selector-scores.jsonl'):
        assert sha(old_output_dir / name) == old_fit['outputs'][name]
    old_prediction = keyed(old_output_dir / 'augmented.jsonl')
    old_scores = keyed(old_output_dir / 'selector-scores.jsonl')
    assert list(inputs) == list(prepared_input) == list(baseline) == list(old_prediction) == list(old_scores)
    bases, raw = keyed(config['base']), single.read_candidates(config['candidates'])
    assert bases.keys() == raw.keys() == inputs.keys()
    crfs = multi.checked_crf(config['crf'], config['crf_manifest'], config['train'], config['input'], set(train))
    input_pools = []
    old_input_count = original_input_count = input_count = 0
    for did, doc in inputs.items():
        old_ontology = multi.pool(doc, bases[did], raw[did], crfs[did])
        old = legacy.merge_bioformer(legacy.merge_pool(old_ontology, public[did]), bioformer[did])
        candidates = check_pool(prepared_input[did], doc, bases[did], old, None, old_features)
        saved = old_scores[did]['candidates']
        assert [{k: v for k, v in c.items() if k != 'selector_probability'} for c in saved] == old
        assert baseline[did]['entities'] == select(saved, [c['selector_probability'] for c in saved])
        assert nonentities(baseline[did]) == nonentities(old_prediction[did])
        original_input_count += len(old_ontology)
        old_input_count += len(old)
        input_count += len(candidates)
        input_pools.append((did, doc, candidates))
    assert original_input_count == old_fit['old_input_candidates']
    assert old_input_count == (old_fit['old_input_candidates'] +
        old_fit['added_public_input_candidates'] + old_fit['added_bioformer_input_candidates'])
    assert old_input_count == prepared['old_input_rows'] and input_count == prepared['new_input_rows']
    checked_sources(plan)
    assert sha(args.plan) == plan_sha
    model.fit(matrix, ys)
    import pickle
    with (target / 'model-vectorizer.pkl').open('xb') as stream:
        pickle.dump({'model':model,'vectorizer':vectorizer,'features':vectorizer.feature_names_},stream,protocol=5)
    assert model.n_iter_ == 160 and not model.do_early_stopping_
    prediction, scores = [], []
    for did, doc, candidates in input_pools:
        probabilities = (model.predict_proba(vectorizer.transform([
            multi.features(doc, c, bases[did]) |
            {name: c[name] for name in old_features} | added_raw_features(c)
            for c in candidates]))[:, 1] if candidates else [])
        predicted = copy.deepcopy(baseline[did])
        predicted['entities'] = select(candidates, probabilities)
        assert nonentities(predicted) == nonentities(baseline[did])
        prediction.append(predicted)
        scores.append({'pmc_id': did, 'candidates': [dict(c, selector_probability=float(p))
                       for c, p in zip(candidates, probabilities)]})
    checked_sources(plan)
    assert sha(args.plan) == plan_sha
    for filename, rows in (('augmented.jsonl', prediction), ('selector-scores.jsonl', scores)):
        with (target / filename).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {
        'source_sha256': sha(__file__), 'plan_sha256': plan_sha, 'fold': args.fold,
        'arguments': vars(args), 'prepared_manifest_sha256': sha(prepared_manifest_path),
        'prepared_train': prepared['train'], 'prepared_input': prepared['input'],
        'old_fit_manifest': fold['old_fit_manifest'], 'reference': fold['reference'],
        'baseline_prediction': fold['baseline_prediction'], 'training_origins': origins,
        'dependencies': {module.__name__: sha(module.__file__) for module in modules},
        'legacy_source_sha256': sha(legacy.__file__),
        'legacy_public_source_sha256': sha(legacy.legacy.__file__),
        'actual_sources': plan['sources'], 'training_documents': len(train),
        'input_documents': len(inputs), 'training_document_ids_in_order': ordered_train,
        'input_document_ids_in_order': list(inputs), 'old_training_examples': old_count,
        'training_examples': len(ys), 'added_training_examples': len(ys) - old_count,
        'old_training_positive': old_positive, 'training_positive': sum(ys),
        'old_input_candidates': old_input_count, 'input_candidates': input_count,
        'added_input_candidates': input_count - old_input_count,
        'features': vectorizer.feature_names_, 'new_features': RAW_FEATURES,
        'estimator': model.get_params(), 'actual_iterations': int(model.n_iter_),
        'early_stopping_performed': bool(model.do_early_stopping_), 'threshold': THRESHOLD,
        'runtime': {'python': platform.python_version(), 'sklearn': single.sklearn.__version__,
                    'numpy': single.np.__version__},
        'assertions': {'old_keys_and_order_preserved': True, 'old_candidate_fields_equal': True, 'old_pool_fields_keys_order_exact': True,
                       'old_labels_equal': True, 'old_feature_columns_preserved': True,
                       'new_raw_keys_can_add_spans': True, 'training_rows_at_most_revised_limit': True,
                       'strict_three_train32_to_input16': True, 'baseline_cut03_replayed': True,
                       'all_nonentity_fields_equal': True},
        'relation': 'Copy all frozen cut .3 baseline non-entity fields, including unchanged .4-origin association.',
        'selection': 'Fixed HGB160 and cut .3; stable descending probability/length, ascending offset greedy nonoverlap.',
        'outputs': {name: sha(target / name) for name in ('augmented.jsonl', 'selector-scores.jsonl', 'model-vectorizer.pkl')},
        'seconds': time.monotonic() - started, 'original_dev_read': False,
        'evaluation_gold_read': False, 'official_actions': False,
    }
    with (target / 'manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: manifest[name] for name in
                      ('fold', 'training_examples', 'training_positive', 'seconds', 'outputs')}))


if __name__ == '__main__':
    main()
