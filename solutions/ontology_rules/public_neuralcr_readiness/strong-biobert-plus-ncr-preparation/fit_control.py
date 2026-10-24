"""One fresh strong BioBERT plus NCR HGB160 fold; held scores only, no quality.

Source preparation is not fit authorization. A later frozen plan must set
execution_ready=true after the root authorizes actual fitting. No task files,
NumPy/sklearn/model modules, or legacy helpers are read/imported on import here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time

from prepare_pools import (
    FEATURES, OLD_FEATURES, PUBLIC_FEATURES, BIOBERT_FEATURES, BASE_MANIFEST_SHA256,
    BASE_FIT_SOURCE_SHA256, checked, read, ref, reject_old_posterior,
    require, rows, same_json, same_old, sha, sources, validate_base_metadata, write)


HELPER_SHA256 = 'e8bac3d485074052175a5a365d750fd3d9fdb8d5b131260ba42bf10af6578aff'
HGB_ARGUMENTS = dict(max_iter=160, max_leaf_nodes=15, min_samples_leaf=35,
                     l2_regularization=5., learning_rate=.06,
                     random_state=20261023, early_stopping=False)


def keyed(path):
    data = rows(path)
    out = {row['pmc_id']: row for row in data}
    require(len(out) == len(data), 'Duplicate document IDs')
    return out


def feature(multi, row, candidate):
    reject_old_posterior(candidate)
    require(type(candidate['ncr_present']) is bool and type(candidate['ncr_score']) is float and
            math.isfinite(candidate['ncr_score']) and 0 <= candidate['ncr_score'] <= 1,
            'NCR feature schema/range mismatch')
    require(candidate['ncr_present'] or candidate['ncr_score'] == 0.0,
            'Missing NCR source must have the explicit zero sentinel')
    values = multi.features(row['input'], candidate, row['base']) | {
        name: candidate[name] for name in OLD_FEATURES + FEATURES}
    require(set(BIOBERT_FEATURES) <= set(values) and
            'selector_probability' not in values, 'Missing preserved BioBERT feature or forbidden posterior')
    return values


def load_prior(plan):
    paths = [path for path, digest in plan['sources'].items()
             if Path(path).name == 'fit_helpers.py' and digest == HELPER_SHA256]
    require(len(paths) == 1, 'Plan must bind exactly the previously audited fit_helpers source')
    spec = importlib.util.spec_from_file_location('ncr_frozen_fit_helpers', paths[0])
    require(spec is not None and spec.loader is not None, 'Cannot load frozen fit helper')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    require(sha(prior.__file__) == HELPER_SHA256, 'Frozen fit helper changed')
    return prior


def state_array(array, np):
    """Actual stored array summary; structural integer slots are counted too."""
    require(isinstance(array, np.ndarray), 'Unexpected non-array fitted state')
    if array.dtype.names:
        fields = []
        slots = 0
        for name in array.dtype.names:
            values = array[name]
            require(values.dtype.names is None and values.dtype.kind in 'biuf',
                    'Unsupported fitted structured field: ' + name)
            size = int(values.size)
            slots += size
            fields.append({'name': name, 'dtype': str(values.dtype),
                           'scalar_slots': size, 'byte_offset': int(array.dtype.fields[name][1])})
    else:
        require(array.dtype.kind in 'biuf', 'Unsupported fitted state array dtype')
        slots = int(array.size)
        fields = None
    result = {'shape': [int(x) for x in array.shape], 'dtype': str(array.dtype),
              'bytes': int(array.nbytes), 'numeric_scalar_slots_including_structure': slots,
              'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
    if fields is not None:
        result['structured_fields'] = fields
        result['itemsize'] = int(array.dtype.itemsize)
    return result


def tree_inventory(model, vectorizer, np):
    """Inventory actual fitted arrays without inferring F, node count or model size."""
    predictors = model._predictors
    require(len(predictors) == int(model.n_iter_) and all(len(items) == 1 for items in predictors),
            'Expected one binary-classification tree per actual boosting iteration')
    totals = {'trees': 0, 'node_records': 0, 'leaf_records': 0, 'internal_records': 0,
              'all_node_value_entries': 0, 'leaf_value_entries': 0,
              'stored_numeric_threshold_entries': 0, 'stored_binned_threshold_entries': 0,
              'active_numeric_split_thresholds': 0,
              'full_node_field_numeric_scalar_slots': 0, 'categorical_bitset_scalar_slots': 0,
              'node_array_bytes': 0, 'categorical_bitset_bytes': 0}
    trees = []
    for iteration, items in enumerate(predictors):
        for tree_index, tree in enumerate(items):
            nodes = tree.nodes
            require(isinstance(nodes, np.ndarray) and nodes.ndim == 1 and len(nodes) > 0 and
                    nodes.dtype.names is not None and
                    {'value', 'num_threshold', 'bin_threshold', 'is_leaf', 'left', 'right',
                     'feature_idx', 'is_categorical'} <= set(nodes.dtype.names),
                    'Unexpected actual HGB node layout')
            leaf = nodes['is_leaf'].astype(bool)
            require(np.isin(nodes['is_leaf'], [0, 1]).all(), 'Non-boolean tree leaf marker')
            n = int(len(nodes))
            leaves = int(leaf.sum())
            internal = n - leaves
            require(leaves <= 15 and n == 2 * leaves - 1, 'Actual binary tree structure/leaf limit mismatch')
            visited = set()
            todo = [0]
            while todo:
                index = todo.pop()
                require(0 <= index < n and index not in visited, 'Tree child index cycle/duplicate/out of range')
                visited.add(index)
                if not leaf[index]:
                    require(0 <= int(nodes['feature_idx'][index]) < len(vectorizer.feature_names_),
                            'Tree feature index outside fitted vectorizer')
                    todo.extend([int(nodes['left'][index]), int(nodes['right'][index])])
            require(len(visited) == n, 'Unreachable actual node records')
            require(np.isfinite(nodes['value']).all(), 'Nonfinite fitted node value')
            numeric_splits = (~leaf) & (~nodes['is_categorical'].astype(bool))
            require(np.isfinite(nodes['num_threshold'][numeric_splits]).all(),
                    'Nonfinite active numeric split threshold')
            node_state = state_array(nodes, np)
            bitsets = {name: state_array(getattr(tree, name), np) for name in
                       ('raw_left_cat_bitsets', 'binned_left_cat_bitsets')}
            trees.append({'iteration': iteration, 'tree_index': tree_index, 'nodes': n,
                          'leaves': leaves, 'internal': internal,
                          'active_numeric_splits': int(numeric_splits.sum()),
                          'node_state': node_state, 'categorical_bitsets': bitsets})
            totals['trees'] += 1
            totals['node_records'] += n
            totals['leaf_records'] += leaves
            totals['internal_records'] += internal
            totals['all_node_value_entries'] += int(nodes['value'].size)
            totals['leaf_value_entries'] += leaves
            totals['stored_numeric_threshold_entries'] += int(nodes['num_threshold'].size)
            totals['stored_binned_threshold_entries'] += int(nodes['bin_threshold'].size)
            totals['active_numeric_split_thresholds'] += int(numeric_splits.sum())
            totals['full_node_field_numeric_scalar_slots'] += node_state['numeric_scalar_slots_including_structure']
            totals['node_array_bytes'] += node_state['bytes']
            totals['categorical_bitset_scalar_slots'] += sum(v['numeric_scalar_slots_including_structure'] for v in bitsets.values())
            totals['categorical_bitset_bytes'] += sum(v['bytes'] for v in bitsets.values())
    mapper = model._bin_mapper
    ancillary = {'baseline_prediction': state_array(model._baseline_prediction, np),
                 'classes': state_array(model.classes_, np),
                 'bin_mapper_n_bins_non_missing': state_array(mapper.n_bins_non_missing_, np),
                 'bin_mapper_thresholds_by_feature': [state_array(value, np) for value in mapper.bin_thresholds_]}
    if mapper.is_categorical_ is not None:
        ancillary['bin_mapper_is_categorical'] = state_array(mapper.is_categorical_, np)
    flat_ancillary = [value for key, value in ancillary.items() if key != 'bin_mapper_thresholds_by_feature'] + ancillary['bin_mapper_thresholds_by_feature']
    ancillary_slots = sum(value['numeric_scalar_slots_including_structure'] for value in flat_ancillary)
    ancillary_bytes = sum(value['bytes'] for value in flat_ancillary)
    totals['ancillary_array_numeric_scalar_slots'] = ancillary_slots
    totals['ancillary_array_bytes'] = ancillary_bytes
    totals['conservative_stored_numeric_scalar_slots'] = (totals['full_node_field_numeric_scalar_slots'] +
                                                         totals['categorical_bitset_scalar_slots'] + ancillary_slots)
    totals['actual_array_bytes'] = totals['node_array_bytes'] + totals['categorical_bitset_bytes'] + ancillary_bytes
    require(int(model.n_features_in_) == len(vectorizer.feature_names_) == len(mapper.bin_thresholds_),
            'Actual model/vectorizer/bin-mapper feature counts disagree')
    return {'status': 'actual_fitted_HGB_structure_inventory', 'actual_F': len(vectorizer.feature_names_),
            'actual_iterations': int(model.n_iter_), 'totals': totals, 'trees': trees,
            'ancillary_arrays': ancillary,
            'counting_scope': 'All named scalar fields of every actual tree node including structural integers, both categorical bitset arrays, baseline/classes/bin-mapper arrays. Padding bytes are in byte count/hash, not numeric-slot count; configuration scalars and Python serialization overhead are not added to array slots.',
            'not_neural_trainable_parameter_count': True,
            'leaf_values_and_active_split_thresholds_are_separately_reported': True,
            'no_counts_inferred_from_max_iter_max_leaf_or_feature_projection': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan = read(args.plan)
    require(plan['execution_ready'] is True,
            'Source-only plan: actual fit awaits explicit root authorization and a frozen ready plan')
    require(plan['cut'] == .3, 'Downstream fixed cut metadata changed; this stage does not decode')
    require(any(Path(path).resolve() == Path(__file__).resolve() for path in plan['sources']),
            'Plan omits this fit source')
    sources(plan)
    plan_sha = sha(args.plan)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require([fold['fold'] for fold in plan['folds']] == [0, 1, 2, 3], 'Expected the original fixed four folds')
    fold = plan['folds'][args.fold]
    require(plan['base_prepared_manifest']['sha256'] == BASE_MANIFEST_SHA256,
            'Not the frozen strong BioBERT base pools')
    prep = read(checked(plan['prepared_manifest']))
    base = read(checked(plan['base_prepared_manifest']))
    validate_base_metadata(base)
    require(prep['status'] == 'prepared_input_only_no_fit_or_quality' and
            prep['base_prepared_manifest'] == plan['base_prepared_manifest'] and
            list(prep['new_features']) == list(FEATURES) and
            list(prep['preserved_source_features']) == list(OLD_FEATURES) and
            prep['base_BioBERT_prefix_preserved'] is True and prep['no_old_selector_posteriors'] is True,
            'New pool manifest is not the frozen strongB plus NCR increment')
    require([item['fold'] for item in prep['folds']] == [0, 1, 2, 3], 'New pool fold order changed')
    prepared = prep['folds'][args.fold]
    original = base['folds'][args.fold]
    for name in ('old_train_rows', 'old_input_rows', 'train_ids', 'input_ids', 'nested_input_ids'):
        require(prepared[name] == original[name], 'New pool changed frozen lineage: ' + name)
    require(prepared['base_fold'] == original and
            prepared['base_pool_refs'] == {role: original[role] for role in ('train', 'input')},
            'Strong base-fold snapshot/reference chain changed')
    require(prepared['base_train_rows'] == original['new_train_rows'] and
            prepared['base_input_rows'] == original['new_input_rows'],
            'Strong base counts accidentally use inherited preB old_*')
    require(fold['base_fit_manifest'] == prepared['base_fit_manifest'], 'Plan strong base-fit lineage changed')
    base_fit = read(checked(fold['base_fit_manifest']))
    reference = read(checked(fold['reference']))
    require(base_fit['status'] == 'fit_and_input_scores_complete_no_decode_or_quality' and
            base_fit['prepared_manifest'] == plan['base_prepared_manifest'] and
            base_fit['reference'] == fold['reference'] and base_fit['fold'] == args.fold and
            base_fit['source_sha256'] == BASE_FIT_SOURCE_SHA256 and
            base_fit['training_examples'] == original['new_train_rows'] and
            base_fit['input_candidates'] == original['new_input_rows'] and
            list(base_fit['new_features']) == BIOBERT_FEATURES and
            set(BIOBERT_FEATURES) <= set(base_fit['features']) and
            'selector_probability' not in base_fit['features'], 'Not the completed strong BioBERT feature recipe')
    require(any(Path(path).name == 'fit_control.py' and digest == BASE_FIT_SOURCE_SHA256
                for path, digest in plan['sources'].items()), 'Plan omits the original strong BioBERT fit source')
    pre_b_fit = read(checked(base_fit['old_fit_manifest']))
    require(base_fit['old_fit_manifest'] == original['old_fit_manifest'] and
            pre_b_fit['old_reference_sha256'] == fold['reference']['sha256'], 'Historical preB feature-code loader lineage changed')
    prior = load_prior(plan)
    require(base_fit['label_function_sha256'] == sha(prior.__file__), 'Strong BioBERT label helper changed')
    legacy, multi, modules = prior.load_modules(plan, pre_b_fit, reference)
    require(all(sha(module.__file__) == base_fit['dependencies'][module.__name__] for module in modules),
            'Strong BioBERT feature dependencies changed')
    single = multi.single
    np = single.np
    require(legacy.PUBLIC_FEATURES + legacy.BIOFORMER_FEATURES == PUBLIC_FEATURES, 'Public feature recipe changed')
    for package, actual in (('sklearn', single.sklearn.__version__), ('numpy', np.__version__)):
        if package in base_fit.get('runtime', {}):
            require(actual == base_fit['runtime'][package], 'Runtime differs from frozen base fold: ' + package)
    config = reference['config']
    require(checked(fold['train_gold']).resolve() == Path(config['train']).resolve() and
            sha(config['train']) == reference['inputs']['train'] and
            sha(config['input']) == reference['inputs']['input'], 'Frozen train/input references changed')
    # Only the authorized execution path below reads the original train48 labels.
    truth = keyed(config['train'])
    inputs = keyed(config['input'])
    require(len(truth) == 48 and len(inputs) == 16 and not truth.keys() & inputs.keys(), 'Outer 48/16 isolation failed')
    require(set(truth) == set(prepared['train_ids']) and list(inputs) == prepared['input_ids'], 'Outer document identities/order changed')
    require(all(set(doc) == {'pmc_id', 'pmid', 'patient', 'full_text'} for doc in inputs.values()),
            'Held input contains labels or unexpected fields')
    specs = read(config['training_manifest'])
    require(sha(config['training_manifest']) == reference['inputs']['training_manifest'] and
            len(specs) == len(base_fit['training_origins']) == len(prepared['nested_input_ids']) == 3,
            'Nested training manifest changed')
    origins = []
    seen = set()
    for index, spec in enumerate(specs):
        actual = {name: sha(path) for name, path in spec.items()}
        require(actual == base_fit['training_origins'][index], 'Nested generator source SHA changed')
        origins.append(actual)
        nested_gold = keyed(spec['gold'])
        generator = keyed(spec['generator_train'])
        nested_input = keyed(spec['input'])
        ids = set(prepared['nested_input_ids'][index])
        require(set(nested_gold) == set(nested_input) == ids and len(ids) == 16 and
                set(generator) == set(truth) - ids and len(generator) == 32 and not seen & ids,
                'Nested 32-to-16 source isolation failed')
        require(all(doc == truth[did] for did, doc in generator.items()), 'Generator labels differ from allowed train48')
        for did, doc in nested_gold.items():
            require(doc == truth[did] and nested_input[did] == prior.strip_labels(doc),
                    'Nested gold/input lineage mismatch')
        seen |= ids
    require(seen == set(truth), 'Nested held inputs do not partition train48')
    pools = {}
    nold = {}
    nnew = {}
    old_positives = 0
    xs = []
    ys = []
    for role in ('train', 'input'):
        before = rows(checked(original[role]))
        after = rows(checked(prepared[role]))
        require(len(before) == len(after) == (48 if role == 'train' else 16) and
                [row['pmc_id'] for row in before] == [row['pmc_id'] for row in after] == prepared[role + '_ids'],
                'Pool document order/count changed')
        nold[role] = nnew[role] = 0
        for previous, row in zip(before, after):
            require(same_json({k: v for k, v in previous.items() if k != 'candidates'},
                              {k: v for k, v in row.items() if k != 'candidates'}), 'Pool metadata/base/input changed')
            did = row['pmc_id']
            same_old(row['candidates'], previous['candidates'])
            require(row['input'] == (prior.strip_labels(truth[did]) if role == 'train' else inputs[did]),
                    'Pool input content does not match original input')
            expected_origin = (next(i for i, group in enumerate(prepared['nested_input_ids']) if did in group)
                               if role == 'train' else None)
            require(row['origin_index'] == expected_origin, 'Pool generator origin changed')
            nold[role] += len(previous['candidates'])
            nnew[role] += len(row['candidates'])
            for candidate in row['candidates']:
                reject_old_posterior(candidate)
            if role == 'train':
                old_positives += sum(prior.labels(truth[did], previous['candidates']))
                ys.extend(prior.labels(truth[did], row['candidates']))
                xs.extend(feature(multi, row, candidate) for candidate in row['candidates'])
        require(nold[role] == original['new_' + role + '_rows'] and
                nnew[role] == prepared['new_' + role + '_rows'] and nnew[role] >= nold[role],
                'Actual prepared pool row counts changed')
        pools[role] = after
    require(nold['train'] == base_fit['training_examples'] and old_positives == base_fit['training_positive'],
            'Frozen base training labels or examples changed')
    require(len(ys) == len(xs) == nnew['train'] and set(ys) == {False, True}, 'Training rows/classes are incomplete')
    vectorizer = single.DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(xs)
    names = list(vectorizer.feature_names_)
    require(set(base_fit['features']) | set(FEATURES) <= set(names) and
            set(BIOBERT_FEATURES) <= set(names) and 'selector_probability' not in names,
            'Actual fresh feature schema contains missing/forbidden columns')
    added_features = sorted(set(names) - set(base_fit['features']))
    require(all(name in FEATURES or name.startswith(('section=', 'ptype=')) for name in added_features),
            'Unexpected new feature beyond NCR and actually observed categorical values')
    require(matrix.shape == (len(ys), len(names)) and np.isfinite(matrix).all(), 'Invalid fresh training feature matrix')
    model = single.HistGradientBoostingClassifier(**HGB_ARGUMENTS)
    require(model.get_params() == base_fit['estimator'] == reference['estimator'], 'Frozen full estimator parameters changed')
    sources(plan)
    require(sha(args.plan) == plan_sha, 'Plan changed before fit')
    target = Path(plan['output_dir']) / ('fold' + str(args.fold)) / 'fitted'
    target.mkdir(parents=True, exist_ok=False)
    fit_start = time.monotonic()
    model.fit(matrix, ys)
    fit_seconds = time.monotonic() - fit_start
    require(model.n_iter_ == 160 and not model.do_early_stopping_ and
            np.array_equal(model.classes_, np.array([False, True])), 'Actual fresh HGB iteration/class contract failed')
    with (target / 'model-vectorizer.pkl').open('xb') as handle:
        pickle.dump({'model': model, 'vectorizer': vectorizer, 'features': names}, handle, protocol=5)
    inventory = tree_inventory(model, vectorizer, np)
    write(target / 'tree-state.json', inventory)
    predict_start = time.monotonic()
    with (target / 'selector-scores.jsonl').open('x', encoding='utf-8') as handle:
        for row in pools['input']:
            candidates = row['candidates']
            if candidates:
                held_matrix = vectorizer.transform([feature(multi, row, candidate) for candidate in candidates])
                require(np.isfinite(held_matrix).all(), 'Invalid held feature matrix')
                probabilities = model.predict_proba(held_matrix)[:, 1]
                require(len(probabilities) == len(candidates) and np.isfinite(probabilities).all() and
                        ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid held probabilities')
            else:
                probabilities = []
            scored = {'pmc_id': row['pmc_id'], 'candidates': [dict(candidate, selector_probability=float(p))
                       for candidate, p in zip(candidates, probabilities)]}
            handle.write(json.dumps(scored, ensure_ascii=False, allow_nan=False) + '\n')
    predict_seconds = time.monotonic() - predict_start
    sources(plan)
    require(sha(args.plan) == plan_sha, 'Plan changed during fit/inference')
    checked(plan['prepared_manifest'])
    result = {
        'status': 'fit_and_input_scores_complete_no_decode_or_quality',
        'fold': args.fold, 'source_sha256': sha(__file__), 'plan_sha256': plan_sha,
        'prepared_manifest': plan['prepared_manifest'], 'base_prepared_manifest': plan['base_prepared_manifest'],
        'base_fit_manifest': fold['base_fit_manifest'], 'preB_fit_manifest': base_fit['old_fit_manifest'],
        'reference': fold['reference'], 'preserved_source_features': OLD_FEATURES,
        'training_origins': origins, 'training_ids': prepared['train_ids'], 'input_ids': prepared['input_ids'],
        'training_documents': 48, 'input_documents': 16,
        'base_training_examples': nold['train'], 'training_examples': len(ys),
        'base_training_positive': int(old_positives), 'training_positive': int(sum(ys)),
        'base_input_candidates': nold['input'], 'input_candidates': nnew['input'],
        'features': names, 'new_features': FEATURES, 'base_feature_count': len(base_fit['features']),
        'actual_feature_count': len(names), 'actual_F': len(names), 'actual_new_feature_names': added_features,
        'estimator': model.get_params(), 'actual_iterations': int(model.n_iter_),
        'early_stopping_performed': False, 'threshold': .3,
        'threshold_scope': 'Frozen downstream metadata only; this script does not decode or scan thresholds.',
        'fresh_vectorizer_and_estimator': True, 'old_model_pickle_or_posterior_loaded': False,
        'base_BioBERT_features_and_candidates_preserved': True,
        'train_probabilities_generated': False,
        'tree_state_totals': inventory['totals'], 'tree_state': ref(target / 'tree-state.json'),
        'tree_state_count_is_conservative_storage_not_neural_trainable_parameters': True,
        'dependencies': {module.__name__: sha(module.__file__) for module in modules},
        'label_function_sha256': sha(prior.__file__),
        'outputs': {name: ref(target / name) for name in
                    ('selector-scores.jsonl', 'model-vectorizer.pkl', 'tree-state.json')},
        'runtime': {'python': platform.python_version(), 'sklearn': single.sklearn.__version__,
                    'numpy': np.__version__, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']},
        'fit_seconds': fit_seconds, 'held_predict_seconds': predict_seconds,
        'original_dev_read': False, 'evaluation_gold_read': False, 'quality_evaluation': False,
        'seconds': time.monotonic() - started
    }
    write(target / 'manifest.json', result)
    print(json.dumps({name: result[name] for name in
                      ('fold', 'status', 'training_examples', 'actual_F', 'fit_seconds', 'seconds')}))


if __name__ == '__main__':
    main()
