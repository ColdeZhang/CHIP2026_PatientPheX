"""One fixed paired document-OOF screen; known F1 labels train meta models.

The published neural/base candidates excluded all F1 task fitting. This is
posthoc development CV, not an untouched validation set or a blind test.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import warnings

from evaluate import module, read_jsonl, ref, relations, sha, write


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(entry):
    assert ref(entry['path']) == entry, entry['path']
    return Path(entry['path'])


def pairs(document):
    return {(str(document['pmc_id']), patient, phenotype)
            for patient, values in relations(document).items() for phenotype in values}


def assemble(base, selected, universe):
    """Protected base-only keys are immutable; no model receives those rows."""
    assert selected <= universe
    result = []
    for original in base:
        pmc = str(original['pmc_id'])
        original_keys = pairs(original)
        expected = (original_keys - universe) | {key for key in selected if key[0] == pmc}
        doc = deepcopy(original)
        for association in doc['association']:
            pid = str(association['patient_id'])
            association['phenotype'] = sorted(key[2] for key in expected if key[:2] == (pmc, pid))
        assert pairs(doc) == expected
        assert {k: v for k, v in doc.items() if k != 'association'} == {
            k: v for k, v in original.items() if k != 'association'}
        assert (pairs(doc) - universe) == (original_keys - universe)
        result.append(doc)
    return result


def inputs(plan):
    refs = plan['refs']
    for name, entry in refs.items():
        # Known labels are authorized for meta fitting, but not CPU preparation.
        if name != 'gold':
            verify(entry)
    original = read_jsonl(refs['input']['path'])
    base = read_jsonl(refs['baseline']['path'])
    scores = read_jsonl(refs['scores']['path'])
    with gzip.open(refs['encoded']['path'], 'rt') as stream:
        encoded = json.load(stream)
    expected = {tuple(key) for doc in encoded for key in doc['keys']}
    assert all('labels' not in doc for doc in encoded)
    keys = [tuple(str(v) for v in row['key']) for row in scores]
    assert len(keys) == len(set(keys)) == len(expected) == 2874
    assert set(keys) == expected
    scores.sort(key=lambda row: tuple(str(v) for v in row['key']))
    keys = [tuple(str(v) for v in row['key']) for row in scores]
    ids = {str(doc['pmc_id']) for doc in original}
    assert len(ids) == len(base) == len(encoded) == 16
    assert {str(doc['pmc_id']) for doc in base} == ids
    assert sum(len(doc['entities']) for doc in base) == 1189
    patient_ids = {str(doc['pmc_id']): {str(row['patient_id']) for row in doc['patient']} for doc in original}
    assert sum(map(len, patient_ids.values())) == 39
    for doc in base:
        patients = [str(row['patient_id']) for row in doc['association']]
        assert len(patients) == len(set(patients)) and set(patients) == patient_ids[str(doc['pmc_id'])]
        source_doc = next(row for row in original if str(row['pmc_id']) == str(doc['pmc_id']))
        assert {row['patient_id'] for row in doc['association']} == {row['patient_id'] for row in source_doc['patient']}
    for row in scores:
        pmc, patient, concept = (str(v) for v in row['key'])
        assert pmc in ids and patient in patient_ids[pmc]
        assert math.isfinite(row['logit']) and 0 <= row['probability'] <= 1
        assert abs(row['probability'] - 1 / (1 + math.exp(-max(-80., min(80., row['logit']))))) < 1e-8
    groups = plan['heldout_document_groups']
    flat = [pmc for group in groups for pmc in group]
    assert len(groups) == 4 and all(len(group) == 4 for group in groups)
    assert len(flat) == len(set(flat)) == 16 and set(flat) == ids
    r1 = json.loads(Path(refs['r1_split_plan']['path']).read_text())
    assert groups == r1['heldout_document_groups']
    ordered = sorted(ids, key=lambda pmc: (hashlib.sha256(('20261024:' + pmc).encode()).hexdigest(), pmc))
    assert groups == [ordered[k::4] for k in range(4)]
    base_keys = set().union(*(pairs(doc) for doc in base))
    reconstructed = assemble(base, base_keys & expected, expected)
    assert all(pairs(a) == pairs(b) for a, b in zip(base, reconstructed))
    return original, base, scores, keys, base_keys, expected, groups


def contract(plan):
    original, base, scores, keys, base_keys, universe, groups = inputs(plan)
    # Distinguish valid HPO compound semantics from text/mixed identifiers.
    synthetic = {'pmc_id': 'doc', 'entities': [], 'association': [
        {'patient_id': 7, 'phenotype': ['HP:0000001;HP:0000002', 'HP:0000003;free text', 'MESH:X', 'free text']}]}
    assert relations(synthetic)['7'] == {'HP:0000001', 'HP:0000002', 'HP:0000003;free text', 'MESH:X', 'free text'}
    u = {('doc', '7', 'HP:0000001'), ('doc', '7', 'HP:0000004')}
    assert pairs(assemble([synthetic], {('doc', '7', 'HP:0000004')}, u)[0]) == (pairs(synthetic) - u) | {('doc', '7', 'HP:0000004')}
    # Same canonical splitter as actual v2, without reading any gold.
    validator = module(plan['refs']['validator']['path'], 'validate_predictions')
    evaluator = module(plan['refs']['evaluator']['path'], 'docre_meta_contract_evaluator')
    # Evaluator preserves raw patient-ID type; our internal stable keys are
    # strings, but assembly retains original IDs and checks them against input.
    assert {str(k): v for k, v in evaluator.associations(synthetic).items()} == dict(relations(synthetic))
    assert assemble([synthetic], set(), u)[0]['association'][0]['patient_id'] == 7
    fold_counts = []
    for group in groups:
        test = [i for i, key in enumerate(keys) if key[0] in group]
        train = [i for i, key in enumerate(keys) if key[0] not in group]
        assert not (set(train) & set(test)) and len(train) + len(test) == 2874
        assert len({keys[i][0] for i in train}) == 12 and len({keys[i][0] for i in test}) == 4
        fold_counts.append({'train_documents': 12, 'held_documents': 4, 'train_pairs': len(train), 'held_pairs': len(test)})
    import sklearn
    import numpy
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    classifier = LogisticRegression(**plan['logistic'])
    scaler = StandardScaler()
    assert classifier.class_weight is None and classifier.solver == 'lbfgs' and classifier.C == 1
    return {'status': 'PASS_COMPLETE_UNLABELED_INPUT_KEY_SPLIT_AND_CANONICAL_RELATION_CONTRACT',
            'at_utc': now(), 'documents': 16, 'patients': 39, 'entities': 1189,
            'modeled_candidate_pairs': len(keys), 'base_selected_relations': len(base_keys),
            'base_selected_inside_candidates': len(base_keys & universe),
            'protected_base_outside_candidates': len(base_keys - universe),
            'candidate_union_size': len(base_keys | universe), 'fold_counts': fold_counts,
            'score_valid_is_constant_one_on_all_modeled_rows': True,
            'gold_read': False, 'classifier_fit_calls': 0, 'scaler_fit_calls': 0, 'v2_calls': 0,
            'versions': {'python': sys.version, 'sklearn': sklearn.__version__, 'numpy': numpy.__version__},
            'source': ref(__file__), 'helper': plan['refs']['helper']}


def dump_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(plan, plan_path):
    start = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError('Fixed 120 second CPU window expired; do not refit on recovery')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'started.json', {'at_utc': now(), 'plan': ref(plan_path), 'source': ref(__file__), 'budget_seconds': 120})
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    warnings.simplefilter('error', ConvergenceWarning)
    original, base, scores, keys, base_keys, universe, groups = inputs(plan)
    verify(plan['refs']['gold'])
    # Explicit known F1 labels train each meta model on its other 12 documents.
    gold = read_jsonl(plan['refs']['gold']['path'])
    assert {str(doc['pmc_id']) for doc in gold} == {str(doc['pmc_id']) for doc in base}
    for doc in gold:
        baseline_doc = next(row for row in base if str(row['pmc_id']) == str(doc['pmc_id']))
        assert {row['patient_id'] for row in doc['association']} == {row['patient_id'] for row in baseline_doc['association']}
    gold_keys = set().union(*(pairs(doc) for doc in gold))
    y = np.array([int(key in gold_keys) for key in keys], dtype=np.int64)
    features = np.array([[int(key in base_keys), row['logit'], 1.] for key, row in zip(keys, scores)], dtype=np.float64)
    assert np.isfinite(features).all() and np.array_equal(features[:, 2], np.ones(len(keys)))
    write(out / 'label-and-feature-manifest.json', {
        'at_utc': now(), 'gold': plan['refs']['gold'], 'feature_names': plan['features'],
        'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
        'label_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
        'candidate_positive_labels': int(y.sum()), 'full_gold_relations': len(gold_keys),
        'rows': len(keys), 'disclosure': plan['disclosure']})
    probabilities = {arm: np.full(len(keys), np.nan) for arm in plan['features']}
    records = []
    fit_calls = 0
    for fold, group in enumerate(groups):
        held = np.array([key[0] in group for key in keys])
        training = ~held
        assert len(set(y[training])) == 2
        train_docs = sorted({key[0] for key, keep in zip(keys, training) if keep})
        assert len(train_docs) == 12 and not set(train_docs) & set(group)
        for arm in plan['features']:
            columns = [0] if arm == 'base_only' else [0, 1, 2]
            pipeline = Pipeline([('scaler', StandardScaler()), ('logistic', LogisticRegression(**plan['logistic']))])
            fit_calls += 1
            pipeline.fit(features[training][:, columns], y[training])
            prediction = pipeline.predict_proba(features[held][:, columns])[:, 1]
            probabilities[arm][held] = prediction
            model_path = out / f'fold{fold}.{arm}.pkl'
            with model_path.open('xb') as stream:
                pickle.dump(pipeline, stream, protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
            with model_path.open('rb') as stream:
                recovered = pickle.load(stream)
            assert np.array_equal(prediction, recovered.predict_proba(features[held][:, columns])[:, 1])
            scaler = pipeline.named_steps['scaler']
            assert scaler.n_samples_seen_ == int(training.sum())
            assert np.allclose(scaler.mean_, features[training][:, columns].mean(axis=0), atol=1e-12, rtol=0)
            classifier = pipeline.named_steps['logistic']
            entry = {'fold': fold, 'arm': arm, 'train_documents': train_docs, 'held_documents': group,
                     'train_rows': int(training.sum()), 'held_rows': int(held.sum()),
                     'train_positives': int(y[training].sum()), 'model': ref(model_path),
                     'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                     'coef': classifier.coef_.tolist(), 'intercept': classifier.intercept_.tolist(),
                     'n_iter': classifier.n_iter_.tolist(), 'reload_probabilities_exact': True}
            write(out / f'fold{fold}.{arm}.json', entry)
            records.append(entry)
    assert fit_calls == 8 and all(np.isfinite(values).all() for values in probabilities.values())
    dump_jsonl(out / 'oof-scores.jsonl', [
        {'key': key, 'held_meta_fold': next(k for k, group in enumerate(groups) if key[0] in group),
         'base_selected': bool(key in base_keys), 'docre_logit': row['logit'], 'score_valid': True,
         **{arm: float(probabilities[arm][i]) for arm in probabilities}}
        for i, (key, row) in enumerate(zip(keys, scores))])
    predictions = {}
    for arm, probs in probabilities.items():
        rows = assemble(base, {keys[i] for i in range(len(keys)) if probs[i] >= .5}, universe)
        path = out / f'{arm}.prediction.jsonl'
        dump_jsonl(path, rows)
        predictions[arm] = {'rows': rows, 'file': ref(path)}
    write(out / 'complete-freeze.json', {'at_utc': now(), 'predictions': {k: v['file'] for k, v in predictions.items()},
          'scores': ref(out / 'oof-scores.jsonl'), 'classifier_fits': fit_calls, 'scaler_fits': fit_calls,
          'documents': 16, 'patients': 39, 'entities': 1189, 'new_v2_calls_before_freeze': 0,
          'known_F1_labels_used_for_meta_training_before_freeze': True,
          'each_model_excluded_own_held4_document_labels': True, 'disclosure': plan['disclosure']})
    validator = module(plan['refs']['validator']['path'], 'validate_predictions')
    evaluator = module(plan['refs']['evaluator']['path'], 'docre_meta_public_evaluator')
    hpo = validator.read_hpo(plan['refs']['hpo']['path'])
    baseline_metrics = json.loads(Path(plan['refs']['baseline_metrics']['path']).read_text())['metrics']
    results = {}
    write(out / 'evaluation-started.json', {'at_utc': now(), 'complete_freeze': ref(out / 'complete-freeze.json'), 'expected_new_v2_calls': 2, 'expected_old_v2_calls': 0})
    for arm, entry in predictions.items():
        validation = validator.validate_predictions(original, entry['rows'], hpo)
        write(out / f'{arm}.validation.json', validation)
        assert not validation['errors'] and not validation['warnings'], validation
        metrics = evaluator.evaluate_report(gold, entry['rows'], 'all', 'text')
        write(out / f'{arm}.metrics.json', metrics)
        new_keys = set().union(*(pairs(doc) for doc in entry['rows']))
        added, removed = new_keys - base_keys, base_keys - new_keys
        results[arm] = {'prediction': entry['file'], 'metrics': metrics['metrics'], 'counts': metrics['counts'],
                        'delta_from_complete_base': {key: metrics['metrics'][key] - baseline_metrics[key] for key in baseline_metrics},
                        'relation_delta': {'added': len(added), 'added_TP': len(added & gold_keys), 'added_FP': len(added - gold_keys),
                                           'removed': len(removed), 'removed_TP': len(removed & gold_keys), 'removed_FP': len(removed - gold_keys)}}
    keep = results['with_docre']['metrics']['Score'] > max(baseline_metrics['Score'], results['base_only']['metrics']['Score'])
    result = {'status': 'COMPLETE_FIXED_PAIRED_F1_META_OOF', 'keep_for_F0': keep,
              'baseline': baseline_metrics, 'results': results, 'classifier_fit_calls': fit_calls,
              'scaler_fit_calls': fit_calls, 'new_v2_calls': 2, 'old_v2_calls': 0,
              'elapsed_seconds': time.monotonic() - start, 'plan': ref(plan_path),
              'disclosure': plan['disclosure'], 'complete_freeze': ref(out / 'complete-freeze.json')}
    write(out / 'result.json', result)
    signal.alarm(0)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['contract', 'run'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    if args.action == 'contract':
        result = contract(plan)
        write(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(plan, args.plan)
