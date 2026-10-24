"""One predeclared F1 meta-training fit. No held gold or evaluator access."""
import argparse
import itertools
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from common import FEATURES, candidates, checked, gold_set, mkdir_new, now, read_json, ref, write_bytes, write_json


def run(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    assert plan['stage'] == 'F1_META_TRAIN_ONLY' and plan['fit_calls'] == 1
    assert plan['features'] == FEATURES and plan['cut'] == 0.5
    assert plan['documents'] == 16 and plan['patients'] == 39
    assert plan['gold']['sha256'] == '665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3'
    for r in plan['sources'].values():
        checked(r)
    ancestry = read_json(plan['ancestry'])
    assert ancestry['all_task_ancestors_exclude_F0'] and ancestry['status'] == 'PASS'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = mkdir_new(output)
    started = time.monotonic()
    write_json(out / 'started.json', {'utc': now(), 'plan': ref(plan_path), 'pid': os.getpid(), 'only_f1_gold_authorized': True})
    try:
        import numpy as np
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        from threadpoolctl import threadpool_limits, threadpool_info
        assert sklearn.__version__ == '1.8.0'
        u, rows, counts = candidates(plan)
        matrix_ref = write_bytes(out / 'candidate-features.jsonl', b''.join((json.dumps(r, separators=(',', ':')) + '\n').encode() for r in rows))
        feature_freeze = write_json(out / 'features-freeze.json', {'utc': now(), 'candidate_matrix': matrix_ref, 'raw_sources': plan['raw_sources'], 'counts': counts, 'features': FEATURES, 'baseline_read': False, 'gold_read': False})
        write_json(out / 'label-read-started.json', {'utc': now(), 'feature_freeze': feature_freeze, 'split': 'F1_META_TRAIN', 'gold': plan['gold']})
        gold = gold_set(plan['gold'], u)
        x = np.asarray([r['features'] for r in rows], dtype=np.float64)
        y = np.asarray([int(tuple(r['key']) in gold) for r in rows], dtype=np.int64)
        assert set(y.tolist()) == {0, 1}
        labels_ref = write_bytes(out / 'training-labels.jsonl', b''.join((json.dumps({'key': r['key'], 'y': int(label)}, separators=(',', ':')) + '\n').encode() for r, label in zip(rows, y)))
        model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=20261024)
        actual_params = model.get_params()
        assert actual_params == plan['sklearn_parameters']
        write_json(out / 'fit-started.json', {'utc': now(), 'labels': labels_ref, 'parameters': actual_params, 'fit_call_index': 1})
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            fit_start = time.monotonic()
            model.fit(x, y)  # The sole fit call in this experiment.
            fit_seconds = time.monotonic() - fit_start
            runtime_threads = threadpool_info()
        warning_rows = [{'category': type(w.message).__name__, 'message': str(w.message)} for w in observed]
        state = {'classes': model.classes_.tolist(), 'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(), 'n_iter': model.n_iter_.tolist(), 'parameters': actual_params, 'effective_penalty': 'L2 (sklearn 1.8 default l1_ratio=0)', 'features': FEATURES, 'cut': 0.5}
        convergence = {'utc': now(), 'warnings': warning_rows, 'finite': bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()), 'n_iter': state['n_iter'], 'max_iter': 1000, 'fit_seconds': fit_seconds, 'runtime_threads': runtime_threads}
        write_json(out / 'convergence.json', convergence)
        assert convergence['finite'] and max(model.n_iter_) < 1000
        assert not any(isinstance(w.message, ConvergenceWarning) for w in observed)
        assert state['classes'] == [0, 1]
        state_ref = write_json(out / 'model.json', state)
        model_ref = write_bytes(out / 'model.pkl', pickle.dumps(model, protocol=5))
        reloaded = pickle.loads(checked(model_ref).read_bytes())
        probabilities = model.predict_proba(x)[:, 1]
        assert np.array_equal(probabilities, reloaded.predict_proba(x)[:, 1])
        patterns = []
        for pattern in itertools.product((0, 1), repeat=3):
            mask = np.all(x == pattern, axis=1)
            probability = float(model.predict_proba(np.asarray([pattern], dtype=np.float64))[0, 1])
            patterns.append({'pattern': list(pattern), 'rows': int(mask.sum()), 'positive_labels': int(y[mask].sum()), 'negative_labels': int(mask.sum() - y[mask].sum()), 'fitted_probability': probability, 'fixed_cut_selected': probability >= 0.5})
        assert sum(p['rows'] for p in patterns) == len(rows) and patterns[0]['rows'] == 0
        patterns_ref = write_json(out / 'patterns.json', {'features': FEATURES, 'scope': 'descriptive F1 meta-training; not held quality', 'patterns': patterns})
        rows_ref = write_bytes(out / 'training-probabilities.jsonl', b''.join((json.dumps({'key': r['key'], 'probability': float(p)}, separators=(',', ':')) + '\n').encode() for r, p in zip(rows, probabilities)))
        result = {'status': 'META_TRAIN_COMPLETE_F0_NOT_EVALUATED', 'utc': now(), 'plan': ref(plan_path), 'ancestry': plan['ancestry'], 'counts': counts, 'documents': 16, 'patients': 39, 'positive_labels': int(y.sum()), 'negative_labels': int(len(y)-y.sum()), 'fit_calls': 1, 'v2_calls': 0, 'F0_gold_read': False, 'baseline_read': False, 'model': model_ref, 'model_state': state_ref, 'patterns': patterns_ref, 'feature_freeze': feature_freeze, 'labels': labels_ref, 'probabilities': rows_ref, 'reload_probabilities_exact': True, 'convergence': ref(out / 'convergence.json'), 'fit_seconds': fit_seconds, 'seconds': time.monotonic()-started, 'scope': 'Known F1 results preceded this fixed meta-training. No F1 score is computed. Held evaluation is F0 only after joint freeze.'}
        write_json(out / 'result.json', result)
    except BaseException as error:
        write_json(out / 'failure.json', {'utc': now(), 'type': type(error).__name__, 'message': str(error), 'seconds': time.monotonic()-started, 'rerun_forbidden_without_inspecting_original_state': True})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.plan, a.output)
