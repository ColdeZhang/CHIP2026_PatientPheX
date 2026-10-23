"""One fixed two-feature F1-only logistic calibration; no validation scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def checked(item):
    actual = ref(item['path'])
    assert actual['sha256'] == item['sha256'], item['path']
    if 'bytes' in item:
        assert actual['bytes'] == item['bytes']
    return Path(item['path'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); args = ap.parse_args()
    started = time.monotonic(); plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text()); assert ref(__file__)['sha256'] == plan['source_sha256']
    root = Path(plan['output_dir']); assert not root.exists()
    assert plan['features'] == ['binary_logit', 'old_sft_positive']
    fixed = dict(C=1.0, penalty='l2', fit_intercept=True, class_weight=None, solver='lbfgs', max_iter=1000)
    assert plan['estimator'] == fixed and plan['CPU_cap_seconds'] == 60
    diagnostic_plan = json.loads(checked(plan['diagnostic_plan']).read_text())
    assert diagnostic_plan['references']['probabilities']['sha256'] == plan['binary_probability_source']['sha256']
    assert diagnostic_plan['references']['old_raw']['sha256'] == plan['old_sft_source']['sha256']
    for item in plan['source_checks'].values(): checked(item)
    diagnostic = json.loads(checked(plan['diagnostic_summary']).read_text())
    assert diagnostic['audit'] == plan['training_pairs']
    pairs = [json.loads(line) for line in checked(plan['training_pairs']).read_text().splitlines()]
    assert len(pairs) == 1557
    ids = sorted({row['pmc_id'] for row in pairs}); assert ids == plan['F1_training_document_ids']
    assert len(ids) == 16 and not set(ids).intersection(plan['excluded_F0_document_ids'])
    assert len({(row['pmc_id'], row['patient_id'], row['concept']) for row in pairs}) == len(pairs)
    # Real, previously bound 0/1 logits are available. No clipped logit(p) fallback.
    X = np.array([[row['binary_logit'], float(row['old_sft_positive'])] for row in pairs], dtype=np.float64)
    y = np.array([row['gold'] for row in pairs], dtype=np.int64)
    assert X.shape == (1557, 2) and np.isfinite(X).all() and int(y.sum()) == 365
    assert set(X[:, 1]) == {0., 1.} and set(y) == {0, 1}
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(**fixed).fit(X, y)
    assert model.classes_.tolist() == [0, 1]
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    assert int(model.n_iter_[0]) < fixed['max_iter']
    root.mkdir()
    model_path = root/'model.pkl'
    with model_path.open('xb') as stream:
        pickle.dump(dict(model=model, features=plan['features'], plan_sha256=ref(plan_path)['sha256']), stream, protocol=5)
    # Verify serialization against coefficients, not F1 quality or a second fit.
    with model_path.open('rb') as stream: restored = pickle.load(stream)
    assert np.array_equal(restored['model'].coef_, model.coef_)
    assert np.array_equal(restored['model'].intercept_, model.intercept_)
    result = dict(status='fixed_F1_only_two_feature_LR_complete_no_validation_scoring',
                  frozen_at=datetime.now(timezone.utc).isoformat(), source=ref(__file__), plan=ref(plan_path),
                  model=ref(model_path), features=plan['features'], estimator=fixed,
                  training_pairs=plan['training_pairs'], training_document_ids=ids,
                  excluded_F0_document_ids=plan['excluded_F0_document_ids'], examples=1557, positives=365,
                  coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(), iterations=model.n_iter_.tolist(),
                  runtime=dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  seconds=time.monotonic()-started, no_F0_labels=True, no_originaldev=True,
                  no_new_neural_training=True, no_training_quality_evaluated=True,
                  probability_feature='Actual frozen binary logit1-logit0; no clipping or reconstruction from p')
    assert result['seconds'] <= 60
    with (root/'fit-freeze.json').open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
