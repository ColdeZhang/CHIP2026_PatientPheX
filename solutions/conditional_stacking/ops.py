"""Six-parameter, prior-shrunk conditional simplex gate. No fit or file I/O."""
import numpy as np

EPSILON = 1e-4
THRESHOLD = .5
PRIOR_PRECISION = 1.


def probabilities(theta, context, bits):
    theta = np.asarray(theta, dtype=np.float64).reshape(2, 3)
    x = np.asarray(context, dtype=np.float64)
    b = np.asarray(bits, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 3 and b.shape == x.shape
    assert np.isfinite(theta).all() and np.isfinite(x).all()
    assert np.isin(b, [0., 1.]).all()
    logits = np.column_stack((x @ theta.T, np.zeros(len(x))))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights = exp / exp.sum(axis=1, keepdims=True)
    experts = EPSILON + (1 - 2 * EPSILON) * b
    p = (weights * experts).sum(axis=1)
    return p, weights, experts


def objective(theta, context, bits, labels, document_ids):
    """Sum of within-document mean NLL plus .5*||theta||^2.

    Every raw-union row is used. No label balancing, B1 filter, C membership,
    patient weighting, or validation-dependent normalization.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(2, 3)
    x = np.asarray(context, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    doc = np.asarray(document_ids)
    assert len(y) == len(doc) == len(x) and np.isin(y, [0., 1.]).all()
    _, inverse, counts = np.unique(doc, return_inverse=True, return_counts=True)
    observation_weights = 1. / counts[inverse]
    p, weights, experts = probabilities(theta, x, bits)
    nll = -(y * np.log(p) + (1-y) * np.log1p(-p))
    value = np.dot(observation_weights, nll) + .5 * PRIOR_PRECISION * np.square(theta).sum()
    dloss_dp = observation_weights * (p-y) / (p*(1-p))
    dp_dlogit = weights[:, :2] * (experts[:, :2] - p[:, None])
    grad = (dloss_dp[:, None] * dp_dlogit).T @ x + PRIOR_PRECISION * theta
    assert np.isfinite(value) and np.isfinite(grad).all()
    return float(value), grad.ravel()


def selected(theta, context, bits):
    return probabilities(theta, context, bits)[0] >= THRESHOLD
