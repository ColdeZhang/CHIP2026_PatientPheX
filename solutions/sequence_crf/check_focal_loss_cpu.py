#!/usr/bin/env python3
"""Synthetic CPU formula/gradient verification; reads no task data or model."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

from focal_loss import masked_span_focal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--upstream-helper', required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('frozen_upstream_loss', args.upstream_helper)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    checks = {}

    def calculate(z, y, mask=None, alpha=.75, gamma=2., dtype=torch.float64):
        logits = torch.tensor(z, dtype=dtype).reshape(1, -1, 1).requires_grad_()
        labels = torch.tensor(y, dtype=dtype).reshape(1, -1)
        valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else torch.tensor(mask).reshape(1, -1)
        loss = masked_span_focal(logits, labels, valid, alpha=alpha, gamma=gamma)
        gradient = torch.autograd.grad(loss, logits)[0].flatten()
        return float(loss), gradient, logits, labels, valid

    # Independent scalar formula and analytic derivative, not this helper's graph.
    z = [-8., -2., 0., 2., 8.] * 2
    y = [0.] * 5 + [1.] * 5
    expected_loss, expected_grad = [], []
    for value, target in zip(z, y):
        t = 2 * target - 1
        signed = t * value
        q = 1 / (1 + math.exp(signed))
        ce = max(-signed, 0.) + math.log1p(math.exp(-abs(signed)))
        weight = .75 if target else .25
        expected_loss.append(weight * q ** 2 * ce)
        expected_grad.append(-t * weight * q ** 2 * (q + 2 * (1-q) * ce) / len(z))
    loss, grad, logits, labels, valid = calculate(z, y)
    assert math.isclose(loss, sum(expected_loss)/len(z), rel_tol=1e-13, abs_tol=1e-15)
    torch.testing.assert_close(grad, torch.tensor(expected_grad, dtype=grad.dtype), rtol=1e-12, atol=1e-15)
    checks['independent_formula_and_analytic_gradient'] = {'loss': loss, 'maximum_gradient_error': float((grad-torch.tensor(expected_grad, dtype=grad.dtype)).abs().max())}
    old = upstream.focal_loss_with_logits(logits.flatten(), labels.flatten(), alpha=.75, gamma=2., prob_margin=0., label_smoothing=0., reduction='mean')
    old_grad = torch.autograd.grad(old, logits)[0].flatten()
    torch.testing.assert_close(old, torch.tensor(loss, dtype=old.dtype), rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(old_grad, grad, rtol=1e-10, atol=1e-13)
    checks['upstream_direct_helper_moderate_logits'] = 'value and gradient equal with explicit keyword parameters; no broken SpanModel.loss call'

    for alpha in [.75, -1]:
        actual, gradient, zt, yt, mask = calculate(z, y, alpha=alpha, gamma=0)
        weights = alpha*yt + (1-alpha)*(1-yt) if alpha >= 0 else torch.ones_like(yt)
        expected = (F.binary_cross_entropy_with_logits(zt.flatten(), yt.flatten(), reduction='none') * weights.flatten()).mean()
        eg = torch.autograd.grad(expected, zt)[0].flatten()
        torch.testing.assert_close(expected, torch.tensor(actual, dtype=expected.dtype), rtol=0, atol=1e-15)
        torch.testing.assert_close(gradient, eg, rtol=0, atol=1e-15)
    checks['gamma_zero'] = 'alpha .75 exactly weighted BCE; alpha -1 exactly unweighted BCE, including gradients'

    zero, zg, *_ = calculate([0., 0.], [1., 0.])
    assert math.isclose(zero, math.log(2)/8, rel_tol=1e-14)
    assert math.isclose(abs(float(zg[0]/zg[1])), 3., rel_tol=1e-14)
    checks['zero_logits_and_alpha_direction'] = {'loss': zero, 'positive_to_negative_gradient_ratio': abs(float(zg[0]/zg[1]))}

    for dtype in [torch.float32, torch.float64]:
        extreme, eg, *_ = calculate([-1000., 1000., 1000., -1000.], [1., 0., 1., 0.], dtype=dtype)
        assert math.isfinite(extreme) and torch.isfinite(eg).all()
        assert eg[0] < 0 and eg[1] > 0 and eg[2] == 0 and eg[3] == 0
        assert math.isclose(extreme, 250., rel_tol=1e-6)
    checks['extreme_logits'] = 'FP32/FP64 +/-1000 finite; hard mistakes retain correcting gradients, easy saturation zero'

    original, og, *_ = calculate([2., -2.], [1., 0.])
    padded, pg, *_ = calculate([2., -2., float('nan'), float('inf'), -1000.], [1., 0., 0., 1., 0.], [1, 1, 0, 0, 0])
    assert original == padded and torch.equal(og, pg[:2]) and (pg[2:] == 0).all()
    flipped, fg, *_ = calculate([2., -2., float('nan'), float('inf'), 1000.], [1., 0., 1., 0., 1.], [1, 1, 0, 0, 0])
    assert flipped == padded and torch.equal(fg, pg)
    checks['padding_and_masked_target_flip'] = 'loss/valid gradients exact; all invalid gradients zero, including NaN/Inf padding'

    zz = torch.tensor([[[2.], [1000.], [1000.]], [[-2.], [1.], [0.]]], dtype=torch.float64, requires_grad=True)
    yy = torch.tensor([[1., 0., 1.], [0., 0., 1.]], dtype=torch.float64)
    mm = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    batched = masked_span_focal(zz, yy, mm)
    flat, *_ = calculate([2., -2., 1., 0.], [1., 0., 0., 1.])
    assert math.isclose(float(batched), flat, rel_tol=1e-14)
    checks['unequal_document_counts_denominator'] = 'sum over four actual valid spans / 4, not per-document means or padded slots'
    for targets, direction in [([0., 0., 0.], 1), ([1., 1., 1.], -1)]:
        ll, gg, *_ = calculate([-2., 0., 2.], targets)
        assert math.isfinite(ll) and (gg*direction > 0).all()
    checks['all_negative_or_positive_tail_batch'] = 'finite correct-sign gradients'
    grad_input = torch.tensor([[[-2.], [.3], [2.]]], dtype=torch.float64, requires_grad=True)
    grad_labels = torch.tensor([[1., 0., 1.]], dtype=torch.float64)
    grad_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda a: masked_span_focal(a, grad_labels, grad_mask), (grad_input,), eps=1e-6, atol=1e-6, rtol=1e-4)
    checks['finite_difference_gradcheck'] = True
    for keywords in [{'alpha': -2}, {'alpha': 1.1}, {'gamma': -1}, {'gamma': float('nan')}]:
        try:
            calculate([0.], [1.], **keywords)
        except ValueError:
            pass
        else:
            raise AssertionError(keywords)
    for arguments in [([0.], [.5], [1]), ([0.], [1.], [0]), ([float('inf')], [1.], [1])]:
        try:
            calculate(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(arguments)
    checks['invalid_arguments_and_zero_valid_guard'] = True
    assert not torch.cuda.is_initialized()
    report = {'status': 'PASS_CPU_synthetic_only', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'formula': 'mean_valid alpha_t * sigmoid(-(2y-1)z)^gamma * BCEWithLogits(z,y)',
              'fixed_proposed_parameters': {'alpha': .75, 'gamma': 2., 'normalization': 'actual valid-span mean'},
              'checks': checks, 'versions': {'python': platform.python_version(), 'torch': torch.__version__},
              'cuda_initialized': torch.cuda.is_initialized(), 'task_data_read': False, 'model_loaded': False,
              'source_sha256': sha(__file__), 'loss_source_sha256': sha(Path(__file__).with_name('focal_loss.py')),
              'upstream_helper_path': args.upstream_helper, 'upstream_helper_sha256': sha(args.upstream_helper),
              'limitations': ['No model backward, resource smoke, training or quality evaluation.',
                              'Stable BCE differs from upstream epsilon-clamped log formula at extreme logits.',
                              'Keep valid-span mean to isolate objective weighting; not upstream sum-reduction reproduction.']}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
