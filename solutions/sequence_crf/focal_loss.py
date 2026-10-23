"""Explicit focal objective for a proposed control; no trainer is changed here."""
import math


def masked_span_focal(logits, labels, span_mask, *, alpha=0.75, gamma=2.0):
    """Mean alpha_t * (1-p_t)**gamma * BCE across actual valid spans.

    Binary single-type targets and the valid-span denominator match the frozen
    BCE trainer. alpha=-1 disables class weights. Modulation is differentiated,
    not detached. Select valid slots before nonlinear operations, so padding
    (including nonfinite unused logits) contributes neither loss nor gradient.
    """
    import torch
    import torch.nn.functional as F

    if not math.isfinite(alpha) or not (alpha == -1 or 0 <= alpha <= 1):
        raise ValueError('alpha must be -1 or within [0, 1]')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    if logits.ndim < 2 or logits.shape[-1] != 1 or not logits.is_floating_point():
        raise ValueError('single-class floating logits required')
    batch = logits.shape[0]
    scores = logits.reshape(batch, -1)
    targets = labels.reshape(batch, -1).to(scores.dtype)
    valid = span_mask.reshape(batch, -1).bool()
    if scores.shape != targets.shape or scores.shape != valid.shape or not valid.any():
        raise ValueError('matching shapes and at least one valid span required')
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError('binary targets required')
    scores, targets = scores[valid], targets[valid]
    if not torch.isfinite(scores).all():
        raise ValueError('valid logits must be finite')
    ce = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
    # sigmoid(-signed_logit) avoids cancellation in 1-sigmoid(signed_logit).
    one_minus_pt = torch.sigmoid(-(2 * targets - 1) * scores)
    modulation = one_minus_pt.pow(gamma) if gamma else torch.ones_like(ce)
    weights = alpha * targets + (1 - alpha) * (1 - targets) if alpha >= 0 else torch.ones_like(ce)
    return (weights * modulation * ce).mean()
