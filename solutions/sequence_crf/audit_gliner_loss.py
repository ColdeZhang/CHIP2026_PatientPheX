#!/usr/bin/env python3
"""Small CPU counterexamples for explicit valid-span binary loss and gradients."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import train_gliner_supervised as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    logits = torch.tensor([-2., 2., 1000., -5., -2., -1000.]).reshape(2, 3, 1, 1).requires_grad_()
    labels = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    loss = trainer.masked_span_bce(logits, labels, mask)
    reference = (torch.nn.functional.softplus(logits.reshape(2, -1)[mask])
                 -logits.reshape(2, -1)[mask]*labels[mask]).mean()
    assert torch.allclose(loss, reference, atol=1e-7)
    loss.backward()
    gradient = logits.grad.reshape(2, -1)
    assert (gradient[mask & (labels == 1)] < 0).all()
    assert (gradient[mask & (labels == 0)] > 0).all()
    assert (gradient[~mask] == 0).all()
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    changed = labels.clone()
    changed[~mask] = 1-changed[~mask]
    assert trainer.masked_span_bce(logits.detach(), changed, mask) == loss.detach()
    negative_logits = torch.tensor([-4., -1., 1., 4.]).reshape(1, 4, 1, 1).requires_grad_()
    negative_loss = trainer.masked_span_bce(negative_logits, torch.zeros(1, 4), torch.ones(1, 4).bool())
    negative_loss.backward()
    assert (negative_logits.grad > 0).all()
    tail_logits = torch.tensor([2., 1000.]).reshape(1, 2, 1, 1).requires_grad_()
    tail_loss = trainer.masked_span_bce(tail_logits, torch.tensor([[1., 0.]]), torch.tensor([[True, False]]))
    tail_loss.backward()
    assert tail_logits.grad.flatten()[0] < 0 and tail_logits.grad.flatten()[1] == 0
    # Normalization must be by valid spans, not padded tensor size.
    assert torch.allclose(tail_loss, torch.nn.functional.softplus(torch.tensor(-2.)))
    report = {'all_checks_pass': True, 'device': 'cpu', 'torch': importlib.metadata.version('torch'),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'trainer_sha256': hashlib.sha256(Path(trainer.__file__).read_bytes()).hexdigest(),
              'checks': ['mixed positive/negative targets equal closed-form BCE',
                         'positive gradients raise positive logits; negative gradients lower negatives',
                         'invalid/padded logits have exactly zero gradient',
                         'changing masked targets does not change loss',
                         'all-negative batch gradients are positive',
                         'one-item tail batch normalizes by valid count only'],
              'mixed_loss': float(loss.detach()), 'mixed_gradient': gradient.tolist(),
              'all_negative_loss': float(negative_loss.detach()),
              'all_negative_gradient': negative_logits.grad.flatten().tolist(),
              'tail_loss': float(tail_loss.detach()), 'tail_gradient': tail_logits.grad.flatten().tolist()}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
