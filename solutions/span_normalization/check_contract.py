"""Small independent loss/mask/position contracts, no task labels or model weights."""
import argparse
import json
import math
import time
import torch
from gp import multilabel_loss, rotate
from common import write_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); a = ap.parse_args()
    torch.set_num_threads(1); started = time.monotonic()
    z = torch.tensor([[[0.2, -0.3], [900.0, 0.7]]], requires_grad=True)
    y = torch.tensor([[[True, False], [False, True]]])
    valid = torch.tensor([[[True, True], [False, True]]])
    loss = multilabel_loss(z, y, valid)
    expected = math.log(1 + math.exp(-0.2) + math.exp(-0.7)) + math.log(1 + math.exp(-0.3))
    assert abs(float(loss.detach()) - expected) < 1e-6
    loss.backward(); assert z.grad[0, 0, 0] < 0 and z.grad[0, 1, 1] < 0 and z.grad[0, 0, 1] > 0
    assert z.grad[0, 1, 0].item() == 0
    z2 = z.detach().clone(); z2[0, 1, 0] = -900
    assert torch.equal(multilabel_loss(z2, y, valid), loss.detach())
    empty = torch.zeros_like(valid)
    assert multilabel_loss(z2, empty, empty).item() == 0
    torch.manual_seed(7); x = torch.randn(2, 5, 64)
    actual = rotate(x); reference = torch.empty_like(x)
    for b in range(2):
        for p in range(5):
            for i in range(32):
                phase = p / 10000 ** (i / 32)
                u, v = float(x[b, p, 2*i]), float(x[b, p, 2*i+1])
                reference[b, p, 2*i] = u*math.cos(phase) - v*math.sin(phase)
                reference[b, p, 2*i+1] = u*math.sin(phase) + v*math.cos(phase)
    delta = (actual - reference).abs().max().item(); assert delta < 1e-6
    result = {'status': 'PASS', 'loss': float(loss.detach()), 'scalar_reference': expected,
              'positive_negative_gradient_directions': True, 'ignored_gradient_exact_zero': True,
              'ignored_value_change_loss_exact': True, 'empty_valid_exact_zero': True,
              'rope_scalar_reference_max_error': delta, 'torch': torch.__version__,
              'seconds': time.monotonic() - started, 'cuda_initialized': torch.cuda.is_initialized()}
    write_json(a.output, result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
