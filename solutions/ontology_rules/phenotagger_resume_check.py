#!/usr/bin/env python3
"""Compare completed uninterrupted vs interrupted/resumed training state on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def equal(left, right):
    if isinstance(left, torch.Tensor): return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if hasattr(left, 'shape'): return bool((left == right).all())
    return left == right


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', required=True); parser.add_argument('--resumed', required=True)
    parser.add_argument('--expected-step', required=True, type=int); parser.add_argument('--output', required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    left = torch.load(args.direct, map_location='cpu', weights_only=False)
    right = torch.load(args.resumed, map_location='cpu', weights_only=False)
    assert left['global_step'] == right['global_step'] == args.expected_step
    keys = ['fingerprint', 'runtime', 'model', 'optimizer', 'epoch', 'offset', 'global_step',
            'epoch_stats', 'completed_epochs', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda']
    checks = {key: bool(equal(left[key], right[key])) for key in keys}
    result = {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
              'global_step': left['global_step'], 'offset': left['offset'],
              'direct_checkpoint_sha256': sha(args.direct), 'resumed_checkpoint_sha256': sha(args.resumed),
              'comparison_source_sha256': sha(__file__), 'training_source': left['fingerprint'],
              'note': 'Wall-clock timing and serialization container bytes need not match; all model/Adam/RNG tensors must.'}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result)); assert result['status'] == 'pass'


if __name__ == '__main__': main()
