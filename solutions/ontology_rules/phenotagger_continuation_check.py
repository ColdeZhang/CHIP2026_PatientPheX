#!/usr/bin/env python3
"""CPU-only check of the actual completed parent and unchanged restored state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from phenotagger_continuation import read_parent
import phenotagger_continuation
from phenotagger_measure import OntologyClassifier, sha
from phenotagger_train import tensor_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ['parent', 'spec', 'model-dir', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    spec = json.loads(Path(args.spec).read_text())
    saved, parent = read_parent(args.parent, spec['parent_checkpoint_sha256'],
                                spec['parent_trainer_sha256'], spec['fingerprint'],
                                spec['runtime'], spec['rows'])
    classes = saved['model']['classifier.weight'].shape[0]
    model = OntologyClassifier(args.model_dir, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['fingerprint']['config']['lr'], eps=1e-8)
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    model_ok = all(torch.equal(value, saved['model'][key]) for key, value in model.state_dict().items())
    actual = optimizer.state_dict()
    optimizer_ok = actual['param_groups'] == saved['optimizer']['param_groups']
    for key, state in saved['optimizer']['state'].items():
        for name, value in state.items():
            optimizer_ok &= bool(torch.equal(value, actual['state'][key][name])) if isinstance(value, torch.Tensor) else value == actual['state'][key][name]
    random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy']); torch.set_rng_state(saved['rng_torch'])
    rng_ok = random.getstate() == saved['rng_python'] and torch.equal(torch.get_rng_state(), saved['rng_torch'])
    actual_numpy = np.random.get_state(); expected_numpy = saved['rng_numpy']
    rng_ok &= actual_numpy[0] == expected_numpy[0] and np.array_equal(actual_numpy[1], expected_numpy[1]) and actual_numpy[2:] == expected_numpy[2:]
    # CUDA bytes are retained and fingerprinted here. Actual set/get equality is
    # checked by the GPU trainer before its first resumed forward pass.
    cuda_rng_sha = [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in saved['rng_cuda']]
    order = torch.randperm(spec['rows'], generator=torch.Generator().manual_seed(spec['fingerprint']['config']['seed'] + saved['epoch']))
    next_batch = order[saved['offset']:saved['offset'] + spec['fingerprint']['config']['batch_size']]
    digest = tensor_digest(model.state_dict())
    assert model_ok and optimizer_ok and rng_ok and digest == parent['parent_terminal']['model_tensor_digest']
    assert sha(Path(args.parent) / 'latest.pt') == spec['parent_checkpoint_sha256']
    report = {'status': 'pass', 'parent': parent, 'source_sha256': sha(__file__),
              'loader_sha256': sha(phenotagger_continuation.__file__), 'model_exact': model_ok,
              'optimizer_exact': optimizer_ok, 'python_numpy_torch_rng_exact': bool(rng_ok),
              'cuda_rng_retained_sha256': cuda_rng_sha, 'cuda_rng_restored_in_this_cpu_check': False,
              'initial_epoch': saved['epoch'], 'initial_step': saved['global_step'], 'initial_offset': saved['offset'],
              'next_epoch_number_1based': saved['epoch'] + 1,
              'next_batch_index_sha256': hashlib.sha256(next_batch.numpy().tobytes()).hexdigest(),
              'target_total_epochs': 100, 'remaining_epochs': 70, 'remaining_steps': parent['additional_steps'],
              'parent_file_unchanged': True, 'gpu_used': False, 'heldout_used': False}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report))


if __name__ == '__main__': main()
