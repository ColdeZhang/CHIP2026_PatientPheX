"""Read an immutable completed training state for a declared larger budget.

This module changes no model/Adam/RNG value and performs no GPU operation.
"""
import json
import math
from pathlib import Path
import torch
from phenotagger_measure import sha


def read_parent(directory, expected_checkpoint_sha, expected_trainer_sha,
                current_fingerprint, runtime, row_count):
    directory = Path(directory)
    complete = json.loads((directory / 'complete.json').read_text())
    assert complete['epochs'] == 30 and complete['heldout_used'] is False
    assert expected_checkpoint_sha == complete['latest_sha256'] == sha(directory / 'latest.pt')
    assert complete['final_sha256'] == sha(directory / 'final.pt')
    saved = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    old = saved['fingerprint']; new = current_fingerprint
    assert old['trainer_sha256'] == expected_trainer_sha
    for key in ['data_manifest_sha256', 'data_outputs', 'classifier_source_sha256', 'base_source_files']:
        assert old[key] == new[key], key
    before = dict(old['config']); after = dict(new['config'])
    previous_epochs = before.pop('epochs'); target_epochs = after.pop('epochs')
    assert before == after, 'Only the total epoch budget may change'
    assert previous_epochs == 30 and target_epochs == 100
    assert saved['runtime'] == runtime
    assert saved['epoch'] == previous_epochs and saved['offset'] == 0
    expected_steps = math.ceil(row_count / new['config']['batch_size']) * previous_epochs
    assert saved['global_step'] == complete['global_step'] == expected_steps
    assert len(saved['completed_epochs']) == previous_epochs
    assert [row['epoch'] for row in saved['completed_epochs']] == list(range(1, previous_epochs + 1))
    assert all(row['count'] == 0 for row in saved['epoch_stats'].values())
    assert saved['optimizer']['state']
    assert all(group['lr'] == before['lr'] and group['eps'] == 1e-8
               and group['weight_decay'] == 0. for group in saved['optimizer']['param_groups'])
    assert all(int(state['step']) == expected_steps for state in saved['optimizer']['state'].values())
    assert len(saved['rng_cuda']) == 1
    metadata = {'parent_directory': str(directory), 'checkpoint_sha256': expected_checkpoint_sha,
                'complete_sha256': sha(directory / 'complete.json'),
                'parent_fingerprint': old, 'parent_terminal': complete,
                'initial_epoch': saved['epoch'], 'initial_offset': saved['offset'],
                'initial_global_step': saved['global_step'],
                'initial_training_seconds': saved['training_seconds'],
                'additional_epochs': target_epochs - previous_epochs,
                'additional_steps': math.ceil(row_count / new['config']['batch_size']) * (target_epochs - previous_epochs),
                'state_policy': 'Exact saved model/Adam/RNG; only total budget changes; parent files read-only'}
    return saved, metadata
