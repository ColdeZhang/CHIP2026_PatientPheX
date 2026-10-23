"""Validated model-only initialization for the fixed real-positive adaptation.

This is deliberately a new optimizer/run, not continuation of parent Adam/RNG.
"""
import json
from pathlib import Path
import torch
from phenotagger_measure import sha


def read_initialization(directory, expected_sha, data_manifest, labels, fingerprint):
    parent = Path(directory)
    complete = json.loads((parent/'complete.json').read_text())
    assert complete['epochs'] == 100 and complete['global_step'] == 45800
    assert complete['final_sha256'] == expected_sha == sha(parent/'final.pt')
    saved = torch.load(parent/'final.pt', map_location='cpu', weights_only=False)
    assert saved['global_step'] == complete['global_step']
    assert saved['labels'] == labels
    previous = saved['fingerprint']
    assert previous['data_manifest_sha256'] == data_manifest['parent_data_manifest_sha256']
    assert previous['data_outputs'] == data_manifest['parent_outputs']
    assert previous['classifier_source_sha256'] == fingerprint['classifier_source_sha256']
    assert previous['base_source_files'] == fingerprint['base_source_files']
    assert previous['config']['max_length'] == fingerprint['config']['max_length'] == 32
    assert fingerprint['config'] == {'epochs': 3, 'batch_size': 256, 'lr': 5e-6,
                                     'max_length': 32, 'seed': 20261023}
    provenance = {'path': str(parent), 'final_sha256': expected_sha,
                  'complete_sha256': sha(parent/'complete.json'), 'parent_terminal': complete,
                  'parent_data_manifest_sha256': previous['data_manifest_sha256'],
                  'parent_config': previous['config'], 'parent_trainer_sha256': previous['trainer_sha256'],
                  'policy': 'All model parameters from final100; fresh Adam and seed, 3 new epochs. No parent optimizer or RNG loaded.'}
    return saved['model'], provenance
