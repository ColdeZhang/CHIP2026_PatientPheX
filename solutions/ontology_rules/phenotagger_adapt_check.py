#!/usr/bin/env python3
"""CPU token and parent-weight checks for the fixed three-epoch adaptation."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from phenotagger_measure import OntologyClassifier, sha
import phenotagger_measure
from phenotagger_adaptation import read_initialization
from phenotagger_train import tensor_digest


def main():
    parser = argparse.ArgumentParser()
    for key in ['data-dir', 'model-dir', 'parent-dir', 'parent-final-sha', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available(), 'This check must run with CUDA_VISIBLE_DEVICES empty'
    data = Path(args.data_dir); manifest = json.loads((data/'manifest.json').read_text())
    for name, digest in manifest['outputs'].items(): assert sha(data/name) == digest
    labels = json.loads((data/'labels.json').read_text())
    with gzip.open(data/'rows.jsonl.gz', 'rt') as stream: rows = [json.loads(s) for s in stream]
    assert len(rows) == manifest['rows'] == 145102
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    texts = sorted({r['text'] for r in rows})
    full = tokenizer(texts, padding=False, truncation=False)['input_ids']
    truncated = tokenizer(texts, padding=False, truncation=True, max_length=32)['input_ids']
    lengths = {text: len(ids) for text, ids in zip(texts, full)}
    tokens = {text: tuple(ids) for text, ids in zip(texts, truncated)}
    stats = defaultdict(Counter); unique = defaultdict(set); unique_long = defaultdict(set)
    collision_labels = defaultdict(set); collision_sources = defaultdict(set); collision_texts = defaultdict(set)
    source_labels = defaultdict(lambda: defaultdict(set))
    for row in rows:
        source = row['source']; text = row['text']; length = lengths[text]
        stats[source]['rows'] += 1; stats[source]['overlength_rows'] += length > 32
        stats[source]['max_tokens'] = max(stats[source]['max_tokens'], length)
        unique[source].add(text)
        if length > 32: unique_long[source].add(text)
        collision_labels[tokens[text]].add(row['label'])
        collision_sources[tokens[text]].add(source); collision_texts[tokens[text]].add(text)
        source_labels[tokens[text]][source].add(row['label'])
    collisions = Counter()
    for ids, targets in collision_labels.items():
        if len(targets) < 2: continue
        collisions['multi_label_token_sequences'] += 1
        collisions['with_none'] += 'HP:None' in targets
        collisions['with_real_positive'] += 'train_real_positive' in collision_sources[ids]
        collisions['cross_source'] += len(collision_sources[ids]) > 1
        collisions['distinct_texts_mapped_together'] += len(collision_texts[ids]) > 1
        sources = sorted(source_labels[ids])
        for i, source in enumerate(sources):
            for other in sources[i:]:
                if any(a != b for a in source_labels[ids][source] for b in source_labels[ids][other]):
                    collisions['pair:' + source + '|' + other] += 1
    fingerprint = {'config': {'epochs': 3, 'batch_size': 256, 'lr': 5e-6,
                              'max_length': 32, 'seed': 20261023},
                   'classifier_source_sha256': sha(phenotagger_measure.__file__),
                   'base_source_files': {p.name: sha(p) for p in Path(args.model_dir).iterdir() if p.is_file()}}
    weights, provenance = read_initialization(args.parent_dir, args.parent_final_sha,
                                               manifest, labels, fingerprint)
    model = OntologyClassifier(args.model_dir, len(labels)); model.load_state_dict(weights)
    actual = tensor_digest(model.state_dict()); assert actual == provenance['parent_terminal']['model_tensor_digest']
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, eps=1e-8)
    assert not optimizer.state_dict()['state'] and all(p.requires_grad for p in model.parameters())
    result = {'source_sha256': sha(__file__), 'data_manifest_sha256': sha(data/'manifest.json'),
        'rows': len(rows), 'unique_texts': len(texts), 'classes': len(labels),
        'tokenizer_files_sha256': {p.name: sha(p) for p in Path(args.model_dir).iterdir()
                                  if p.is_file() and p.suffix != '.bin'},
        'token_audit': {k: dict(v, unique_texts=len(unique[k]), overlength_unique=len(unique_long[k]))
                        for k, v in stats.items()},
        'token_sequence_label_conflicts': dict(collisions),
        'truncation_policy': 'All 145102 rows retained; same max32 truncation as parent; no rewritten text or label-based selection.',
        'initialization': {'parent': provenance, 'actual_model_tensor_digest': actual,
                           'all_parameters_loaded_exact': True, 'all_parameters_trainable': True,
                           'new_optimizer_state_entries': 0, 'new_global_step': 0,
                           'parent_optimizer_and_rng_restored': False},
        'gpu_used': False, 'heldout_used': False}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'rows': len(rows), 'token_audit': result['token_audit'],
                      'conflicts': dict(collisions), 'model_exact': True}), flush=True)


if __name__ == '__main__': main()
