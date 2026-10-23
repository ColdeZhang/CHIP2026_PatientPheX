#!/usr/bin/env python3
"""Measure a bounded full-encoder, all-HPO softmax training pilot.

This intentionally performs no validation inference or epoch selection.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import transformers
from transformers import AutoModel, AutoTokenizer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


class OntologyClassifier(nn.Module):
    def __init__(self, model_dir, classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, classes)

    def forward(self, ids, mask, segments):
        x = self.encoder(input_ids=ids, attention_mask=mask, token_type_ids=segments)
        return self.classifier(x.last_hidden_state[:, 0, :])


def main():
    ap = argparse.ArgumentParser()
    for name in ['data-dir', 'model-dir', 'output-dir']: ap.add_argument('--' + name, required=True)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-length', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20261023)
    a = ap.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'measurement.json').exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    data = Path(a.data_dir); manifest = json.loads((data / 'manifest.json').read_text())
    assert sha(data / 'rows.jsonl.gz') == manifest['outputs']['rows.jsonl.gz']
    assert sha(data / 'labels.json') == manifest['outputs']['labels.json']
    labels = json.loads((data / 'labels.json').read_text()); label_index = {label: i for i, label in enumerate(labels)}
    assert labels[0] == 'HP:None' and '-1' not in labels and len(label_index) == len(labels)
    with gzip.open(data / 'rows.jsonl.gz', 'rt') as f: rows = [json.loads(line) for line in f]
    assert len(rows) == manifest['rows']
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, use_fast=True, local_files_only=True)
    unpadded = tokenizer([r['text'] for r in rows], truncation=False, padding=False)
    lengths = [len(x) for x in unpadded['input_ids']]
    encoded = tokenizer([r['text'] for r in rows], truncation=True, max_length=a.max_length, padding='max_length', return_tensors='pt')
    target = torch.tensor([label_index[r['label']] for r in rows], dtype=torch.long)
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])), target)
    generator = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    assert len(loader) >= a.steps
    model = OntologyClassifier(a.model_dir, len(labels)).cuda()
    encoder_parameters = sum(p.numel() for p in model.encoder.parameters())
    head_parameters = sum(p.numel() for p in model.classifier.parameters())
    assert all(p.requires_grad for p in model.encoder.parameters())
    probe_name, probe = next((name, p) for name, p in model.encoder.named_parameters() if name.endswith('layer.0.attention.self.query.weight'))
    encoder_before = probe.detach().float().cpu().clone()
    head_before = model.classifier.weight.detach().float().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    model.train(); losses = []; step_times = []; seen_classes = set(); seen_positive = 0
    group_loss_sum = {'hpo_positive': 0., 'reject_none': 0.}
    group_count = {'hpo_positive': 0, 'reject_none': 0}
    group_correct = {'hpo_positive': 0, 'reject_none': 0}
    first_encoder_gradient = None; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(loader):
        if step >= a.steps: break
        ids, mask, segments, y = [v.cuda(non_blocking=True) for v in batch]
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); begin = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ids, mask, segments)
            per_row_loss = nn.functional.cross_entropy(logits.float(), y, reduction='none')
            loss = per_row_loss.mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        if step == 0:
            assert probe.grad is not None and bool(torch.isfinite(probe.grad).all())
            first_encoder_gradient = float(probe.grad.float().norm().item())
            assert first_encoder_gradient > 0
        optimizer.step(); torch.cuda.synchronize()
        step_times.append(time.monotonic() - begin); losses.append(float(loss.item()))
        for group, selected in [('hpo_positive', y != 0), ('reject_none', y == 0)]:
            group_count[group] += int(selected.sum().item())
            group_loss_sum[group] += float(per_row_loss.detach()[selected].sum().item())
            group_correct[group] += int(((logits.detach().argmax(-1) == y) & selected).sum().item())
        seen_classes.update(int(v) for v in y.cpu().tolist()); seen_positive += int((y != 0).sum().item())
        if (step + 1) % 10 == 0:
            print(json.dumps({'event': 'training_step', 'step': step + 1, 'loss': losses[-1]}), flush=True)
    encoder_delta = float((probe.detach().float().cpu() - encoder_before).norm().item())
    classifier_delta = float((model.classifier.weight.detach().float().cpu() - head_before).norm().item())
    train_elapsed = time.monotonic() - started
    assert encoder_delta > 0 and classifier_delta > 0
    steady = step_times[10:] or step_times
    mean_step = sum(steady) / len(steady); epoch_steps = math.ceil(len(rows) / a.batch_size)
    report = {'config': vars(a), 'source_sha256': sha(__file__), 'data_manifest_sha256': sha(data / 'manifest.json'),
        'base_source_files': {p.name: sha(p) for p in Path(a.model_dir).iterdir() if p.is_file()},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                    'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'cpu_threads': 1},
        'architecture': '[CLS] last hidden -> linear(all HPO + reject) softmax; all encoder weights trainable; no extra dropout',
        'encoder_parameters': encoder_parameters, 'classifier_parameters': head_parameters,
        'optimizer': {'type': 'torch.optim.Adam', 'lr': 5e-6, 'weight_decay': 0., 'epsilon': 1e-8},
        'token_audit': {'rows': len(rows), 'max_tokens_with_special': max(lengths), 'truncated_rows': sum(n > a.max_length for n in lengths),
                        'truncation_policy': 'Explicit original-recipe max32 for cost measurement; no performance claim.'},
        'measured_steps': len(losses), 'examples_seen': min(a.steps * a.batch_size, len(rows)),
        'positive_examples_seen': seen_positive, 'unique_classes_seen_in_pilot': len(seen_classes),
        'loss_first': losses[0], 'loss_last': losses[-1], 'loss_mean': sum(losses) / len(losses),
        'online_training_groups': {group: {'examples': group_count[group],
            'mean_loss': group_loss_sum[group] / max(1, group_count[group]),
            'accuracy': group_correct[group] / max(1, group_count[group])} for group in group_count},
        'all_losses_finite': all(math.isfinite(x) for x in losses), 'encoder_probe': probe_name,
        'encoder_gradient_norm_first_step': first_encoder_gradient, 'encoder_parameter_delta_l2': encoder_delta,
        'classifier_parameter_delta_l2': classifier_delta, 'train_elapsed_seconds': train_elapsed,
        'steady_step_seconds_mean': mean_step, 'steps_per_epoch': epoch_steps,
        'estimated_epoch_seconds_from_training_only': epoch_steps * mean_step,
        'estimated_30epoch_minutes_from_training_only': epoch_steps * mean_step * 30 / 60,
        'estimated_100epoch_minutes_from_training_only': epoch_steps * mean_step * 100 / 60,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'heldout_annotations_or_inference_used': False, 'model_checkpoint_saved': False,
        'status': 'cost and gradient smoke only; not an accuracy evaluation or trained model'}
    (out / 'measurement.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
