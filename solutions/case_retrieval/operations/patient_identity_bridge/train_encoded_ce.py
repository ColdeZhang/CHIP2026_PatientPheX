#!/usr/bin/env python3
"""Train/probe the fixed patient CE recipe on already frozen encoded arrays.

No tokenizer, candidate generator or evaluator is used. A fresh public base is
required for either arm. The probe executes exactly one ordinary first update
(warmup LR=0), reports gradients/cost, and never saves reusable model state.
"""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load_inputs(plan, root):
    path = root / plan['encoded_file']
    require(sha(path) == plan['encoded_sha256'], 'Encoded package SHA changed')
    with gzip.open(path, 'rt') as handle:
        package = json.load(handle)
    require(package['schema'] == 'patient-identity-encoded-ce-v1', 'Wrong encoded package')
    require(package['input_source_sha256'] == plan['input_source_sha256'] and
            package['labels_source_sha256'] == plan['labels_source_sha256'], 'Package provenance mismatch')
    rows = package['rows']
    require(len(rows) == 4127 and len({tuple(r['key']) for r in rows}) == 4127, 'Changed candidate keys')
    counts = np.bincount([r['label'] for r in rows], minlength=2)
    require(counts.tolist() == [3158, 969], 'Changed labels/classes')
    for row in rows:
        old = row['old_active_tokens']
        extra = row['bridge_active_tokens']
        require(0 < old <= 384 and 0 <= extra <= 128 and old + extra <= 512, 'Encoded length exceeds contract')
        require(set(row['arms']) == {'baseline', 'bridge'}, 'Wrong arms')
        for arm in ['baseline', 'bridge']:
            item = row['arms'][arm]
            require(set(item) == {'input_ids', 'token_type_ids', 'attention_mask'}, 'Unexpected encoded fields')
            active = old + (extra if arm == 'bridge' else 0)
            require(all(len(item[k]) == 512 for k in item), 'Not exactly512 tensor length')
            require(item['attention_mask'] == [1] * active + [0] * (512 - active), 'Mask no longer matches frozen lengths')
            require(item['input_ids'][old - 1] == 3 and item['input_ids'][active - 1] == 3, 'Original/final SEP moved')
            require(all(x == 0 for x in item['input_ids'][active:]), 'Padding IDs changed')
            require(all(0 <= x < 30522 for x in item['input_ids']), 'New/missing token embeddings')
            require(all(x in [0, 1] for x in item['token_type_ids']), 'Invalid segment IDs')
        for field in ['input_ids', 'token_type_ids', 'attention_mask']:
            require(row['arms']['baseline'][field][:old] == row['arms']['bridge'][field][:old], 'Original384 active prefix changed')
        require(row['arms']['bridge']['token_type_ids'][old:old + extra] == [1] * extra, 'Bridge segment changed')
        if not extra:
            require(row['arms']['baseline'] == row['arms']['bridge'], 'Empty bridge is not exact')
    indices = sorted(range(len(rows)), key=lambda i: (-sum(rows[i]['arms']['bridge']['attention_mask']), i))[:16]
    require(indices == plan['probe_row_indices'], 'Probe must use longest actual input independently of labels')
    return package, rows, counts


def collate(rows, arm):
    # Exact precomputed IDs/masks. Never call tokenizer or truncate/repad here.
    batch = {field: torch.tensor([r['arms'][arm][field] for r in rows], dtype=torch.long)
             for field in ['input_ids', 'token_type_ids', 'attention_mask']}
    batch['position_ids'] = torch.arange(512, dtype=torch.long).unsqueeze(0).expand(len(rows), -1).clone()
    batch['labels'] = torch.tensor([r['label'] for r in rows], dtype=torch.long)
    require(batch['input_ids'].shape == batch['position_ids'].shape == (len(rows), 512), 'Collator shape drift')
    return batch


def weighted_ce(logits, labels, weights):
    return torch.nn.functional.cross_entropy(logits.float(), labels, weight=weights)


def cpu_checks(plan, rows, counts):
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    expected = len(rows) / (2 * counts)
    weights = torch.tensor(expected, dtype=torch.float32)
    toy = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    y = torch.tensor([0, 1])
    loss = weighted_ce(toy, y, weights)
    loss.backward()
    expected_gradient = weights[:, None] * torch.tensor([[-.5, .5], [.5, -.5]]) / weights.sum()
    require(torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-7), 'Weighted CE scalar wrong')
    require(torch.allclose(toy.grad, expected_gradient, atol=1e-7), 'Weighted CE gradient wrong')
    ids = plan['probe_row_indices']
    packed = [rows[i] for i in ids]
    batch = collate(packed, 'bridge')
    require(batch['position_ids'].tolist() == [list(range(512))] * 16, 'Explicit positions changed')
    empty = next(row for row in rows if row['bridge_active_tokens'] == 0)
    require(all(torch.equal(collate([empty], 'baseline')[k], collate([empty], 'bridge')[k])
                for k in ['input_ids', 'token_type_ids', 'attention_mask', 'position_ids', 'labels']), 'Empty bridge differs in actual collator')
    # Supplying nonsense auxiliary text cannot affect the exact encoded input.
    noisy = [dict(row, header='ignored', evidence='ignored', bridge_text='ignored') for row in packed]
    require(all(torch.equal(batch[k], collate(noisy, 'bridge')[k]) for k in batch), 'Text fields affected encoded tensors')
    return dict(status='pass_cpu_contract', class_counts=counts.tolist(), class_weights=expected.tolist(),
                toy_loss=float(loss.detach()), toy_logits_gradient=toy.grad.tolist(),
                exact_expected_gradient=expected_gradient.tolist(), probe_indices=ids,
                probe_labels=[r['label'] for r in packed], probe_active_lengths=[sum(r['arms']['bridge']['attention_mask']) for r in packed],
                probe_shape=list(batch['input_ids'].shape), actual_forward_position_ids='explicit arange512 for both arms',
                empty_bridge_collated_arrays_exact=True, ignored_text_cannot_retokenize=True,
                cuda_initialized=torch.cuda.is_initialized(), torch_version=torch.__version__,
                transformers_version=transformers.__version__)


def verify_model(root, plan):
    model = root / plan['model_directory']
    actual = {}
    require(set(plan['model_files']) == {'pytorch_model.bin', 'config.json', 'vocab.txt', 'tokenizer_config.json', 'README.md', 'LICENSE.md'}, 'Wrong public model file set')
    for name, expected in plan['model_files'].items():
        path = model / name
        actual[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        require(actual[name] == expected, 'Public base bytes changed: ' + name)
    config = json.loads((model / 'config.json').read_text())
    require(config['architectures'] == ['BertForMaskedLM'] and config['vocab_size'] == 30522 and
            config['max_position_embeddings'] == 512 and config['num_hidden_layers'] == 12, 'Not the fixed public BiomedBERT base')
    require(not (model / 'training-stats.json').exists(), 'Trained old CE directory supplied')
    return model, actual


def fresh_gpu(plan):
    uuid = plan['gpu_uuid']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'Wrong visible GPU')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader'], text=True)
    require(not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()), 'Assigned GPU occupied; no process will be stopped')
    return dict(gpu_uuid=uuid, compute_apps=apps, server_epoch=time.time())


def one_update(model, batch, optimizer, scheduler, scaler, weights, diagnostics=False):
    batch = {key: value.cuda() for key, value in batch.items()}
    labels = batch.pop('labels')
    optimizer.zero_grad(set_to_none=True)
    lr_used = optimizer.param_groups[0]['lr']
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(**batch).logits
        loss = weighted_ce(logits, labels, weights)
    if diagnostics:
        logits.retain_grad()
    require(torch.isfinite(loss).item(), 'Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    require(torch.isfinite(norm).item(), 'Nonfinite parameter gradient')
    extra = {}
    if diagnostics:
        derivative = logits.grad.detach().float()
        own = derivative.gather(1, labels[:, None])[:, 0]
        other = derivative.gather(1, (1 - labels)[:, None])[:, 0]
        require(bool(torch.all(own < 0)) and bool(torch.all(other > 0)) and bool(torch.isfinite(derivative).all()), 'CE gradient direction is wrong')
        encoder_grads = [p.grad for name, p in model.named_parameters() if name.startswith('bert.') and p.grad is not None]
        require(encoder_grads and any(bool(torch.any(g != 0)) for g in encoder_grads), 'Encoder receives no gradient')
        require(model.classifier.weight.grad is not None and bool(torch.any(model.classifier.weight.grad != 0)), 'Classifier receives no gradient')
        extra = dict(target_logit_gradient_negative=True, other_logit_gradient_positive=True,
                     finite_encoder_and_classifier_gradient=True, gradient_norm_before_clip=float(norm),
                     logits_gradient_scale=float(scaler.get_scale()))
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return dict(loss=float(loss.detach()), learning_rate_used=lr_used,
                learning_rate_next=optimizer.param_groups[0]['lr'], **extra)


def run(plan_path, mode, output):
    started = time.monotonic()
    require(not output.exists(), 'Output already exists; recover original job, never restart/overwrite')
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    require(plan['source_sha256'] == sha(__file__), 'Trainer source differs from frozen plan')
    require(plan['schema'] == 'patient-identity-encoded-ce-plan-v1', 'Wrong execution plan')
    require(plan['runtime_versions'] == dict(torch=torch.__version__, cuda=torch.version.cuda,
                                             transformers=transformers.__version__), 'Runtime version drift')
    require(plan['recipe'] == dict(epochs=4, batch_size=16, learning_rate=2e-5, weight_decay=.01,
                                  max_grad_norm=1., warmup_steps=103, total_steps=1032, seed=20261023,
                                  tensor_length=512, loss='inverse_frequency_weighted_cross_entropy'), 'Recipe drift')
    package, rows, counts = load_inputs(plan, root)
    torch.set_num_threads(1)
    cpu = cpu_checks(plan, rows, counts)
    if mode == 'cpu':
        output.mkdir(parents=True)
        save(output / 'cpu-contract.json', dict(**cpu, source_sha256=sha(__file__), plan_sha256=sha(plan_path), elapsed_seconds=time.monotonic() - started))
        print(json.dumps(cpu, indent=2))
        return
    require(plan['execution_mode'] == mode, 'Plan does not authorize this execution mode')
    require(mode in ['probe', 'train'], 'Unknown GPU mode')
    if mode == 'probe':
        require(plan['arm'] == 'bridge' and plan['hard_budget_seconds'] == 300 and plan['max_updates'] == 1, 'Probe scope changed')
    else:
        require(plan['max_updates'] == 1032 and plan['arm'] in ['baseline', 'bridge'], 'Not a fixed four-epoch arm')
    model_path, files = verify_model(root, plan)
    gpu = fresh_gpu(plan)
    output.mkdir(parents=True)
    random.seed(20261023)
    np.random.seed(20261023)
    torch.manual_seed(20261023)
    torch.cuda.manual_seed_all(20261023)
    require(torch.cuda.device_count() == 1, 'Expected one assigned GPU')
    model, loading = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_loading_info=True)
    require(not loading['mismatched_keys'] and not loading['error_msgs'], 'Base loading failed')
    require(set(loading['missing_keys']) <= {'classifier.weight', 'classifier.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias'}, 'Unexpected missing public-base weights')
    model.cuda()
    weights = torch.tensor(len(rows) / (2 * counts), dtype=torch.float32, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 103, 1032)
    scaler = torch.amp.GradScaler('cuda')
    manifest = dict(schema=plan['schema'], mode=mode, arm=plan['arm'], source_sha256=sha(__file__),
                    plan_sha256=sha(plan_path), encoded_sha256=plan['encoded_sha256'], class_counts=counts.tolist(),
                    public_model_files=files, loading_information=loading, fresh_public_base=True,
                    old_gold_CE_or_probe_state_reused=False, parameters=sum(p.numel() for p in model.parameters()),
                    cpu_contract=cpu, gpu_before=gpu,
                    runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                                 device=torch.cuda.get_device_name(0), attention_implementation=model.config._attn_implementation),
                    recipe=plan['recipe'])
    require(manifest['parameters'] == 109483778, 'Parameter count differs from public BiomedBERT two-class model')
    save(output / 'run-manifest.json', manifest)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    if mode == 'probe':
        batch = collate([rows[i] for i in plan['probe_row_indices']], 'bridge')
        before = model.classifier.weight.detach().clone()
        torch.cuda.synchronize()
        step_start = time.monotonic()
        step = one_update(model, batch, optimizer, scheduler, scaler, weights, diagnostics=True)
        torch.cuda.synchronize()
        seconds = time.monotonic() - step_start
        unchanged = torch.equal(before, model.classifier.weight.detach())
        require(step['learning_rate_used'] == 0. and unchanged, 'Original warmup first step should not move weights')
        result = dict(status='complete_one_fresh_probe_no_model_saved', **step, actual_updates=1,
                      optimizer_states=len(optimizer.state), classifier_unchanged_expected_at_warmup0=unchanged,
                      explicit_positions512=True, actual_step_seconds=seconds,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.monotonic() - started, source_sha256=sha(__file__), plan_sha256=sha(plan_path),
                      no_checkpoint_or_optimizer_saved=True, gpu_uuid=plan['gpu_uuid'])
        save(output / 'probe-complete.json', result)
        print(json.dumps(result, indent=2))
        return
    # This branch requires its own later frozen train plan; the probe plan fails
    # above if used with --mode train. Both formal arms start fresh independently.
    arm = plan['arm']
    loader = DataLoader(rows, batch_size=16, shuffle=True, num_workers=0, collate_fn=lambda batch: collate(batch, arm))
    updates = 0
    epochs = []
    for epoch in range(4):
        epoch_start = time.monotonic()
        loss_sum = 0.
        for batch in loader:
            result = one_update(model, batch, optimizer, scheduler, scaler, weights)
            updates += 1
            loss_sum += result['loss']
        event = dict(epoch=epoch + 1, updates=updates, mean_batch_loss=loss_sum / len(loader), seconds=time.monotonic() - epoch_start)
        epochs.append(event)
        print(json.dumps(event), flush=True)
    require(updates == 1032, 'Incomplete formal four epochs')
    model.save_pretrained(output / 'model')
    save(output / 'training-complete.json', dict(status='complete_four_epochs', updates=updates, epochs=epochs,
         elapsed_seconds=time.monotonic() - started, max_allocated_bytes=torch.cuda.max_memory_allocated(),
         max_reserved_bytes=torch.cuda.max_memory_reserved(), source_sha256=sha(__file__), plan_sha256=sha(plan_path),
         no_heldout_labels_used=True, inference_or_evaluation_performed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--mode', choices=['cpu', 'probe', 'train'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.plan.resolve(), args.mode, args.output.resolve())


if __name__ == '__main__':
    main()
