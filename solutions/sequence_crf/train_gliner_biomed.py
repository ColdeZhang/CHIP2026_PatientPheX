#!/usr/bin/env python3
"""One frozen supervised span control; no validation labels or checkpoint selection."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

import check_gliner_biomed
from check_gliner_biomed import load
from run_gliner_pilot import sha


def masked_span_bce(logits, labels, span_mask):
    """Explicit binary targets; bypass the upstream positional loss-call bug."""
    import torch
    import torch.nn.functional as F
    batch = logits.shape[0]
    assert logits.shape[-1] == 1
    scores = logits.reshape(batch, -1)
    targets = labels.reshape(batch, -1).to(scores.dtype)
    valid = span_mask.reshape(batch, -1).bool()
    assert scores.shape == targets.shape == valid.shape and valid.any()
    assert ((targets == 0) | (targets == 1)).all()
    return F.binary_cross_entropy_with_logits(scores[valid], targets[valid], reduction='mean')


def training_counts(logits, labels, span_mask):
    valid = span_mask.reshape(logits.shape[0], -1).bool()
    scores = logits.detach().reshape(logits.shape[0], -1)
    probability = scores.sigmoid()
    predicted = (scores > 0) & valid
    positive = (labels.reshape(logits.shape[0], -1) > 0) & valid
    return {'gold_positive': int(positive.sum()), 'predicted_positive': int(predicted.sum()),
            'gold_negative': int((valid & ~positive).sum()),
            'tp': int((positive & predicted).sum()), 'fp': int((~positive & predicted).sum()),
            'positive_probability_sum': float(probability[positive].sum()),
            'negative_probability_sum': float(probability[valid & ~positive].sum())}


def count_metrics(counts):
    return {'precision': counts['tp']/max(1, counts['predicted_positive']),
            'recall': counts['tp']/max(1, counts['gold_positive']),
            'positive_mean_probability': counts['positive_probability_sum']/max(1, counts['gold_positive']),
            'negative_mean_probability': counts['negative_probability_sum']/max(1, counts['gold_negative'])}


def collate(processor, rows):
    raw = processor.collate_raw_batch(rows, entity_types=[['phenotype'] for _ in rows])
    tensors = processor.collate_fn(raw, prepare_labels=True)
    tensors.update(span_idx=raw['span_idx'], span_mask=raw['span_mask'], text_lengths=raw['seq_length'])
    labels = tensors['labels'].reshape(len(rows), -1)
    for i, row in enumerate(rows):
        assert int(tensors['attention_mask'][i].sum()) == row['tokens'] <= 384
        assert int(tensors['words_mask'][i].max()) == len(row['tokenized_text'])
        assert int(tensors['text_lengths'][i]) == len(row['tokenized_text'])
        expected = {a*processor.config.max_width+b-a for a, b, _ in row['ner']}
        actual = set((labels[i] > 0).nonzero().flatten().tolist())
        assert actual == expected
        assert all(float(labels[i, index]) == 0 for index in row['no_span_indices'])
    return {k: v for k, v in tensors.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data-manifest', 'train', 'excluded-input', 'model-dir', 'tokenizer-dir', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, choices=[2, 10], default=2)
    parser.add_argument('--training-part', choices=['nested_train32', 'train48', 'train64'], default='nested_train32')
    parser.add_argument('--prune-old-epochs', action='store_true',
                        help='Keep only latest model/state after atomic checkpoint save; record old hashes.')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    seed, epochs, batch_size = 20261023, args.epochs, 4
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = json.loads(Path(args.data_manifest).read_text())
    expected = data['artifacts'][args.training_part]
    assert sha(args.train) == expected['sha256']
    rows = [json.loads(x) for x in Path(args.train).read_text().splitlines()]
    ids = {r['pmc_id'] for r in rows}
    excluded = [json.loads(x) for x in Path(args.excluded_input).read_text().splitlines()]
    assert all('entities' not in d and 'association' not in d for d in excluded)
    assert ids == set(expected['pmc_ids']) and len(ids) == {'nested_train32': 32, 'train48': 48, 'train64': 64}[args.training_part]
    assert not ids & {str(d['pmc_id']) for d in excluded}
    if args.training_part == 'nested_train32':
        assert {str(d['pmc_id']) for d in excluded} == set(data['artifacts']['nested_validation16']['pmc_ids'])
    else:
        assert sha(args.excluded_input) == data['files']['excluded_input']
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configuration = {'epochs': epochs, 'batch_size': batch_size, 'encoder_lr': 1e-5,
                     'other_lr': 5e-5, 'weight_decay': .01, 'warmup_fraction': .05,
                     'seed': seed, 'precision': 'float32', 'gradient_clip': 1.,
                     'loss': 'binary CE sum over valid spans / valid span count',
                     'label': 'phenotype', 'all_valid_negative_spans': True,
                     'checkpoint_choice': f'epoch{epochs} only; no validation selection',
                     'training_part': args.training_part}
    fingerprint = {'configuration': configuration, 'source_sha256': sha(__file__),
                   'loader_source_sha256': sha(check_gliner_biomed.__file__),
                   'data_manifest_sha256': sha(args.data_manifest), 'train_sha256': sha(args.train),
                   'excluded_input_sha256': sha(args.excluded_input),
                   'base_weight_sha256': sha(Path(args.model_dir)/'pytorch_model.bin')}
    fingerprint_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    record_path = out/'run_manifest.json'
    previous_results = []
    if record_path.exists():
        assert args.resume and not args.smoke_only
        previous = json.loads(record_path.read_text())
        assert previous['fingerprint_id'] == fingerprint_id
        previous_results = previous['epoch_results']
    else:
        assert not args.resume
    assert torch.cuda.is_available()
    model = load(Path(args.model_dir), Path(args.tokenizer_dir)).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    grouped = {}
    names = {}
    for name, parameter in model.model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = ('encoder' if 'token_rep_layer' in name else 'other',
               'nodecay' if parameter.ndim == 1 or name.endswith('bias') else 'decay')
        grouped.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    optimizer = torch.optim.AdamW([{'params': ps, 'lr': 1e-5 if key[0] == 'encoder' else 5e-5,
                                   'weight_decay': .01 if key[1] == 'decay' else 0.}
                                  for key, ps in sorted(grouped.items())])
    steps_per_epoch = math.ceil(len(rows)/batch_size)
    total_steps = epochs*steps_per_epoch
    warmup = max(1, math.ceil(total_steps*.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step+1)/warmup, max(0., (total_steps-step)/(total_steps-warmup))))
    start_epoch, global_step = 0, 0
    if args.resume:
        state = torch.load(out/'latest-state.pt', map_location='cpu', weights_only=False)
        assert state['fingerprint_id'] == fingerprint_id
        model.model.load_state_dict(torch.load(out/state['model_file'], map_location='cpu', weights_only=True), strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch, global_step = state['epoch'], state['step']
    report = {'fingerprint': fingerprint, 'fingerprint_id': fingerprint_id,
              'arguments': vars(args), 'documents': len(ids), 'rows': len(rows),
              'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps,
              'parameters': sum(p.numel() for p in model.parameters()),
              'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'gliner']},
              'optimizer_parameter_names': {':'.join(k): v for k, v in names.items()},
              'gpu': torch.cuda.get_device_name(0), 'free_bytes_before': shutil.disk_usage(out).free,
              'gold_validation_read': False, 'completed_epochs': start_epoch, 'epoch_results': previous_results,
              'upstream_loss_bypassed': True,
              'loss_scale_note': 'Default upstream sum replaced by mean over valid spans only; no smoothing or focal weighting.'}
    record_path.write_text(json.dumps(report, indent=2)+'\n')
    started = time.monotonic()
    model.train()
    for epoch in range(start_epoch, epochs):
        order = list(range(len(rows)))
        random.Random(seed+epoch).shuffle(order)
        if args.smoke_only:
            order = sorted(order, key=lambda i: (-rows[i]['tokens'], -len(rows[i]['tokenized_text']), rows[i]['row_id']))[:batch_size]
        losses, epoch_start, running_counts = [], time.monotonic(), Counter()
        for index in range(0, len(order), batch_size):
            batch = [rows[i] for i in order[index:index+batch_size]]
            tensors = {k: v.to('cuda') for k, v in collate(model.data_processor, batch).items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model.model(**{k: v for k, v in tensors.items() if k != 'labels'})
            assert outputs.loss is None
            loss = masked_span_bce(outputs.logits, tensors['labels'], tensors['span_mask'])
            running_counts.update(training_counts(outputs.logits, tensors['labels'], tensors['span_mask']))
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm), 'Nonfinite gradient'
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach()))
            if global_step % 50 == 0 or args.smoke_only:
                print(json.dumps({'epoch': epoch+1, 'step': global_step, 'loss': losses[-1]}), flush=True)
        if args.smoke_only:
            report.update(smoke_completed=True, smoke_rows=[rows[i]['row_id'] for i in order],
                          smoke_loss=losses[0], smoke_gradient_norm=float(norm),
                          smoke_counts=dict(running_counts),
                          smoke_metrics=count_metrics(running_counts),
                          elapsed_seconds=time.monotonic()-started,
                          max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            record_path.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'smoke_complete': True, 'seconds': report['elapsed_seconds'],
                              'max_gpu_allocated_bytes': report['max_gpu_allocated_bytes']}), flush=True)
            return
        assert shutil.disk_usage(out).free > 4*1024**3, 'Insufficient space for atomic checkpoint and optimizer'
        # A fixed training-only positive-window probe detects collapse. It does
        # not alter the optimizer, decoder, epoch choice or heldout predictions.
        first_positive = {}
        for row in rows:
            if row['ner']:
                first_positive.setdefault(row['pmc_id'], row)
        probe = [first_positive[key] for key in sorted(first_positive)]
        probe_counts = Counter()
        model.eval()
        with torch.no_grad():
            for i in range(0, len(probe), batch_size):
                tensors = {k: v.to('cuda') for k, v in collate(model.data_processor, probe[i:i+batch_size]).items()}
                logits = model.model(**{k: v for k, v in tensors.items() if k != 'labels'}).logits
                probe_counts.update(training_counts(logits, tensors['labels'], tensors['span_mask']))
        model.train()
        model_file = f'epoch-{epoch+1:03d}/pytorch_model.bin'
        (out/model_file).parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(args.model_dir)/'gliner_config.json', (out/model_file).parent/'gliner_config.json')
        torch.save({k: v.detach().cpu() for k, v in model.model.state_dict().items()}, out/(model_file+'.tmp'))
        (out/(model_file+'.tmp')).replace(out/model_file)
        state = {'fingerprint_id': fingerprint_id, 'model_file': model_file,
                 'epoch': epoch+1, 'step': global_step, 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(state, out/'latest-state.pt.tmp')
        (out/'latest-state.pt.tmp').replace(out/'latest-state.pt')
        result = {'epoch': epoch+1, 'step': global_step, 'mean_loss': sum(losses)/len(losses),
                  'seconds': time.monotonic()-epoch_start, 'model_file': model_file,
                  'training_online_counts': dict(running_counts),
                  'training_probe_rows': [r['row_id'] for r in probe], 'training_probe_counts': dict(probe_counts),
                  'training_probe_metrics': count_metrics(probe_counts),
                  'model_sha256': sha(out/model_file), 'state_sha256': sha(out/'latest-state.pt')}
        report['epoch_results'].append(result)
        report.update(completed_epochs=epoch+1, elapsed_seconds=time.monotonic()-started,
                      max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      free_bytes_after=shutil.disk_usage(out).free)
        record_path.write_text(json.dumps(report, indent=2)+'\n')
        if args.prune_old_epochs:
            # The current state already refers to this epoch's complete model.
            # Do not delete unknown files or silently lose the prior identity.
            for prior in report['epoch_results'][:-1]:
                old_model = out/prior['model_file']
                if old_model.exists():
                    assert old_model != out/model_file
                    assert sha(old_model) == prior['model_sha256']
                    old_model.unlink()
                    prior['model_pruned_after_epoch'] = epoch+1
            record_path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
