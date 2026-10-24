"""Fixed document-relation training, final checkpoint, and complete held logits."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModel

from model import WindowDocRE


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoded', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['probe', 'formal'], required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261024)
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    train = load(Path(args.encoded) / 'train.json.gz')
    held = load(Path(args.encoded) / 'held.json.gz')
    assert len(train) == 32 and len(held) == 16
    assert {x['pmc_id'] for x in train}.isdisjoint({x['pmc_id'] for x in held})
    assert all('labels' in x and len(x['labels']) == len(x['pairs']) for x in train)
    assert all('labels' not in x for x in held)
    assets = {}
    for path in sorted(Path(args.model).iterdir()):
        if path.is_file() and path.suffix in ['.json', '.txt', '.bin', '.safetensors']:
            assets[path.name] = reference(path)
    encoder, loading = AutoModel.from_pretrained(args.model, local_files_only=True,
                        attn_implementation='eager', output_loading_info=True)
    # A published pretraining head may be unused; missing encoder tensors are not.
    assert not loading.get('missing_keys') and not loading.get('mismatched_keys')
    model = WindowDocRE(encoder).cuda()
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count < 10_000_000_000
    manifest = {'configuration': vars(args), 'assets': assets, 'parameters': parameter_count,
        'loading': loading, 'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers']},
        'encoded': {split: reference(Path(args.encoded) / f'{split}.json.gz') for split in ['train', 'held']},
        'sources': {name: reference(Path(__file__).parent / name) for name in ['train.py', 'model.py', 'encode.py']},
        'precision': 'FP32 parameters/Adam; BF16 autocast; shared pooling/loss FP32',
        'checkpointing': 'non-reentrant full-window recomputation; RNG preserved',
        'declared_loss': 'unweighted BCE per pair, mean per document, actual accumulation group mean',
        'gradient_checkpointing_encoder_internal': False}
    # AutoModel loading metadata consists of lists/dicts on this pinned runtime.
    write(output / 'manifest.json', manifest)
    backbone = list(model.encoder.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 2e-5}, {'params': head, 'lr': 1e-4}], weight_decay=0.01)
    model.train()
    if args.mode == 'probe':
        document = max(train, key=lambda x: (len(x['windows']), x['token_length']))
        before = {name: p.detach().clone() for name, p in model.named_parameters()
                  if name in ['encoder.embeddings.word_embeddings.weight', 'head.weight', 'classifier.weight']}
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, diagnostic = model(document)
            labels = torch.tensor(document['labels'], dtype=torch.float32, device='cuda')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
        loss.backward()
        gradients = {name: {'finite': bool(torch.isfinite(p.grad).all()), 'norm': float(p.grad.float().norm())}
                     for name, p in model.named_parameters() if p.grad is not None}
        assert gradients and all(row['finite'] for row in gradients.values())
        assert all(any(row['norm'] > 0 for name, row in gradients.items() if name.startswith(prefix))
                   for prefix in ['encoder.', 'head.', 'tail.', 'classifier.'])
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        torch.cuda.synchronize()
        deltas = {name: float((dict(model.named_parameters())[name].detach() - value).abs().max()) for name, value in before.items()}
        assert all(value > 0 for value in deltas.values())
        first_update_seconds = time.monotonic() - begin
        del before
        # Match the formal memory state: an existing Adam state and an earlier
        # document's accumulated gradients when the largest document runs.
        optimizer.zero_grad(set_to_none=True)
        stress = []
        shortest = min(train, key=lambda x: (len(x['windows']), x['token_length']))
        for index, item in enumerate([shortest, document]):
            step_started = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                stress_logits, stress_diagnostic = model(item)
                stress_labels = torch.tensor(item['labels'], dtype=torch.float32, device='cuda')
                stress_loss = torch.nn.functional.binary_cross_entropy_with_logits(stress_logits.float(), stress_labels)
            (stress_loss / 4).backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            torch.cuda.synchronize()
            stress.append({'pmc_id': item['pmc_id'], 'windows': len(item['windows']),
                           'loss': float(stress_loss.detach()), 'seconds': time.monotonic() - step_started,
                           'existing_Adam': True, 'previous_document_gradients': index > 0,
                           'diagnostic': stress_diagnostic})
        result = {'status': 'complete_engineering_probe_no_quality_no_checkpoint', 'document': document['pmc_id'],
            'tokens': document['token_length'], 'windows': len(document['windows']), 'pairs': len(document['pairs']),
            'loss': float(loss), 'gradient_norm_before_clip': norm, 'gradient_tensors': len(gradients),
            'all_gradients_finite': True, 'actual_parameter_changes': deltas, 'diagnostic': diagnostic,
            'actual_forward_backward_calls': 3, 'actual_Adam_steps': 1, 'stress_with_existing_state': stress,
            'first_longest_update_seconds': first_update_seconds,
            'three_pass_training_seconds': time.monotonic() - begin, 'elapsed_seconds': time.monotonic() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        write(output / 'probe.json', result)
        print(json.dumps(result), flush=True)
        return
    total_steps = math.ceil(len(train) / 4) * args.epochs
    warmup = max(1, math.ceil(total_steps * .1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0., (total_steps - step) / (total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    global_step = 0
    epochs = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        begin = time.monotonic()
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        losses, context_zero, pairs_seen = [], 0, 0
        for first in range(0, len(order), 4):
            group = order[first:first + 4]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                document = train[index]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, diagnostic = model(document)
                    labels = torch.tensor(document['labels'], dtype=torch.float32, device='cuda')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
                if not bool(torch.isfinite(loss)):
                    write(output / 'nonfinite.json', {'epoch': epoch + 1, 'document': document['pmc_id'], 'stage': 'loss'})
                    raise ValueError('Non-finite loss')
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
                context_zero += diagnostic['zero_context_pairs']
                pairs_seen += diagnostic['pairs']
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert bool(torch.isfinite(norm)), 'Non-finite gradient norm'
            optimizer.step()
            scheduler.step()
            global_step += 1
        torch.cuda.synchronize()
        record = {'epoch': epoch + 1, 'global_step': global_step, 'documents': len(order),
                  'document_order_sha256': hashlib.sha256(json.dumps([train[i]['pmc_id'] for i in order]).encode()).hexdigest(),
                  'mean_document_loss': sum(losses) / len(losses), 'seconds': time.monotonic() - begin,
                  'zero_context_pairs': context_zero, 'pairs_seen': pairs_seen,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        epochs.append(record)
        write(output / 'progress.json', record)
        print(json.dumps(record), flush=True)
    assert global_step == total_steps == math.ceil(len(train) / 4) * args.epochs
    checkpoint_path = output / 'checkpoint.pt'
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': args.epochs, 'global_step': global_step, 'python_rng': random.getstate(),
             'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
             'configuration': vars(args)}
    with checkpoint_path.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert loaded['global_step'] == total_steps and loaded['epoch'] == args.epochs
    for name, value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(), loaded['model'][name]), name
    checkpoint_ref = reference(checkpoint_path)
    write(output / 'checkpoint-receipt.json', {'status': 'saved_fsync_reopened_full_model_exact',
          'checkpoint': checkpoint_ref, 'model_tensors': len(loaded['model']),
          'optimizer_tensors': sum(len(s) for s in loaded['optimizer']['state'].values()),
          'full_state_present': all(k in loaded for k in ['optimizer', 'scheduler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'])})
    del loaded, state, optimizer, scheduler
    torch.cuda.empty_cache()
    model.eval()
    scores_path = output / 'held-scores.jsonl'
    inference = []
    with scores_path.open('w') as stream:
        for document in held:
            begin = time.monotonic()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits, diagnostic = model(document)
            values = logits.float().cpu().tolist()
            assert len(values) == len(document['keys']) and all(math.isfinite(v) for v in values)
            for key, value in zip(document['keys'], values):
                probability = 1 / (1 + math.exp(-max(-80., min(80., value))))
                stream.write(json.dumps({'key': key, 'logit': value, 'probability': probability}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            inference.append({'pmc_id': document['pmc_id'], 'pairs': len(values), **diagnostic,
                              'seconds': time.monotonic() - begin})
    result = {'status': 'complete_formal_and_all_held_logits_no_quality', 'epochs': epochs,
              'global_step': global_step, 'checkpoint': checkpoint_ref, 'inference': inference,
              'scores': reference(scores_path), 'elapsed_seconds': time.monotonic() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write(output / 'complete.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'], 'scores': result['scores']}), flush=True)


if __name__ == '__main__':
    main()
