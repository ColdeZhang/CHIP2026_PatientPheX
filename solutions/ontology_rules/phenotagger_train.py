#!/usr/bin/env python3
"""Fixed-budget full-encoder HPO classifier with atomic, exact-state resume.

No held-out data is accepted by this entry point. Checkpoints contain only
training state. --stop-after-step is a controlled interruption for resume tests.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from phenotagger_measure import OntologyClassifier, sha
import phenotagger_measure
from phenotagger_continuation import read_parent
import phenotagger_continuation
from phenotagger_adaptation import read_initialization
import phenotagger_adaptation


def atomic_save(value, path):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(value, path):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def new_stats():
    return {k: {'count': 0, 'loss_sum': 0., 'correct': 0, 'predicted_none': 0}
            for k in ['hpo_positive', 'reject_none']}


def summarize(stats):
    return {k: dict(v, mean_loss=v['loss_sum'] / max(1, v['count']),
                    accuracy=v['correct'] / max(1, v['count'])) for k, v in stats.items()}


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ['data-dir', 'model-dir', 'output-dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--save-every-seconds', type=float, default=240.)
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue-from')
    parser.add_argument('--parent-checkpoint-sha')
    parser.add_argument('--parent-trainer-sha')
    parser.add_argument('--initialize-from')
    parser.add_argument('--initial-final-sha')
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not (args.resume and args.continue_from)
    assert sum(bool(v) for v in [args.resume, args.continue_from, args.initialize_from]) <= 1
    if args.initialize_from:
        assert out.resolve() != Path(args.initialize_from).resolve() and args.initial_final_sha
    if args.continue_from:
        assert out.resolve() != Path(args.continue_from).resolve()
        assert args.parent_checkpoint_sha and args.parent_trainer_sha
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    data = Path(args.data_dir)
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ['rows.jsonl.gz', 'labels.json']:
        assert sha(data / name) == manifest['outputs'][name]
    labels = json.loads((data / 'labels.json').read_text())
    assert labels[0] == 'HP:None' and len(set(labels)) == len(labels) and '-1' not in labels
    with gzip.open(data / 'rows.jsonl.gz', 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == manifest['rows']
    index = {name: i for i, name in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    encoded = tokenizer([row['text'] for row in rows], truncation=True,
                        max_length=args.max_length, padding='max_length', return_tensors='pt')
    tensors = [encoded['input_ids'], encoded['attention_mask'],
               encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])),
               torch.tensor([index[row['label']] for row in rows], dtype=torch.long)]
    real_rows = torch.tensor([row['source'] == 'train_real_positive' for row in rows], dtype=torch.bool)
    configuration = {key: getattr(args, key) for key in
                     ['epochs', 'batch_size', 'lr', 'max_length', 'seed']}
    fingerprint = {'config': configuration, 'data_manifest_sha256': sha(data / 'manifest.json'),
                   'data_outputs': manifest['outputs'], 'trainer_sha256': sha(__file__),
                   'continuation_loader_sha256': sha(phenotagger_continuation.__file__),
                   'adaptation_loader_sha256': sha(phenotagger_adaptation.__file__),
                   'classifier_source_sha256': sha(phenotagger_measure.__file__),
                   'base_source_files': {p.name: sha(p) for p in Path(args.model_dir).iterdir() if p.is_file()}}
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'transformers': transformers.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'cpu_threads': 1,
               'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
               'deterministic_algorithms': True}
    assert runtime['CUBLAS_WORKSPACE_CONFIG'] in [':4096:8', ':16:8']
    model = OntologyClassifier(args.model_dir, len(labels)).cuda()
    assert all(p.requires_grad for p in model.encoder.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    epoch = offset = global_step = 0
    epoch_stats = new_stats(); completed_epochs = []; training_seconds = 0.
    checkpoint = out / 'latest.pt'
    parent = None; adaptation = None; saved = None
    if args.continue_from:
        assert not checkpoint.exists() and not (out / 'final.pt').exists()
        saved, parent = read_parent(args.continue_from, args.parent_checkpoint_sha,
                                    args.parent_trainer_sha, fingerprint, runtime, len(rows))
        fingerprint['continuation_parent'] = parent
    elif args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'continuation_parent' in saved['fingerprint']:
            parent = saved['fingerprint']['continuation_parent']
            fingerprint['continuation_parent'] = parent
        if 'adaptation_parent' in saved['fingerprint']:
            adaptation = saved['fingerprint']['adaptation_parent']
            fingerprint['adaptation_parent'] = adaptation
        assert saved['fingerprint'] == fingerprint
        assert saved['runtime'] == runtime
    else:
        assert not checkpoint.exists() and not (out / 'final.pt').exists()
    if args.initialize_from:
        weights, adaptation = read_initialization(args.initialize_from, args.initial_final_sha,
                                                  manifest, labels, fingerprint)
        model.load_state_dict(weights)
        fingerprint['adaptation_parent'] = adaptation
        assert not optimizer.state_dict()['state']
        assert tensor_digest(model.state_dict()) == adaptation['parent_terminal']['model_tensor_digest']
        assert epoch == offset == global_step == 0
        print(json.dumps({'event': 'adaptation_initialized', 'parent_final_sha256': args.initial_final_sha,
                          'model_tensor_digest': adaptation['parent_terminal']['model_tensor_digest'],
                          'model_exact': True, 'optimizer_state_empty': True,
                          'new_epoch': epoch, 'new_global_step': global_step}), flush=True)
        del weights
    if saved is not None:
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        epoch, offset, global_step = saved['epoch'], saved['offset'], saved['global_step']
        epoch_stats, completed_epochs = saved['epoch_stats'], saved['completed_epochs']
        training_seconds = saved['training_seconds']
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch']); torch.cuda.set_rng_state_all(saved['rng_cuda'])
        if args.continue_from:
            # This happens before any new forward/dropout/update and records a
            # real transfer, not just successful checkpoint deserialization.
            assert tensor_digest(model.state_dict()) == parent['parent_terminal']['model_tensor_digest']
            loaded_optimizer = optimizer.state_dict()
            assert loaded_optimizer['param_groups'] == saved['optimizer']['param_groups']
            for key, old_state in saved['optimizer']['state'].items():
                for name, old_value in old_state.items():
                    actual = loaded_optimizer['state'][key][name]
                    assert torch.equal(actual.cpu(), old_value) if isinstance(old_value, torch.Tensor) else actual == old_value
            assert torch.equal(torch.get_rng_state(), saved['rng_torch'])
            assert all(torch.equal(actual, expected) for actual, expected in
                       zip(torch.cuda.get_rng_state_all(), saved['rng_cuda']))
            print(json.dumps({'event': 'completed_parent_exactly_restored', 'epoch': epoch,
                              'offset': offset, 'global_step': global_step,
                              'model_tensor_digest': parent['parent_terminal']['model_tensor_digest'],
                              'optimizer_and_rng_exact': True}), flush=True)
        del saved
    parent_seconds = parent['initial_training_seconds'] if parent else 0.
    atomic_json({'fingerprint': fingerprint, 'runtime': runtime, 'command_config': vars(args),
                 'initialization': (adaptation['policy'] if adaptation else
                                    'Exact completed 30epoch model/Adam/RNG continued to fixed total100; parent preserved'
                                    if parent else 'fresh general BiomedBERT + randomly initialized HPO head; pilots never resumed'),
                 'heldout_used': False, 'optimizer': 'Adam eps1e-8 no weight decay; no scheduler; BF16 autocast'},
                out / 'manifest.json')
    model.train(); torch.cuda.reset_peak_memory_stats()
    started = last_saved = time.monotonic(); interval_started = started

    def save(status):
        nonlocal training_seconds, interval_started, last_saved
        torch.cuda.synchronize()
        now = time.monotonic(); training_seconds += now - interval_started
        state = {'fingerprint': fingerprint, 'runtime': runtime, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'offset': offset,
                 'global_step': global_step, 'epoch_stats': epoch_stats, 'completed_epochs': completed_epochs,
                 'training_seconds': training_seconds, 'rng_python': random.getstate(),
                 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state_all()}
        atomic_save(state, checkpoint)
        atomic_json({'status': status, 'epoch': epoch, 'offset': offset, 'global_step': global_step,
                     'completed_epochs': completed_epochs, 'current_epoch_online_training': summarize(epoch_stats),
                     'training_seconds_excluding_checkpoint_io': training_seconds,
                     'additional_training_seconds_excluding_checkpoint_io': training_seconds - parent_seconds,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'model_tensor_digest': tensor_digest(model.state_dict()),
                     'checkpoint_sha256': sha(checkpoint), 'heldout_used': False}, out / 'progress.json')
        last_saved = interval_started = time.monotonic()

    while epoch < args.epochs:
        # A private generator makes order independent of dropout/global RNG and restart.
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed + epoch))
        while offset < len(rows):
            take = order[offset:offset + args.batch_size]
            ids, mask, segments, target = [value[take].cuda() for value in tensors]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(ids, mask, segments)
                per_row = torch.nn.functional.cross_entropy(logits.float(), target, reduction='none')
                loss = per_row.mean()
            assert bool(torch.isfinite(loss))
            loss.backward(); optimizer.step()
            predicted = logits.detach().argmax(-1)
            groups = [('hpo_positive', target != 0), ('reject_none', target == 0)]
            if adaptation:
                real = real_rows[take].cuda()
                groups.extend([('real_positive', real), ('hpo_replay', (target != 0) & ~real)])
            for name, selected in groups:
                stats = epoch_stats.setdefault(name, {'count': 0, 'loss_sum': 0., 'correct': 0, 'predicted_none': 0})
                stats['count'] += int(selected.sum().item())
                stats['loss_sum'] += float(per_row.detach()[selected].sum().item())
                stats['correct'] += int(((predicted == target) & selected).sum().item())
                stats['predicted_none'] += int(((predicted == 0) & selected).sum().item())
            offset += len(take); global_step += 1
            if global_step % 100 == 0:
                print(json.dumps({'event': 'training', 'epoch': epoch + 1, 'step': global_step,
                                  'offset': offset, 'groups': summarize(epoch_stats)}), flush=True)
            if args.stop_after_step is not None and global_step >= args.stop_after_step:
                save('controlled_interruption'); return
            if time.monotonic() - last_saved >= args.save_every_seconds:
                save('running')
        completed_epochs.append({'epoch': epoch + 1, 'online_training': summarize(epoch_stats)})
        print(json.dumps({'event': 'epoch_complete', **completed_epochs[-1]}), flush=True)
        epoch += 1; offset = 0; epoch_stats = new_stats()
    save('complete')
    atomic_save({'fingerprint': fingerprint, 'runtime': runtime, 'model': model.state_dict(),
                 'labels': labels, 'completed_epochs': completed_epochs,
                 'global_step': global_step}, out / 'final.pt')
    atomic_json({'final_sha256': sha(out / 'final.pt'), 'latest_sha256': sha(checkpoint),
                 'model_tensor_digest': tensor_digest(model.state_dict()),
                 'global_step': global_step, 'epochs': epoch, 'heldout_used': False,
                 'checkpoint_retention': 'latest full training state + final inference model only'},
                out / 'complete.json')


if __name__ == '__main__':
    main()
