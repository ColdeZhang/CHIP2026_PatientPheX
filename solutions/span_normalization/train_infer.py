"""Fixed eight-epoch GP training and full held-input inference; no held gold.

Raw generation boundary: all special-token-free i<=j spans scoring >=-2,
including nested spans. No overlap/word-boundary filter before raw persistence.
The later task decoder applies the fixed >0 decision and word-boundary rule.
"""
import argparse
from collections import defaultdict
import contextlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from common import read_jsonl, write_json, write_jsonl, ref, sha
from gp import GlobalPointer, multilabel_loss


class Collator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer

    def __call__(self, rows):
        batch = self.tokenizer.pad([r['features'] for r in rows], padding=True, return_tensors='pt')
        n = batch['input_ids'].shape[1]
        valid = torch.zeros((len(rows), n, n), dtype=torch.bool)
        positive = torch.zeros_like(valid)
        triangle = torch.ones((n, n), dtype=torch.bool).triu()
        for b, row in enumerate(rows):
            tokens = torch.zeros(n, dtype=torch.bool)
            tokens[:len(row['offsets'])] = torch.tensor([end > start for start, end in row['offsets']])
            valid[b] = tokens[:, None] & tokens[None, :] & triangle
            for i, j in row.get('positive', []): positive[b, i, j] = True
        assert not torch.any(positive & ~valid)
        return batch, positive, valid, rows


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model(path, out):
    encoder, info = AutoModel.from_pretrained(path, local_files_only=True,
                                            add_pooling_layer=False, output_loading_info=True)
    write_json(out, info)
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    assert all(k.startswith('cls.') for k in info.get('unexpected_keys', [])), info.get('unexpected_keys')
    return GlobalPointer(encoder).cuda()


def move(batch): return {k: v.cuda() for k, v in batch.items()}


def persist_torch(path, state):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    with path.open('rb') as f: os.fsync(f.fileno())
    return ref(path)


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'preparation', 'output', 'gpu-uuid'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); torch.set_num_threads(1)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu_uuid
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    prep = json.loads(Path(a.preparation).read_text()); cfg = prep['configuration']
    train = read_jsonl(prep['windows']['train']['path']); held = read_jsonl(prep['windows']['held']['path'])
    for key in ('train', 'held'):
        assert sha(prep['windows'][key]['path']) == prep['windows'][key]['sha256']
    assert all('positive' not in r for r in held)
    sources = {'preparation': ref(a.preparation), 'trainer': ref(__file__),
               'gp': ref(Path(__file__).with_name('gp.py')), 'common': ref(Path(__file__).with_name('common.py'))}
    assets = [ref(p) for p in sorted(Path(a.model).iterdir()) if p.is_file()]
    write_json(out / 'inputs.json', {'sources': sources, 'model_assets': assets, 'configuration': cfg})
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True)
    collate = Collator(tokenizer)
    # One actual longest training batch, update discarded; fits inside the same driver budget.
    seed_all(cfg['seed']); probe = load_model(a.model, out / 'probe-loading.json')
    rows = sorted(train, key=lambda x: len(x['features']['input_ids']), reverse=True)[:2]
    batch, pos, valid, _ = collate(rows); opt = torch.optim.AdamW(probe.parameters(), lr=cfg['learning_rate'])
    before = probe.head.weight.detach().clone()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = probe(**move(batch)); loss = multilabel_loss(logits, pos.cuda(), valid.cuda())
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None)
    encoder_nonzero = sum(bool(torch.count_nonzero(p.grad).item()) for p in probe.encoder.parameters() if p.grad is not None)
    assert encoder_nonzero > 0 and torch.count_nonzero(probe.head.weight.grad).item() > 0
    opt.step(); assert not torch.equal(before, probe.head.weight)
    write_json(out / 'probe.json', {'status': 'PASS', 'rows': [r['row_id'] for r in rows],
                                  'loss': float(loss.detach()), 'encoder_nonzero_grad_tensors': encoder_nonzero,
                                  'head_updated': True, 'seconds_from_start': time.monotonic()-start,
                                  'discarded_before_fresh_training': True})
    del probe, opt, logits, loss, before; torch.cuda.empty_cache()
    seed_all(cfg['seed']); model = load_model(a.model, out / 'formal-loading.json')
    parameters = sum(p.numel() for p in model.parameters()); assert parameters < 10_000_000_000
    loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    total_steps = len(loader)*cfg['epochs']; scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*.1), total_steps)
    log = []; steps = 0
    for epoch in range(cfg['epochs']):
        model.train(); loss_sum = 0.0; epoch_start = time.monotonic()
        for batch, pos, valid, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**move(batch)); loss = multilabel_loss(logits, pos.cuda(), valid.cuda())
            assert torch.isfinite(loss), 'nonfinite training loss'
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), 'nonfinite gradient norm'
            optimizer.step(); scheduler.step(); steps += 1; loss_sum += float(loss.detach())
        item = {'epoch': epoch+1, 'steps': steps, 'loss': loss_sum/len(loader), 'seconds': time.monotonic()-epoch_start}
        log.append(item); write_json(out/'progress.json', item); print(json.dumps(item), flush=True)
    model_ref = persist_torch(out/'model.pt', model.state_dict())
    state_ref = persist_torch(out/'training-state.pt', {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                             'completed_steps': steps, 'completed_epochs': cfg['epochs'], 'model': model_ref})
    assert steps == total_steps
    model.eval(); pool = defaultdict(dict); seen = []; inference_start = time.monotonic()
    with torch.inference_mode():
        for batch, _, valid, rows in DataLoader(held, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate):
            with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(**move(batch))
            scores = logits.float().cpu()
            for b, row in enumerate(rows):
                seen.append(row['row_id'])
                indices = torch.nonzero(valid[b] & (scores[b] >= cfg['raw_pool_floor']))
                for i, j in indices.tolist():
                    lo, hi = row['offsets'][i][0], row['offsets'][j][1]
                    key = (row['passage_index'], lo, hi); value = float(scores[b, i, j])
                    current = pool[row['pmc_id']].get(key)
                    if current is None or value > current: pool[row['pmc_id']][key] = value
    assert seen == [r['row_id'] for r in held]
    raw = [{'pmc_id': pmc, 'passage_index': pi, 'start': lo, 'end': hi, 'gp_logit': score}
           for pmc in prep['held_pmcs'] for (pi, lo, hi), score in sorted(pool[pmc].items())]
    write_jsonl(out/'span-pool.jsonl', raw)
    result = {'status': 'COMPLETE_TRAINING_AND_FULL_HELD_INFERENCE', 'model_parameters': parameters,
              'training_windows': len(train), 'held_windows': len(held), 'held_documents': len(prep['held_pmcs']),
              'epochs': log, 'optimizer_steps': steps, 'training_state': state_ref, 'model': model_ref,
              'span_pool': ref(out/'span-pool.jsonl'), 'pool_candidates': len(raw),
              'generation_boundary': 'All i<=j, nonspecial token spans; max across overlap windows; raw GP logit>=-2.0; no other prefilter',
              'inference_seconds': time.monotonic()-inference_start, 'seconds': time.monotonic()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'runtime': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__},
              'sources': sources, 'no_held_gold': True}
    write_json(out/'complete.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
