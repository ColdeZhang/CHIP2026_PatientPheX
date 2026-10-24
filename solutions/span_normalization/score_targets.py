"""Score every original candidate boundary directly, without the GP generation floor."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_infer import Collator, load_model, move
from common import read_jsonl, write_json, write_jsonl, ref, sha


def key(r): return (str(r['pmc_id']), int(r['offset']), int(r['length']))


def build_mapping(rows, targets):
    bydoc = defaultdict(list)
    for t in targets: bydoc[t['pmc_id']].append(t)
    mapping = {}; presence = defaultdict(lambda: [False, False]); coverage = Counter()
    for r in rows:
        base = r['passage_offset']; starts = {a+base: i for i, (a,b) in enumerate(r['offsets']) if b>a}
        ends = {b+base: i for i, (a,b) in enumerate(r['offsets']) if b>a}; m = []
        for t in bydoc[r['pmc_id']]:
            k = key(t); a = t['offset']; b = a+t['length']
            presence[k][0] |= a in starts; presence[k][1] |= b in ends
            if a in starts and b in ends and starts[a] <= ends[b]:
                m.append((k, starts[a], ends[b])); coverage[k] += 1
        mapping[r['row_id']] = m
    return mapping, presence, coverage


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'weights', 'weights-sha256', 'preparation', 'targets', 'output', 'gpu-uuid'):
        ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); start = time.monotonic(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'complete.json').exists(), 'Already scored: recover existing output'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu_uuid and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    assert sha(a.weights) == a.weights_sha256
    prep = json.loads(Path(a.preparation).read_text()); hw = prep['windows']['held']
    assert sha(hw['path']) == hw['sha256']; rows = read_jsonl(hw['path']); targets = read_jsonl(a.targets)
    assert all('positive' not in r for r in rows) and len({key(t) for t in targets}) == len(targets)
    assert {t['pmc_id'] for t in targets} == set(prep['held_pmcs'])
    mapping, presence, coverage = build_mapping(rows, targets)
    model = load_model(a.model, out/'loading.json')
    loaded = model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True), strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys; model.eval()
    collate = Collator(AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True))
    values = {}; seen = []; total_target_window_scores = 0
    with torch.inference_mode():
        for batch, _, valid, chunk in DataLoader(rows, batch_size=16, shuffle=False, collate_fn=collate):
            with torch.autocast('cuda', dtype=torch.bfloat16): scores = model(**move(batch)).float().cpu()
            for n, row in enumerate(chunk):
                seen.append(row['row_id'])
                for k, i, j in mapping[row['row_id']]:
                    assert valid[n,i,j]; value = float(scores[n,i,j]); assert math.isfinite(value)
                    values[k] = max(values.get(k, -math.inf), value); total_target_window_scores += 1
    assert seen == [r['row_id'] for r in rows] and total_target_window_scores == sum(coverage.values())
    output = []; reasons = Counter()
    for t in targets:
        k = key(t); observed = k in values
        why = None if observed else ('unaligned_endpoint' if not all(presence[k]) else 'no_common_window')
        if why: reasons[why] += 1
        output.append({**t, 'gp_span_valid': observed, 'gp_span_logit': values[k] if observed else 0.0,
                       'missing_reason': why, 'valid_window_count': coverage[k]})
    write_jsonl(out/'target-scores.jsonl', output)
    result = {'status': 'COMPLETE_ALL_TARGET_BOUNDARIES', 'source': ref(__file__),
              'sources': {k: ref(getattr(a,k)) for k in ('weights', 'preparation', 'targets')},
              'output': ref(out/'target-scores.jsonl'), 'held_windows': len(rows), 'held_documents': len(prep['held_pmcs']),
              'target_spans': len(targets), 'valid_spans': len(values), 'missing_reasons': dict(reasons),
              'valid_scores_below_minus2': sum(v < -2. for v in values.values()), 'no_score_floor': True,
              'missing_policy': 'valid=false/logit=0; scores for valid targets never clipped; max over all matching windows',
              'parameters': sum(p.numel() for p in model.parameters()), 'strict_state_reload': True,
              'held_gold_read': False, 'seconds': time.monotonic()-start}
    write_json(out/'complete.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
