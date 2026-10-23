#!/usr/bin/env python3
"""Fixed-budget max-MIL relation background filter; all occurrence evidence."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]


def write_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def encode_occurrence(tokenizer, row, occurrence, limit=256):
    header = tokenizer.encode(occurrence['section']+'. Phenotype: '+row['name'], add_special_tokens=False)
    assert len(header) <= 64, 'Unexpected long HPO header; do not silently truncate'
    encoded = tokenizer(occurrence['text'], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    markers = [tokenizer.convert_tokens_to_ids('[unused'+str(i)+']') for i in [1, 2, 5, 6]]
    assert len(set(markers)) == 4 and tokenizer.unk_token_id not in markers
    assert not set(ids) & set(markers)
    start, end = occurrence['phenotype']
    inside = [i for i, (a, b) in enumerate(offsets) if a < end and start < b]
    assert inside
    first, last = min(inside), max(inside)+1
    patient_ranges = []
    for a, b in occurrence['patients']:
        positions = [i for i, (x, y) in enumerate(offsets) if x < b and a < y]
        assert positions
        patient_ranges.append((min(positions), max(positions)+1))
    left, right = 0, len(ids)
    while True:
        retained = [(a, b) for a, b in patient_ranges if left <= a and b <= right]
        openings, closings = {}, {}
        for a, b in retained:
            openings.setdefault(a, []).append(markers[0]); closings.setdefault(b, []).append(markers[1])
        openings.setdefault(first, []).append(markers[2]); closings.setdefault(last, []).append(markers[3])
        context = []
        for i in range(left, right+1):
            context.extend(closings.get(i, [])); context.extend(openings.get(i, []))
            if i < right:
                context.append(ids[i])
        sequence = tokenizer.build_inputs_with_special_tokens(header, context)
        if len(sequence) <= limit:
            break
        if first-left >= right-last and left < first:
            left += 1
        elif right > last:
            right -= 1
        elif left < first:
            left += 1
        else:
            raise ValueError('Full phenotype cannot fit fixed token budget')
    assert sequence.count(markers[2]) == sequence.count(markers[3]) == 1
    a, b = sequence.index(markers[2]), sequence.index(markers[3])
    assert a < b
    # Patient markers may be nested in the phenotype; strip just these before
    # verifying that every original phenotype WordPiece survived cropping.
    assert [v for v in sequence[a+1:b] if v not in markers[:2]] == ids[first:last]
    assert sequence.count(markers[0]) == sequence.count(markers[1]) == len(retained)
    types = tokenizer.create_token_type_ids_from_sequences(header, context)
    return {'input_ids': sequence, 'token_type_ids': types}, {
        'occurrences': 1, 'tokens': len(sequence), 'context_cropped': int(left > 0 or right < len(ids)),
        'phenotype_tokens': len(inside), 'phenotype_fully_retained': 1,
        'provided_patient_markers_before_crop': len(patient_ranges),
        'provided_patient_markers_retained': len(retained), 'patient_marker_occurrences': int(bool(retained)),
        'max_tokens': len(sequence), 'max_phenotype_tokens': len(inside)}


def encode_rows(tokenizer, rows):
    result, counts = [], Counter()
    for row in rows:
        encoded = []
        for occurrence in row['occurrences']:
            values, stat = encode_occurrence(tokenizer, row, occurrence)
            encoded.append(values)
            for k, v in stat.items():
                counts[k] = max(counts[k], v) if k.startswith('max_') else counts[k]+v
        result.append(dict(row, encoded=encoded))
    return result, dict(counts)


def collate(rows, tokenizer):
    occurrences = [o for row in rows for o in row['encoded']]
    longest = max(len(o['input_ids']) for o in occurrences)
    result = {k: [] for k in ['input_ids', 'token_type_ids', 'attention_mask']}
    for occurrence in occurrences:
        length = len(occurrence['input_ids']); padding = longest-length
        result['input_ids'].append(occurrence['input_ids']+[tokenizer.pad_token_id]*padding)
        result['token_type_ids'].append(occurrence['token_type_ids']+[0]*padding)
        result['attention_mask'].append([1]*length+[0]*padding)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}, [len(r['encoded']) for r in rows]


def bag_max(logits, sizes):
    assert logits.ndim == 1 and all(size > 0 for size in sizes) and sum(sizes) == len(logits)
    return torch.stack([part.max() for part in logits.split(sizes)])


def loss_check(tokenizer, rows):
    logits = torch.tensor([1., 0., -2.], requires_grad=True)
    reduced = bag_max(logits, [2, 1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(reduced, torch.tensor([1., 0.]))
    loss.backward()
    assert reduced.tolist() == [1., -2.]
    assert logits.grad[0] < 0 and logits.grad[1] == 0 and logits.grad[2] > 0
    assert torch.isfinite(loss)
    chosen = [next(r for r in rows if r['label'] == label) for label in [0, 1]]
    batch, sizes = collate(chosen, tokenizer)
    for i, occurrence in enumerate([o for r in chosen for o in r['encoded']]):
        n = len(occurrence['input_ids'])
        assert batch['attention_mask'][i, :n].all() and not batch['attention_mask'][i, n:].any()
        assert (batch['input_ids'][i, n:] == tokenizer.pad_token_id).all()
    # Padded occurrence slots do not exist: sizes partition only real logits.
    try:
        bag_max(torch.tensor([1., 0., -2., 999.]), [2, 1])
    except AssertionError:
        rejected = True
    else:
        rejected = False
    assert rejected
    return {'loss': float(loss.detach()), 'gradient': logits.grad.tolist(), 'max_logits': reduced.tolist(),
            'unweighted_BCE': True, 'padding_attention_checked': True,
            'extra_occurrence_padding_rejected': True, 'fixture_bag_sizes': sizes}


def batches(rows, order):
    current, count = [], 0
    for index in order:
        n = len(rows[index]['encoded'])
        if current and (count+n > 32 or len(current) >= 8):
            yield current; current, count = [], 0
        current.append(rows[index]); count += n
    if current:
        yield current


class Model(torch.nn.Module):
    def __init__(self, path, vocabulary_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True, add_pooling_layer=False)
        self.encoder.resize_token_embeddings(vocabulary_size, mean_resizing=False)
        self.dropout = torch.nn.Dropout(.1)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, **batch):
        return self.head(self.dropout(self.encoder(**batch).last_hidden_state[:, 0])).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    for key in ['data', 'model', 'output']:
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--phase', choices=['check', 'train'], required=True)
    args = ap.parse_args(); start = time.monotonic(); torch.set_num_threads(1)
    random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
    data, out = Path(args.data), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data/'manifest.json').read_text())
    for name in ['train.jsonl.gz', 'input.jsonl.gz']:
        assert manifest['outputs'][name] == sha(data/name)
    rawtrain, rawinput = read(data/'train.jsonl.gz'), read(data/'input.jsonl.gz')
    assert all('label' not in row for row in rawinput)
    assert not {r['pmc_id'] for r in rawtrain} & {r['pmc_id'] for r in rawinput}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    # PubMedBERT's vocabulary has no BERT-style unused tokens: without an
    # explicit registration all four direct marker IDs would be [UNK].
    marker_names = ['[unused'+str(i)+']' for i in [1, 2, 5, 6]]
    original_vocabulary = len(tokenizer)
    tokenizer.add_special_tokens({'additional_special_tokens': marker_names})
    tokenizer.save_pretrained(out/'tokenizer')
    restored = AutoTokenizer.from_pretrained(out/'tokenizer', use_fast=True, local_files_only=True)
    assert restored.get_vocab() == tokenizer.get_vocab()
    assert restored.special_tokens_map == tokenizer.special_tokens_map
    assert restored.encode(' '.join(marker_names)) == tokenizer.encode(' '.join(marker_names))
    training, train_stat = encode_rows(tokenizer, rawtrain)
    testing, input_stat = encode_rows(tokenizer, rawinput)
    check = loss_check(tokenizer, training)
    fingerprint = {'source_sha256': sha(__file__), 'data_manifest_sha256': sha(data/'manifest.json'),
                   'base': {p.name: sha(p) for p in sorted(Path(args.model).iterdir()) if p.is_file()},
                   'tokenizer_markers': {'original_vocabulary': original_vocabulary, 'vocabulary': len(tokenizer),
                                         'ids': {name: tokenizer.convert_tokens_to_ids(name) for name in marker_names}},
                   'saved_tokenizer': {p.name: sha(p) for p in sorted((out/'tokenizer').iterdir()) if p.is_file()},
                   'config': {'epochs': 3, 'learning_rate': 2e-5, 'weight_decay': .01, 'dropout': .1,
                              'seed': 20261023, 'max_tokens': 256, 'max_bags': 8, 'occurrence_batch_budget': 32,
                              'all_occurrences': True, 'BCE_weight': None, 'pooling': 'max_logit', 'threshold': .5}}
    audit = {'fingerprint': fingerprint, 'training': train_stat, 'input': input_stat, 'loss_check': check,
             'torch': torch.__version__, 'phase': args.phase, 'saved_tokenizer_roundtrip_exact': True}
    write_json(out/'cpucheck.json', audit)
    if args.phase == 'check':
        print(json.dumps(audit)); return
    assert not (out/'final.pt').exists()
    torch.cuda.manual_seed_all(20261023)
    model = Model(args.model, len(tokenizer)).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    stats, step = [], 0
    probe = dict(model.named_parameters())['encoder.encoder.layer.0.attention.self.query.weight']
    first_update = None
    for epoch in range(3):
        model.train(); epoch_start = time.monotonic(); losses, labels, correct = [], [], []
        order = list(range(len(training))); random.Random(20261023+epoch).shuffle(order)
        for batch_rows in batches(training, order):
            batch, sizes = collate(batch_rows, tokenizer)
            labels_tensor = torch.tensor([r['label'] for r in batch_rows], dtype=torch.float32, device='cuda')
            batch = {k: v.cuda() for k, v in batch.items()}; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = bag_max(model(**batch).float(), sizes)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_tensor)
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                before_probe = probe.detach().clone(); before_head = model.head.weight.detach().clone()
                first_update = {'encoder_gradient_norm': float(probe.grad.norm()),
                                'head_gradient_norm': float(model.head.weight.grad.norm())}
                assert first_update['encoder_gradient_norm'] > 0 and first_update['head_gradient_norm'] > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 0:
                first_update.update(encoder_delta_max=float((probe.detach()-before_probe).abs().max()),
                                    head_delta_max=float((model.head.weight.detach()-before_head).abs().max()))
                assert first_update['encoder_delta_max'] > 0 and first_update['head_delta_max'] > 0
                write_json(out/'first-update.json', first_update)
            step += 1
            losses.append(float(loss.detach())*len(batch_rows)); labels.extend(r['label'] for r in batch_rows)
            correct.extend(((logits.detach() >= 0) == labels_tensor.bool()).cpu().tolist())
        stat = {'epoch': epoch+1, 'global_step': step, 'bags': len(labels), 'positive': sum(labels),
                'loss': sum(losses)/len(labels), 'positive_accuracy': sum(c for y, c in zip(labels, correct) if y)/sum(labels),
                'negative_accuracy': sum(c for y, c in zip(labels, correct) if not y)/(len(labels)-sum(labels)),
                'seconds': time.monotonic()-epoch_start}
        stats.append(stat); print(json.dumps(stat), flush=True)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                      'global_step': step, 'fingerprint': fingerprint, 'stats': stats,
                      'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                      'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        torch.save(checkpoint, out/'latest.pt.tmp'); (out/'latest.pt.tmp').replace(out/'latest.pt')
        write_json(out/'progress.json', {'stats': stats, 'global_step': step, 'fingerprint': fingerprint})
    torch.save({'model': model.state_dict(), 'fingerprint': fingerprint, 'epochs': 3}, out/'final.pt')
    model.eval(); scored = []
    with torch.inference_mode():
        for batch_rows in batches(testing, range(len(testing))):
            batch, sizes = collate(batch_rows, tokenizer); batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = bag_max(model(**batch).float(), sizes)
            for row, value in zip(batch_rows, logits.cpu().tolist()):
                scored.append({'pmc_id': row['pmc_id'], 'concept': row['concept'], 'occurrences': len(row['encoded']),
                               'logit': value, 'probability': float(torch.sigmoid(torch.tensor(value)))})
    path = out/'bag-scores.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in scored))
    complete = {'fingerprint': fingerprint, 'statistics': stats, 'input_bags': len(scored),
                'scores_sha256': sha(path), 'final_sha256': sha(out/'final.pt'), 'latest_sha256': sha(out/'latest.pt'),
                'global_step': step, 'epochs': 3, 'first_update': first_update,
                'parameters': sum(p.numel() for p in model.parameters()),
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start,
                'input_annotations_used': False}
    write_json(out/'complete.json', complete); print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
