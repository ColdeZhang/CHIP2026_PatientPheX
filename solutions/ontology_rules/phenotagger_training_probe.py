#!/usr/bin/env python3
"""Endpoint training-only sanity probe; never selects a checkpoint or threshold."""
import argparse
import gzip
import json
from pathlib import Path
import random
import torch
from transformers import AutoTokenizer
from phenotagger_measure import OntologyClassifier, sha


def main():
    parser = argparse.ArgumentParser()
    for key in ['data-dir', 'training-dir', 'model-dir', 'output-dir']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'report.json').exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 1
    trained = Path(args.training_dir); data = Path(args.data_dir)
    complete = json.loads((trained / 'complete.json').read_text())
    assert complete['epochs'] == 30 and sha(trained / 'final.pt') == complete['final_sha256']
    saved = torch.load(trained / 'final.pt', map_location='cpu', weights_only=False)
    assert sha(data / 'manifest.json') == saved['fingerprint']['data_manifest_sha256']
    assert sha(data / 'rows.jsonl.gz') == saved['fingerprint']['data_outputs']['rows.jsonl.gz']
    with gzip.open(data / 'rows.jsonl.gz', 'rt') as handle: rows = [json.loads(line) for line in handle]
    labels = saved['labels']; lookup = {name: i for i, name in enumerate(labels)}
    rng = random.Random(20261023)
    selected = {}
    for group, positive in [('hpo_positive', True), ('reject_none', False)]:
        selected[group] = sorted(rng.sample([i for i, row in enumerate(rows)
                                           if (row['label'] != 'HP:None') == positive], 512))
    model = OntologyClassifier(args.model_dir, len(labels)).cuda()
    model.load_state_dict(saved['model']); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    records = []
    with torch.inference_mode():
        for group, indices in selected.items():
            for start in range(0, len(indices), 256):
                at = indices[start:start + 256]
                batch = tokenizer([rows[i]['text'] for i in at], max_length=32,
                                  truncation=True, padding='max_length', return_tensors='pt')
                target = torch.tensor([lookup[rows[i]['label']] for i in at], device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch['input_ids'].cuda(), batch['attention_mask'].cuda(),
                                   batch.get('token_type_ids', torch.zeros_like(batch['input_ids'])).cuda())
                losses = torch.nn.functional.cross_entropy(logits.float(), target, reduction='none').cpu().tolist()
                probs = logits.float().softmax(-1); top = probs.topk(3, dim=-1)
                for k, (index, loss, ids, values) in enumerate(zip(at, losses, top.indices.cpu().tolist(), top.values.cpu().tolist())):
                    records.append({'group': group, 'training_row': index, 'target': rows[index]['label'],
                                    'top3': [(labels[i], value) for i, value in zip(ids, values)],
                                    'loss': loss, 'none_probability': float(probs[k, 0].item())})
    result = {}
    for group in selected:
        current = [row for row in records if row['group'] == group]
        result[group] = {'rows': len(current), 'mean_loss': sum(r['loss'] for r in current) / len(current),
                         'top1_accuracy': sum(r['target'] == r['top3'][0][0] for r in current) / len(current),
                         'top3_accuracy': sum(r['target'] in [p[0] for p in r['top3']] for r in current) / len(current),
                         'top1_none_fraction': sum(r['top3'][0][0] == 'HP:None' for r in current) / len(current),
                         'mean_none_probability': sum(r['none_probability'] for r in current) / len(current)}
    (out / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    report = {'source_sha256': sha(__file__), 'training_source': saved['fingerprint'], 'terminal': complete,
              'sample_policy': 'seed20261023, 512 without-replacement rows each HPO positive and None in training data, endpoint only',
              'selected_indices': selected, 'groups': result, 'heldout_used': False,
              'interpretation': 'Training self-check, not generalization accuracy; does not select epochs or threshold',
              'output_rows_sha256': sha(out / 'rows.jsonl')}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(result))


if __name__ == '__main__': main()
