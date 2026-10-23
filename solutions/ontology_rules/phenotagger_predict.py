#!/usr/bin/env python3
"""Score frozen textual candidates with the fixed terminal ontology classifier."""
import argparse
import gzip
import io
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from phenotagger_measure import OntologyClassifier, sha
import phenotagger_measure


def main():
    parser = argparse.ArgumentParser()
    for key in ['candidate-dir', 'training-dir', 'model-dir', 'output-dir']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--expected-epochs', type=int, choices=[3, 30, 100], default=30)
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 1
    trained = Path(args.training_dir); terminal = json.loads((trained / 'complete.json').read_text())
    assert terminal['epochs'] == args.expected_epochs and terminal['final_sha256'] == sha(trained / 'final.pt')
    saved = torch.load(trained / 'final.pt', map_location='cpu', weights_only=False)
    assert saved['global_step'] == terminal['global_step']
    assert saved['fingerprint']['classifier_source_sha256'] == sha(phenotagger_measure.__file__)
    for name, expected in saved['fingerprint']['base_source_files'].items():
        assert sha(Path(args.model_dir) / name) == expected
    labels = saved['labels']; assert labels[0] == 'HP:None'
    candidate_dir = Path(args.candidate_dir)
    preparation = json.loads((candidate_dir / 'manifest.json').read_text())
    assert sha(candidate_dir / 'candidates.jsonl.gz') == preparation['output_sha256']
    expected_manifest = saved['fingerprint']['data_manifest_sha256']
    if 'adaptation_parent' in saved['fingerprint']:
        assert args.expected_epochs == 3 and terminal['global_step'] == 1701
        expected_manifest = saved['fingerprint']['adaptation_parent']['parent_data_manifest_sha256']
    assert preparation['inputs']['training_manifest'] == expected_manifest
    with gzip.open(candidate_dir / 'candidates.jsonl.gz', 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == preparation['rows']
    texts = sorted({row['model_text'] for row in rows}); index = {text: i for i, text in enumerate(texts)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    lengths = [len(ids) for ids in tokenizer(texts, padding=False, truncation=False)['input_ids']]
    max_length = saved['fingerprint']['config']['max_length']
    model = OntologyClassifier(args.model_dir, len(labels)).cuda()
    model.load_state_dict(saved['model']); del saved; model.eval()
    scores = []; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], padding='max_length',
                              truncation=True, max_length=max_length, return_tensors='pt')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch['input_ids'].cuda(), batch['attention_mask'].cuda(),
                               batch.get('token_type_ids', torch.zeros_like(batch['input_ids'])).cuda())
            top = logits.float().softmax(-1).topk(3, dim=-1)
            for values, ids in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                scores.append([(labels[ident], probability) for ident, probability in zip(ids, values)])
            if start % (args.batch_size * 50) == 0:
                print(json.dumps({'scored': len(scores), 'total': len(texts)}), flush=True)
    target = out / 'scored.jsonl.gz'
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as z:
        with io.TextIOWrapper(z, encoding='utf-8') as handle:
            for row in rows:
                at = index[row['model_text']]; top = scores[at]
                handle.write(json.dumps(dict(row, identifier=top[0][0], probability=top[0][1],
                    top3=top, tokens=lengths[at], overlength=lengths[at] > max_length)) + '\n')
    report = {'source_sha256': sha(__file__), 'classifier_source_sha256': sha(phenotagger_measure.__file__),
              'terminal': terminal, 'candidate_manifest_sha256': sha(candidate_dir / 'manifest.json'),
              'rows': len(rows), 'unique_texts': len(texts), 'overlength_unique': sum(n > max_length for n in lengths),
              'overlength_policy': 'Scored for transparent diagnostics; assembler rejects model additions beyond max32.',
              'prediction_seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'input_annotations_used': False, 'output_sha256': sha(target), 'config': vars(args)}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
