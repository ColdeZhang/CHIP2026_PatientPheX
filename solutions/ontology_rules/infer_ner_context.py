#!/usr/bin/env python3
"""Conditional input-only BioMed context inference through target-only outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import ner_context
import ner_context_runtime as runtime
from ner_context_runtime import sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'model-dir', 'tokenizer-dir', 'output-dir']:
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--training-manifest')
    ap.add_argument('--cpu-prepare-only', action='store_true')
    args = ap.parse_args()
    import torch
    from gliner.decoding.decoder import SpanDecoder
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    started = time.monotonic()
    assert sha(args.input) == runtime.HELD_INPUT_SHA
    docs = runtime.read_rows(args.input)
    assert len(docs) == len({str(d['pmc_id']) for d in docs}) == 16
    assert all('entities' not in d and 'association' not in d for d in docs)
    config, tokenizer, processor, model_files = runtime.native_processor(args.model_dir, args.tokenizer_dir)
    builder = ner_context.ContextBuilder(docs, tokenizer, config)
    rows = builder.input_rows()
    dependencies = runtime.source_fingerprint(__file__)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists() and not (out/'candidates.jsonl').exists()
    if args.cpu_prepare_only:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        chosen = rows[:3]+[max(rows, key=lambda r: r['tokens'])]
        tensor = ner_context.collate(processor, chosen, builder, prepare_labels=False)
        assert 'labels' not in tensor
        report = {'scope': 'CPU inference-entry input fixture only; no weight/model/prediction/score',
                  'sources': dependencies, 'model_config_tokenizer': model_files,
                  'input_sha256': sha(args.input), 'documents': 16, 'target_windows': len(rows),
                  'fixture_row_ids': [r['row_id'] for r in chosen],
                  'fixture_tokens': tensor['attention_mask'].sum(-1).tolist(),
                  'fixture_target_lengths': tensor['text_lengths'].flatten().tolist(),
                  'cuda_initialized': False, 'checkpoint_opened': False,
                  'seconds': time.monotonic()-started}
        path = out/'cpu-preparation.json'; assert not path.exists()
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'cpu_preparation': True, 'target_windows': len(rows)}), flush=True)
        return
    assert args.training_manifest and torch.cuda.is_available()
    trained = json.loads(Path(args.training_manifest).read_text())
    assert trained['completed_epochs'] == 10 and trained['total_steps'] == 2880 and trained['rows'] == 1150
    assert trained['fingerprint']['train_sha256'] == runtime.PREPARED_TRAIN_SHA
    assert trained['fingerprint']['raw_train_sha256'] == runtime.RAW_TRAIN_SHA
    assert trained['fingerprint']['excluded_input_sha256'] == sha(args.input)
    assert trained['fingerprint']['base_weight_sha256'] == runtime.BIO_BASE_SHA
    for key in ['adapter_sha256', 'runtime_helper_sha256', 'gliner_python_sources', 'package_versions']:
        assert trained['fingerprint']['dependencies'][key] == dependencies[key], key
    assert trained['fingerprint']['model_config_tokenizer'] == model_files
    final = trained['epoch_results'][-1]
    assert final['epoch'] == 10 and final['step'] == 2880
    weight = Path(args.model_dir)/'pytorch_model.bin'
    assert sha(weight) == final['model_sha256']
    model = runtime.load_model(args.model_dir, args.tokenizer_dir, final['model_sha256']).to('cuda')
    assert model.config.to_dict() == processor.config.to_dict()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    decoder = SpanDecoder(config)
    by_doc = {str(d['pmc_id']): {} for d in docs}
    source_docs = {str(d['pmc_id']): d for d in docs}
    raw = []
    begin = time.monotonic()
    with torch.no_grad():
        for index, row in enumerate(rows):
            # Fixed one-window batching matches the original high-level policy.
            tensors = ner_context.collate(processor, [row], builder, prepare_labels=False)
            tensors = {k: v.to('cuda') for k, v in tensors.items()}
            output = model.model(**tensors)
            assert output.loss is None
            decoded = ner_context.decode(decoder, [row], output.logits, tensors['span_mask'], threshold=.5)
            entities = ner_context.global_entities([row], decoded)[0]
            passage = source_docs[row['pmc_id']]['full_text'][row['passage_index']]
            for entity in entities:
                offset, length = entity['offset'], entity['length']
                local = offset-int(passage['offset'])
                assert passage['text'][local:local+length] == entity['text']
                item = dict(entity, start=offset-row['window_start'], end=offset-row['window_start']+length,
                            pmc_id=source_docs[row['pmc_id']]['pmc_id'], window_index=index)
                raw.append(item)
                key = offset, length, entity['label']
                previous = by_doc[row['pmc_id']].get(key)
                if previous is None or previous['score'] < item['score']:
                    by_doc[row['pmc_id']][key] = item
            if (index+1) % 50 == 0 or index+1 == len(rows):
                print(json.dumps({'completed': index+1, 'chunks': len(rows), 'raw_spans': len(raw)}), flush=True)
    candidate_path = out/'candidates.jsonl'
    candidate_path.write_text(''.join(json.dumps({'pmc_id': d['pmc_id'], 'candidates':
        sorted(by_doc[str(d['pmc_id'])].values(), key=lambda e: (e['offset'], e['length']))},
        ensure_ascii=False)+'\n' for d in docs))
    (out/'raw-spans.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in raw))
    manifest = {'configuration': vars(args), 'source_sha256': sha(__file__),
        'loader_source_sha256': sha(runtime.__file__), 'dependencies': dependencies,
        'input_sha256': sha(args.input), 'training_manifest_sha256': sha(args.training_manifest),
        'label': ['phenotype'], 'threshold': .5, 'flat_ner': False, 'overlap_words': 12,
        'token_budget': 384, 'batch_size': 1, 'context_target_only': True,
        'model_parameters': sum(p.numel() for p in model.parameters()), 'strict_state_dict': True,
        'files': {str(weight): sha(weight), str(Path(args.model_dir)/'gliner_config.json'):
            sha(Path(args.model_dir)/'gliner_config.json')},
        'model_config_tokenizer': model_files, 'stats': {'documents': len(docs), 'chunks': len(rows)},
        'inference_seconds': time.monotonic()-begin, 'total_seconds': time.monotonic()-started,
        'gpu': torch.cuda.get_device_name(0), 'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'raw_spans': len(raw), 'unique_spans': sum(len(v) for v in by_doc.values()),
        'all_returned_spans_match_full_words_and_original_coordinates': True,
        'gold_read': False, 'hpo_linking': False, 'prediction_scoring': False,
        'candidates_sha256': sha(candidate_path)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'gold_read': False,
        'candidate_sha256': sha(candidate_path), 'raw_span_sha256': sha(out/'raw-spans.jsonl'),
        'manifest_sha256': sha(out/'manifest.json'), 'input_sha256': sha(args.input),
        'model_sha256': sha(weight), 'source_sha256': sha(__file__)}
    (out/'prediction-freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    print(json.dumps({'complete': True, 'unique_spans': manifest['unique_spans'],
                      'inference_seconds': manifest['inference_seconds']}), flush=True)


if __name__ == '__main__':
    main()
