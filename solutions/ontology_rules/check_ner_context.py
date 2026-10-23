#!/usr/bin/env python3
"""CPU-only actual GLiNER processor/decoder check; no checkpoint is opened."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

import ner_context


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'rows', 'model-config-dir', 'reference-trainer', 'output']:
        ap.add_argument('--'+name, required=True)
    a = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    assert sha(a.train) == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert sha(a.rows) == 'a76db949c670f21909031ac99ac42e91d2eb9a959f4ae0d6cec9ed08f7154ed0'
    assert sha(a.reference_trainer) == '70c5a2ea93b00b3d8dfbe6bfb9183f57ef122406e1fc33b46ac4a7b4371eb5a9'
    import torch
    from transformers import AutoTokenizer
    from gliner import GLiNERConfig
    from gliner.data_processing import SpanProcessor
    from gliner.data_processing.tokenizer import WordsSplitter
    from gliner.decoding.decoder import SpanDecoder
    from gliner.modeling.base import extract_word_embeddings
    from gliner.modeling.span_rep import SpanRepLayer
    import importlib.util
    import sys
    sys.path.insert(0, str(Path(a.reference_trainer).parent))
    spec = importlib.util.spec_from_file_location('reference_trainer', a.reference_trainer)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    directory = Path(a.model_config_dir)
    assert sha(directory/'gliner_config.json') == 'f747cc67847f8b5412dca016377cb1a8c1642c802657dfb1949aca4d22354b76'
    assert sha(directory/'tokenizer.json') == 'ee028763434d18611c1c36356ea1d050e90a9fa94ede57fac48b39f85f818ad1'
    config = GLiNERConfig(**json.loads((directory/'gliner_config.json').read_text()))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    assert len(tokenizer) == 128003 and '[FLERT]' not in tokenizer.get_vocab()
    processor = SpanProcessor(config, tokenizer, WordsSplitter(config.words_splitter_type))
    decoder = SpanDecoder(config)
    expected_sources = {
        'processor': (SpanProcessor, 'bbc5393459e4c989d40d26ac2bc342ecb8607ebe7f445c25316790b51a7a4c12'),
        'decoder': (SpanDecoder, '02c0ce0aabccd58dbee60372608054d2eb304f8903126440a3d105f8e72517fa'),
        'word_extraction': (extract_word_embeddings, 'b3ef985596410c07a447f18f5e199339b97ca4f603441a9ff36f9242d949e5bd')}
    sources = {}
    for name, (obj, expected) in expected_sources.items():
        path = inspect.getsourcefile(obj)
        sources[name] = {'path': path, 'sha256': sha(path)}
        assert sources[name]['sha256'] == expected, name
    docs = [json.loads(s) for s in Path(a.train).read_text().splitlines() if s]
    inputs = [{'pmc_id': str(d['pmc_id']), 'full_text': d['full_text']} for d in docs]
    builder = ner_context.ContextBuilder(inputs, tokenizer, config)
    rows = [json.loads(s) for s in Path(a.rows).read_text().splitlines() if s]
    untouched = json.dumps(rows, sort_keys=True)
    inferred = builder.input_rows()
    assert len(inferred) == len(rows) == 1150
    for original, infer in zip(rows, inferred):
        assert all(original[k] == v for k, v in infer.items())
        assert builder.context(original) == builder.context(infer)
    counts = Counter()
    digest = hashlib.sha256()
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset+4]
        baseline = reference.collate(processor, batch)
        disabled = ner_context.collate(processor, batch, builder, context=False)
        enhanced = ner_context.collate(processor, batch, builder)
        unlabelled = ner_context.collate(processor, inferred[offset:offset+4], builder, prepare_labels=False)
        assert set(baseline) == set(disabled) == set(enhanced)
        for key in baseline:
            assert torch.equal(baseline[key], disabled[key]), ('disabled', key, offset)
        for key in ['span_idx', 'span_mask', 'text_lengths', 'labels']:
            assert torch.equal(baseline[key], enhanced[key]), ('supervision', key, offset)
        for key in unlabelled:
            assert torch.equal(enhanced[key], unlabelled[key]), ('inference', key, offset)
        assert 'labels' not in unlabelled
        for i, row in enumerate(batch):
            count = int(enhanced['attention_mask'][i].sum())
            counts['rows_with_context'] += count > row['tokens']
            counts['encoded_tokens'] += count
            counts['target_tokens'] += row['tokens']
            counts['NO_instances'] += len(row['no_span_indices'])
            assert (enhanced['labels'].reshape(len(batch), -1)[i, row['no_span_indices']] == 0).all()
            digest.update(json.dumps({'row_id': row['row_id'], 'tokens': count,
                'valid': int(enhanced['span_mask'][i].sum()),
                'positive': int((enhanced['labels'].reshape(len(batch), -1)[i] > 0).sum())}, sort_keys=True).encode())
        counts['rows'] += len(batch)
        counts['positive_instances'] += int((enhanced['labels'] > 0).sum())
        counts['valid_target_spans'] += int(enhanced['span_mask'].sum())
    assert json.dumps(rows, sort_keys=True) == untouched
    assert dict(counts) == {'rows_with_context': 1121, 'encoded_tokens': 213323,
        'target_tokens': 122290, 'NO_instances': 87, 'rows': 1150,
        'positive_instances': 2236, 'valid_target_spans': 1125538}

    # Real library word extraction/span representation on deterministic CPU
    # embeddings, not a substitute for a full pretrained model forward.
    selected = [rows[i] for i in [0, 1, 30, max(range(len(rows)), key=lambda j: rows[j]['tokens'])]]
    plain = ner_context.collate(processor, selected, builder, context=False)
    extra = ner_context.collate(processor, selected, builder)
    head = SpanRepLayer(16, 12, config.span_mode).eval()
    target_outputs = []
    for batch in [plain, extra]:
        wm = batch['words_mask']
        embeddings = wm.unsqueeze(-1).float().expand(-1, -1, 16).clone()
        embeddings[wm == 0] = 10000.  # Context cannot become a target word slot.
        words, valid_words = extract_word_embeddings(embeddings, wm, batch['attention_mask'],
            len(selected), int(batch['text_lengths'].max()), 16, batch['text_lengths'])
        span = head(words, batch['span_idx']*batch['span_mask'].unsqueeze(-1))
        logits = torch.einsum('blkd,bcd->blkc', span, torch.ones(len(selected), 1, 16))
        assert logits.shape == (len(selected), int(batch['text_lengths'].max()), 12, 1)
        target_outputs.append((words, valid_words, logits))
    assert all(torch.equal(a, b) for a, b in zip(*target_outputs))
    logits = torch.randn_like(target_outputs[0][2])
    raw = decoder.decode([r['tokenized_text'] for r in selected], {1:'phenotype'}, logits,
                         flat_ner=False, threshold=.5)
    mapped = ner_context.decode(decoder, selected, logits, plain['span_mask'])
    assert mapped == raw
    poisoned = logits.clone()
    valid = plain['span_mask'].reshape(logits.shape[:-1]).bool()
    poisoned[~valid] = 10000.
    assert ner_context.decode(decoder, selected, poisoned, plain['span_mask']) == mapped
    global_output = ner_context.global_entities(selected, mapped)
    for row, entities in zip(selected, global_output):
        passage = builder.docs[row['pmc_id']]['full_text'][row['passage_index']]
        for ent in entities:
            start = ent['offset']-int(passage['offset'])
            assert passage['text'][start:start+ent['length']] == ent['text']
    train_logits = logits.detach().requires_grad_()
    base_loss = reference.masked_span_bce(train_logits, plain['labels'], plain['span_mask'])
    base_loss.backward()
    assert (train_logits.grad[~valid] == 0).all()
    altered_labels = plain['labels'].reshape(logits.shape).clone()
    altered_labels[~valid] = 1.
    changed_loss = reference.masked_span_bce(poisoned, altered_labels, plain['span_mask'])
    assert torch.equal(base_loss.detach(), changed_loss.detach())
    assert not torch.cuda.is_initialized()
    report = {'status': 'PASS_CPU_context_adapter_target_geometry_and_decode',
        'sources': sources, 'adapter_sha256': sha(ner_context.__file__), 'checker_sha256': sha(__file__),
        'reference_trainer_sha256': sha(a.reference_trainer),
        'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in {
            'train32': a.train, 'original1150': a.rows, 'config': directory/'gliner_config.json',
            'tokenizer': directory/'tokenizer.json'}.items()},
        'counts': dict(counts), 'row_digest_sha256': digest.hexdigest(),
        'checks': {'all1150_disabled_tensors_equal_old_collator': True,
            'all1150_enabled_target_geometry_labels_NO_and_input_row_bytes_unchanged': True,
            'all1150_inference_context_and_tensors_equal_without_labels': True,
            'all_context_attention_one_words_mask_zero': True,
            'all_actual_input_lengths_at_most384_no_truncation': True,
            'real_extract_word_embeddings_and_span_rep_target_shapes_unchanged': True,
            'real_decoder_random_target_logits_equal_original': True,
            'invalid_high_logits_removed_before_greedy': True,
            'global_offsets_match_original_passage_slices': True,
            'masked_BCE_value_invariant_and_invalid_gradient_zero': True},
        'shape_probe_rows': [r['row_id'] for r in selected], 'shape_probe_logits': list(logits.shape),
        'shape_probe_scope': 'Real library word extractor and span head, deterministic synthetic encoder embeddings; no RNN or pretrained transformer/full-model forward.',
        'limitations': ['No evidence of quality improvement.', 'Pretrained full-model/GPU compatibility and throughput remain untested.',
            'Context has encoder attention only; RNN and span head retain original targets.'],
        'versions': {k: importlib.metadata.version(k) for k in ['torch','transformers','gliner','tokenizers']},
        'cuda_initialized': False, 'checkpoint_opened': False, 'heldout_gold_read': False,
        'seconds': time.monotonic()-started}
    output = Path(a.output)
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'counts': dict(counts), 'seconds': report['seconds'],
                      'sha256': sha(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
