#!/usr/bin/env python3
"""Input-only GLiNER resource and coordinate pilot; no HPO linking or scoring."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def windows(text, splitter, tokenizer, prefix, max_tokens, overlap):
    words = list(splitter(text))
    start, covered = 0, set()
    while start < len(words):
        lo, hi = start+1, min(len(words), start+max_tokens)
        best = None
        while lo <= hi:
            end = (lo+hi)//2
            n = len(tokenizer(prefix+[w[0] for w in words[start:end]],
                              is_split_into_words=True, truncation=False, verbose=False)['input_ids'])
            if n <= max_tokens:
                best = end, n
                lo = end+1
            else:
                hi = end-1
        if best is None:
            raise ValueError('Single full word exceeds frozen token budget; no truncation permitted')
        end, n = best
        begin_char, end_char = words[start][1], words[end-1][2]
        assert all(text[a:b] == w for w, a, b in words[start:end])
        covered.update(range(start, end))
        yield {'start': begin_char, 'end': end_char, 'word_start': start,
               'word_end': end, 'word_count': end-start, 'tokens': n,
               'text': text[begin_char:end_char]}
        if end == len(words):
            break
        start = max(start+1, end-overlap)
    assert len(covered) == len(words)


from check_gliner_biomed import load
import check_gliner_biomed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir', 'tokenizer-dir', 'input', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    started = time.monotonic()
    docs = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    assert len(docs) == len({str(d['pmc_id']) for d in docs})
    assert all('entities' not in d and 'association' not in d for d in docs)
    model_dir, tokenizer_dir = Path(args.model_dir), Path(args.tokenizer_dir)
    model = load(model_dir, tokenizer_dir)
    labels, threshold = ['phenotype'], .5
    params = sum(p.numel() for p in model.parameters())
    assert 0 < params < 10_000_000_000
    splitter = model.data_processor.words_splitter
    tokenizer = model.data_processor.transformer_tokenizer
    prefix = [model.config.ent_token, 'phenotype', model.config.sep_token]
    chunks, stats = [], Counter()
    for doc in docs:
        for pi, passage in enumerate(doc['full_text']):
            text = passage['text']
            words = list(splitter(text))
            stats['source_words'] += len(words)
            for chunk in windows(text, splitter, tokenizer, prefix, 384, 12):
                real, _ = model.prepare_model_inputs([chunk['text']], labels)
                actual_tokens = int(real['attention_mask'].sum())
                assert actual_tokens == chunk['tokens'] <= 384
                assert int(real['words_mask'].max()) == chunk['word_count']
                assert int(real['text_lengths'][0]) == chunk['word_count']
                chunks.append(dict(chunk, pmc_id=doc['pmc_id'], passage_index=pi,
                                   passage_offset=int(passage['offset'])))
    stats['chunks'] = len(chunks)
    stats['documents'] = len(docs)
    stats['max_input_tokens'] = max((c['tokens'] for c in chunks), default=0)
    stats['max_words_per_window'] = max((c['word_count'] for c in chunks), default=0)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'configuration': vars(args), 'source_sha256': sha(__file__), 'loader_source_sha256': sha(check_gliner_biomed.__file__),
                'input_sha256': sha(args.input), 'label': labels, 'threshold': threshold,
                'flat_ner': False, 'overlap_words': 12, 'token_budget': 384,
                'model_parameters': params, 'strict_state_dict': True,
                'files': {str(p): sha(p) for p in [model_dir/'pytorch_model.bin', model_dir/'gliner_config.json',
                                                  tokenizer_dir/'added_tokens.json', tokenizer_dir/'special_tokens_map.json', tokenizer_dir/'tokenizer.json', tokenizer_dir/'spm.model',
                                                  tokenizer_dir/'tokenizer_config.json']},
                'versions': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'gliner', 'tokenizers', 'sentencepiece']},
                'stats': dict(stats), 'all_source_word_indices_covered': True,
                'all_actual_word_masks_and_token_counts_match_no_truncation': True,
                'gold_read': False, 'hpo_linking': False, 'prediction_scoring': False}
    (out/'windows.json').write_text(json.dumps([{k:v for k,v in c.items() if k != 'text'} for c in chunks], indent=2)+'\n')
    if args.preflight_only:
        manifest['preflight_seconds'] = time.monotonic()-started
        (out/'preflight.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'stage': 'cpu_preflight_pass', 'parameters': params, 'stats': dict(stats)}))
        return
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')
    begin = time.monotonic()
    by_doc = {str(d['pmc_id']): {} for d in docs}
    raw = []
    source_docs = {str(d['pmc_id']): d for d in docs}
    for ci, chunk in enumerate(chunks):
        entities = model.predict_entities(chunk['text'], labels, flat_ner=False, threshold=threshold)
        word_spans = list(splitter(chunk['text']))
        starts, ends = {w[1] for w in word_spans}, {w[2] for w in word_spans}
        for entity in entities:
            a, b = int(entity['start']), int(entity['end'])
            assert a in starts and b in ends and a < b
            assert chunk['text'][a:b] == entity['text']
            offset = chunk['passage_offset'] + chunk['start'] + a
            p = source_docs[str(chunk['pmc_id'])]['full_text'][chunk['passage_index']]
            assert p['text'][offset-int(p['offset']):offset-int(p['offset'])+b-a] == entity['text']
            row = dict(entity, offset=offset, length=b-a, pmc_id=chunk['pmc_id'], window_index=ci)
            raw.append(row)
            key = offset, b-a, entity['label']
            old = by_doc[str(chunk['pmc_id'])].get(key)
            if old is None or old['score'] < row['score']:
                by_doc[str(chunk['pmc_id'])][key] = row
        print(json.dumps({'completed': ci+1, 'chunks': len(chunks), 'raw_spans': len(raw)}), flush=True)
    path = out/'candidates.jsonl'
    path.write_text(''.join(json.dumps({'pmc_id':d['pmc_id'], 'candidates':sorted(by_doc[str(d['pmc_id'])].values(), key=lambda e:(e['offset'],e['length']))}, ensure_ascii=False)+'\n' for d in docs))
    (out/'raw-spans.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in raw))
    manifest.update({'inference_seconds':time.monotonic()-begin, 'total_seconds':time.monotonic()-started,
                     'gpu':torch.cuda.get_device_name(0), 'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'raw_spans':len(raw), 'unique_spans':sum(len(x) for x in by_doc.values()),
                     'all_returned_spans_match_full_words_and_original_coordinates':True,
                     'candidates_sha256':sha(path)})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'stage':'complete','inference_seconds':manifest['inference_seconds'],
                      'parameters':params,'unique_spans':manifest['unique_spans']}), flush=True)


if __name__ == '__main__':
    main()
