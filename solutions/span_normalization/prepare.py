"""Encode all passages with exact span alignment, no held-out annotations."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
from transformers import AutoTokenizer
from common import read_jsonl, write_json, write_jsonl, ref


def load_legacy(path):
    spec = importlib.util.spec_from_file_location('gp_legacy_linker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rows(docs, tokenizer, allowed, aliases, training, length=384, stride=64):
    rows = []; stats = Counter(); gold_keys = set(); aligned_keys = set()
    nested = set(); max_gold_tokens = 0
    for doc in docs:
        pmc = str(doc['pmc_id']); stats['documents'] += 1
        if not training:
            assert not any(k in doc for k in ('entities', 'association'))
        for pi, passage in enumerate(doc['full_text']):
            text = passage.get('text', ''); base = int(passage.get('offset', 0))
            if not text.strip():
                stats['blank_passages'] += 1; continue
            gold = set()
            if training:
                for e in doc.get('entities', []):
                    if str(e.get('note', '')).upper() == 'NO': continue
                    ids = [aliases.get(x.strip(), x.strip()) for x in str(e.get('identifier', '-1')).split(';')]
                    if not (str(e.get('identifier', '-1')) == '-1' or any(x in allowed for x in ids)):
                        continue
                    a = int(e['offset']) - base; b = a + int(e['length'])
                    if 0 <= a < b <= len(text):
                        assert text[a:b] == e['text'], (pmc, pi, a, b, 'gold coordinate mismatch')
                        gold.add((a, b)); gold_keys.add((pmc, pi, a, b))
                for a, b in gold:
                    if any((c, d) != (a, b) and c <= a and b <= d for c, d in gold):
                        nested.add((pmc, pi, a, b))
            enc = tokenizer(text, truncation=True, max_length=length, stride=stride,
                            return_overflowing_tokens=True, return_offsets_mapping=True)
            for ci, offsets in enumerate(enc['offset_mapping']):
                features = {k: enc[k][ci] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in enc}
                starts = {a: i for i, (a, b) in enumerate(offsets) if b > a}
                ends = {b: i for i, (a, b) in enumerate(offsets) if b > a}
                positives = []
                for a, b in sorted(gold):
                    if a in starts and b in ends and starts[a] <= ends[b]:
                        i, j = starts[a], ends[b]
                        assert text[offsets[i][0]:offsets[j][1]] == text[a:b]
                        positives.append([i, j]); aligned_keys.add((pmc, pi, a, b))
                        max_gold_tokens = max(max_gold_tokens, j - i + 1)
                row = {'row_id': f'{pmc}:{pi}:{ci}', 'pmc_id': pmc, 'passage_index': pi,
                       'passage_offset': base, 'features': features, 'offsets': offsets}
                if training: row['positive'] = positives
                rows.append(row); stats['windows'] += 1; stats['tokens'] += len(features['input_ids'])
                stats['positive_window_spans'] += len(positives)
    stats.update({'unique_positive_spans': len(gold_keys), 'aligned_unique_spans': len(aligned_keys),
                  'unrepresentable_unique_spans': len(gold_keys - aligned_keys),
                  'nested_positive_spans': len(nested), 'max_gold_wordpiece_length': max_gold_tokens})
    assert len(rows) == len({r['row_id'] for r in rows})
    return rows, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    for name in ('train', 'input', 'hpo', 'model', 'legacy-source', 'output'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args(); start = time.monotonic(); out = Path(a.output)
    train = read_jsonl(a.train); held = read_jsonl(a.input)
    assert len(train) == 32 and len(held) == 16
    train_ids = {str(x['pmc_id']) for x in train}; held_ids = {str(x['pmc_id']) for x in held}
    assert not train_ids & held_ids and len(train_ids) == 32 and len(held_ids) == 16
    legacy = load_legacy(a.legacy_source); _, allowed, aliases, version = legacy.parse_hpo(a.hpo)
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True)
    tr, ts = make_rows(train, tokenizer, allowed, aliases, True)
    va, vs = make_rows(held, tokenizer, allowed, aliases, False)
    write_jsonl(out / 'train-windows.jsonl', tr); write_jsonl(out / 'held-windows.jsonl', va)
    # Actual tokenizer proves nested spans can coexist in a single output matrix.
    toy = {'pmc_id': 'synthetic', 'full_text': [{'offset': 13, 'text': 'severe intellectual disability'}],
           'entities': [{'offset': 13, 'length': 30, 'text': 'severe intellectual disability', 'identifier': '-1'},
                        {'offset': 20, 'length': 23, 'text': 'intellectual disability', 'identifier': '-1'}]}
    toy_rows, toy_stats = make_rows([toy], tokenizer, allowed, aliases, True)
    assert len(toy_rows[0]['positive']) == 2 and toy_stats['nested_positive_spans'] == 1
    summary = {'status': 'ENCODED_COMPLETE', 'train': ts, 'held': vs, 'train_pmcs': sorted(train_ids),
               'held_pmcs': sorted(held_ids), 'hpo_version': version, 'training_only_gold': True,
               'configuration': {'max_length': 384, 'stride': 64, 'head_size': 64, 'epochs': 8,
                                 'learning_rate': 3e-5, 'batch_size': 16, 'seed': 20261024,
                                 'raw_pool_floor': -2.0, 'decision_threshold': 0.0},
               'nested_roundtrip': toy_stats, 'sources': {k: ref(getattr(a, k)) for k in ('train', 'input', 'hpo', 'legacy_source')},
               'windows': {'train': ref(out / 'train-windows.jsonl'), 'held': ref(out / 'held-windows.jsonl')},
               'seconds': time.monotonic() - start, 'tokenizer_class': type(tokenizer).__name__}
    write_json(out / 'preparation.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
