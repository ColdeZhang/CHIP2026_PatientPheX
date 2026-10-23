#!/usr/bin/env python3
"""Official-training-only GLiNER span labels and actual tensor alignment audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

from run_gliner_pilot import load, sha, windows
import run_gliner_pilot


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'excluded-input', 'model-dir', 'tokenizer-dir', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--nested-train')
    parser.add_argument('--training-part', choices=['train48', 'train64'], default='train48')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    random.seed(20261023)
    torch.manual_seed(20261023)
    start = time.monotonic()
    docs, excluded = read(args.train), read(args.excluded_input)
    assert bool(args.nested_train) == (args.training_part == 'train48')
    nested = read(args.nested_train) if args.nested_train else []
    ids = {str(d['pmc_id']) for d in docs}
    excluded_ids = {str(d['pmc_id']) for d in excluded}
    nested_ids = {str(d['pmc_id']) for d in nested}
    assert len(ids) == len(docs) == {'train48': 48, 'train64': 64}[args.training_part]
    assert len(nested_ids) == len(nested) == (32 if args.training_part == 'train48' else 0)
    assert nested_ids < ids and not ids & excluded_ids
    assert all('entities' not in d and 'association' not in d for d in excluded)
    full_by_id = {str(d['pmc_id']): d for d in docs}
    assert all(d == full_by_id[str(d['pmc_id'])] for d in nested)
    model = load(Path(args.model_dir), Path(args.tokenizer_dir))
    processor = model.data_processor
    splitter, tokenizer = processor.words_splitter, processor.transformer_tokenizer
    prefix = [model.config.ent_token, 'phenotype', model.config.sep_token]
    rows, totals, per_doc = [], Counter(), []
    for doc in docs:
        count = Counter()
        positives = {(int(e['offset']), int(e['length'])) for e in doc['entities']
                     if str(e.get('note') or '').upper() != 'NO'}
        negatives = {(int(e['offset']), int(e['length'])) for e in doc['entities']
                     if str(e.get('note') or '').upper() == 'NO'}
        # Do not silently turn a conflicting positive/negative annotation into a target.
        assert not positives & negatives
        count.update(positive_unique_spans=len(positives), no_unique_spans=len(negatives))
        located, represented = set(), set()
        for pi, passage in enumerate(doc['full_text']):
            text, offset = passage['text'], int(passage['offset'])
            words = list(splitter(text))
            starts = {a: i for i, (_, a, _) in enumerate(words)}
            ends = {b: i for i, (_, _, b) in enumerate(words)}
            eligible = {}
            for span in positives:
                a, b = span[0]-offset, sum(span)-offset
                if not 0 <= a < b <= len(text):
                    continue
                assert span not in located
                located.add(span)
                if a not in starts or b not in ends:
                    count['positive_word_boundary_mismatch'] += 1
                    continue
                if ends[b]-starts[a]+1 > model.config.max_width:
                    count['positive_above_max_width'] += 1
                    continue
                eligible[span] = starts[a], ends[b]
            for wi, chunk in enumerate(windows(text, splitter, tokenizer, prefix, 384, 12)):
                first, last = chunk['word_start'], chunk['word_end']
                target, spans, no_indices = [], [], []
                for span, (a, b) in sorted(eligible.items()):
                    if first <= a <= b < last:
                        target.append([a-first, b-first, 'phenotype'])
                        spans.append(list(span))
                        represented.add(span)
                chunk_words = words[first:last]
                for span in negatives:
                    a, b = span[0]-offset, sum(span)-offset
                    if a in starts and b in ends and first <= starts[a] <= ends[b] < last:
                        width = ends[b]-starts[a]+1
                        if width <= model.config.max_width:
                            no_indices.append((starts[a]-first)*model.config.max_width+width-1)
                row = {'row_id': f'{doc["pmc_id"]}:{pi}:{wi}', 'pmc_id': str(doc['pmc_id']),
                       'passage_index': pi, 'window_start': offset+chunk['start'],
                       'text': chunk['text'], 'tokenized_text': [w[0] for w in chunk_words],
                       'word_offsets': [[offset+a, b-a] for _, a, b in chunk_words],
                       'ner': target, 'positive_spans': spans, 'label': ['phenotype'],
                       'tokens': chunk['tokens'], 'no_span_indices': no_indices}
                raw = processor.collate_raw_batch([row], entity_types=[['phenotype']])
                tensors = processor.collate_fn(raw, prepare_labels=True)
                assert int(tensors['attention_mask'].sum()) == row['tokens'] <= 384
                assert int(tensors['words_mask'].max()) == len(chunk_words)
                assert int(raw['seq_length'][0]) == len(chunk_words)
                labels = tensors['labels'].reshape(-1)
                expected = {a*model.config.max_width+b-a for a, b, _ in target}
                actual = set(torch.where(labels > 0)[0].tolist())
                assert actual == expected
                assert all(float(labels[index]) == 0 for index in no_indices)
                assert not (labels[~raw['span_mask'].reshape(-1)] > 0).any()
                count['positive_window_instances'] += len(target)
                count['no_window_instances'] += len(no_indices)
                count['valid_candidate_span_instances'] += int(raw['span_mask'].sum())
                count['windows_without_positive'] += not target
                rows.append(row)
        assert located == positives
        count['represented_unique_positive_spans'] = len(represented)
        count['positive_window_boundary_loss'] = (len(positives)-len(represented)
                                                -count['positive_word_boundary_mismatch']
                                                -count['positive_above_max_width'])
        assert count['positive_window_boundary_loss'] >= 0
        totals.update(count)
        per_doc.append({'pmc_id': str(doc['pmc_id']), 'counts': dict(count)})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {args.training_part: rows}
    if nested_ids:
        datasets.update(nested_train32=[r for r in rows if r['pmc_id'] in nested_ids],
                        nested_validation16=[r for r in rows if r['pmc_id'] not in nested_ids])
    artifacts = {}
    for name, values in datasets.items():
        path = out/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
        artifacts[name] = {'path': str(path), 'rows': len(values), 'sha256': sha(path),
                           'bytes': path.stat().st_size, 'pmc_ids': sorted({r['pmc_id'] for r in values})}
    report = {'scope': f'official {args.training_part} span alignment, no excluded-input gold',
              'configuration': vars(args), 'source_sha256': sha(__file__),
              'pilot_dependency_sha256': sha(run_gliner_pilot.__file__),
              'files': {k: sha(getattr(args, k)) for k in ['train', 'excluded_input', 'nested_train'] if getattr(args, k)},
              'model_weights_sha256': sha(Path(args.model_dir)/'pytorch_model.bin'),
              'model_parameters': sum(p.numel() for p in model.parameters()),
              'totals': dict(totals), 'per_document': per_doc, 'artifacts': artifacts,
              'maximum_tokens': max(r['tokens'] for r in rows),
              'all_actual_labels_word_masks_lengths_and_negative_targets_pass': True,
              'seconds': time.monotonic()-start,
              'limitations': ['Positive HPO/compound/NIL spans collapse to one phenotype target; no HPO training target.',
                              'NO spans are negative candidates, not a second output label.',
                              'Unrepresentable word-boundary/width spans counted, never snapped or exported as invented gold.',
                              'Nested validation labels are audit-only: training must select nested_train32, never train48 when scoring those 16.']}
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'windows': len(rows), 'counts': dict(totals), 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
