"""Full-coverage WordPiece windows and one owner per physical entity mention."""
import argparse
from bisect import bisect_left, bisect_right
import gzip
import hashlib
import json
from pathlib import Path
import time


def spans_for_length(length, width=510, stride=384):
    assert 0 < stride <= width and length > 0
    windows = []
    for start in range(0, length, stride):
        end = min(length, start + width)
        windows.append((start, end))
        if end == length:
            break
    return windows


def owner_for_span(span, windows):
    first, end = span
    owners = [(min(first - start, stop - end), -i, i)
              for i, (start, stop) in enumerate(windows) if start <= first < end <= stop]
    if not owners:
        raise ValueError(f'No complete window for token span {span}')
    return max(owners)[2]


def encode_document(row, tokenizer):
    tokenized = tokenizer(row['text'], add_special_tokens=False, return_offsets_mapping=True,
                          truncation=False, return_attention_mask=False, return_token_type_ids=False)
    ids, offsets = tokenized['input_ids'], tokenized['offset_mapping']
    windows = spans_for_length(len(ids))
    starts, ends = [p[0] for p in offsets], [p[1] for p in offsets]
    encoded = []
    for start, end in windows:
        encoded.append({'start': start, 'end': end,
                        'ids': [tokenizer.cls_token_id] + ids[start:end] + [tokenizer.sep_token_id],
                        'mentions': []})
    ownership = []
    for entity_id, entity in enumerate(row['entities']):
        for char_start, char_end in entity['spans']:
            first = bisect_right(ends, char_start)
            end = bisect_left(starts, char_end)
            assert 0 <= first < end <= len(ids)
            assert offsets[first][1] > char_start and offsets[end - 1][0] < char_end
            window_id = owner_for_span((first, end), windows)
            local_start = first - windows[window_id][0] + 1
            local_end = end - windows[window_id][0] + 1
            encoded[window_id]['mentions'].append([entity_id, local_start, local_end])
            ownership.append([entity_id, char_start, char_end, first, end, window_id])
    assert len(ownership) == row['unique_physical_mentions']
    assert len({tuple(x[:3]) for x in ownership}) == len(ownership)
    count = [0] * len(ids)
    for start, end in windows:
        for i in range(start, end):
            count[i] += 1
    assert min(count) >= 1
    # Recover the full token sequence independently from overlapping windows.
    restored = [None] * len(ids)
    for chunk in encoded:
        for i, token_id in enumerate(chunk['ids'][1:-1], chunk['start']):
            assert restored[i] is None or restored[i] == token_id
            restored[i] = token_id
    assert restored == ids
    out = {key: row[key] for key in ['pmc_id', 'pairs', 'keys']}
    out.update({'entities': [{'kind': e['kind'], 'id': e['id'], 'mentions': len(e['spans'])}
                             for e in row['entities']],
                'windows': encoded, 'token_length': len(ids), 'ownership': ownership,
                'token_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    if 'labels' in row:
        out['labels'] = row['labels']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    import transformers
    from transformers import AutoTokenizer
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = {'tokenizer_class': type(tokenizer).__name__, 'transformers': transformers.__version__,
            'window_tokens': 510, 'stride': 384, 'added_special_tokens': 0,
            'model': args.model, 'files': {}, 'splits': {}}
    for split in ['train', 'held']:
        src = Path(args.prepared) / f'{split}.jsonl'
        rows = [json.loads(line) for line in src.read_text().splitlines()]
        encoded = [encode_document(row, tokenizer) for row in rows]
        assert all(('labels' in row) == (split == 'train') for row in encoded)
        dest = out / f'{split}.json.gz'
        with gzip.open(dest, 'wt', encoding='utf-8') as stream:
            json.dump(encoded, stream, ensure_ascii=False)
        info['files'][split] = {'input_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                               'encoded_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                               'encoded_bytes': dest.stat().st_size}
        info['splits'][split] = {'documents': len(encoded),
            'tokens': sum(row['token_length'] for row in encoded),
            'windows': sum(len(row['windows']) for row in encoded),
            'encoded_tokens_with_overlap_and_special': sum(len(w['ids']) for row in encoded for w in row['windows']),
            'maximum_tokens': max(row['token_length'] for row in encoded),
            'maximum_windows': max(len(row['windows']) for row in encoded),
            'mentions': sum(len(row['ownership']) for row in encoded),
            'pairs': sum(len(row['pairs']) for row in encoded)}
    info['elapsed_seconds'] = time.monotonic() - started
    (out / 'encoding.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
