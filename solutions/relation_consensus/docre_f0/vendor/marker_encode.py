"""Independent ATLOP boundary-marker representation over exact existing tokens.

One existing-vocabulary '*' is inserted at each distinct start/end boundary.
Entity hidden/attention queries use its starting marker. Owner windows still
contain the complete marked mention; original tokens are never dropped.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from encode import owner_for_span, spans_for_length


def restored_tokens(document):
    ids = [None] * document['token_length']
    for window in document['windows']:
        assert len(window['ids']) == window['end'] - window['start'] + 2
        for position, value in enumerate(window['ids'][1:-1], window['start']):
            assert ids[position] is None or ids[position] == value
            ids[position] = value
    assert all(value is not None for value in ids)
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == document['token_ids_sha256']
    return ids


def insert_markers(original, spans, marker):
    starts = {first for first, end in spans}
    ends = {end for first, end in spans}
    assert all(0 <= first < end <= len(original) for first, end in spans)
    expanded, inserted, start_positions, end_positions = [], [], {}, {}
    original_positions = []
    for position in range(len(original) + 1):
        if position in ends:
            end_positions[position] = len(expanded)
            inserted.append(len(expanded))
            expanded.append(marker)
        if position in starts:
            start_positions[position] = len(expanded)
            inserted.append(len(expanded))
            expanded.append(marker)
        if position < len(original):
            original_positions.append(len(expanded))
            expanded.append(original[position])
    inserted_set = set(inserted)
    assert [value for i, value in enumerate(expanded) if i not in inserted_set] == original
    assert [expanded[i] for i in original_positions] == original
    assert len(inserted) == len(starts) + len(ends)
    return expanded, inserted, start_positions, end_positions


def transform(document, marker):
    original = restored_tokens(document)
    spans = {(row[3], row[4]) for row in document['ownership']}
    expanded, inserted, starts, ends = insert_markers(original, spans, marker)
    windows = spans_for_length(len(expanded))
    cls, sep = document['windows'][0]['ids'][0], document['windows'][0]['ids'][-1]
    out = deepcopy(document)
    out['windows'] = [{'start': first, 'end': end, 'ids': [cls] + expanded[first:end] + [sep],
                       'mentions': []} for first, end in windows]
    out['ownership'] = []
    for entity, char_first, char_end, first, end, old_owner in document['ownership']:
        marked_first, marked_end = starts[first], ends[end] + 1
        owner = owner_for_span((marked_first, marked_end), windows)
        local = marked_first - windows[owner][0] + 1
        out['windows'][owner]['mentions'].append([entity, local, local + 1])
        out['ownership'].append([entity, char_first, char_end, marked_first, marked_end, owner])
        assert out['windows'][owner]['ids'][local] == marker
        assert windows[owner][0] <= marked_first < marked_end <= windows[owner][1]
    out['token_length'] = len(expanded)
    out['token_ids_sha256'] = hashlib.sha256(json.dumps(expanded).encode()).hexdigest()
    out['boundary_markers'] = {'token_id': marker, 'inserted_positions': inserted,
                               'original_token_length': len(original),
                               'original_token_ids_sha256': document['token_ids_sha256'],
                               'query': 'single_start_marker; full_marked_mention_in_same_owner'}
    assert restored_tokens(out) == expanded
    assert {tuple(row[:3]) for row in out['ownership']} == {tuple(row[:3]) for row in document['ownership']}
    for i, entity in enumerate(out['entities']):
        assert sum(query[0] == i for w in out['windows'] for query in w['mentions']) == entity['mentions']
    for name in ['pmc_id', 'pairs', 'keys', 'entities', 'labels']:
        assert (name in out) == (name in document)
        if name in out:
            assert out[name] == document[name]
    return out


def contracts():
    ids = [10, 20, 30, 40]
    expanded, inserted, starts, ends = insert_markers(ids, [(0, 1), (1, 3), (1, 3), (0, 4)], 99)
    assert expanded == [99, 10, 99, 99, 20, 30, 99, 40, 99]
    assert starts == {0: 0, 1: 3} and ends == {1: 2, 3: 6, 4: 8}
    assert len(inserted) == 5
    # Adjacent boundaries remain two markers; duplicates/nesting do not create
    # repeated markers at the same side, and a literal '*' in source is kept.
    assert insert_markers([99], [(0, 1)], 99)[0] == [99, 99, 99]
    assert owner_for_span((390, 520), spans_for_length(900)) == 1
    return {'adjacent_start_and_end_separate': True, 'shared_boundaries_deduplicated': True,
            'nested_and_duplicate_spans': True, 'literal_source_marker_preserved': True,
            'full_marked_span_window_owner': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoded', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    vocabulary = Path(args.vocab)
    assert hashlib.sha256(vocabulary.read_bytes()).hexdigest() == '79489a52be45e6fa033521e8ce8e4f62aedc0a742ee2aa6fc04667e5b0b1454d'
    tokens = vocabulary.read_text().splitlines()
    assert tokens.count('*') == 1
    marker = tokens.index('*')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'mechanism': 'ATLOP-style explicit boundary markers and start-marker entity queries',
              'marker': {'text': '*', 'id': marker, 'new_parameters': 0},
              'contracts': contracts(), 'splits': {},
              'scope': 'All original tokens/candidate keys/labels retained; no held gold read or quality call.'}
    for split in ['train', 'held']:
        src = Path(args.encoded) / (split + '.json.gz')
        rows = json.load(gzip.open(src, 'rt'))
        assert all(('labels' in row) == (split == 'train') for row in rows)
        marked = [transform(row, marker) for row in rows]
        data = gzip.compress(json.dumps(marked).encode(), mtime=0)
        dest = output / src.name
        dest.write_bytes(data)
        result['splits'][split] = {'input_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'output': {'path': str(dest), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()},
            'documents': len(rows), 'pairs': sum(len(row['pairs']) for row in rows),
            'old_tokens': sum(row['token_length'] for row in rows),
            'new_tokens': sum(row['token_length'] for row in marked),
            'old_windows': sum(len(row['windows']) for row in rows),
            'new_windows': sum(len(row['windows']) for row in marked),
            'maximum_windows': max(len(row['windows']) for row in marked),
            'maximum_tokens': max(row['token_length'] for row in marked),
            'original_token_reconstruction': 'all exact after deleting recorded inserted positions',
            'unique_owner_and_complete_marked_span': 'all pass'}
    (output / 'encoding.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
