#!/usr/bin/env python3
"""Deterministic fold0 real-surface adaptation rows; no tokenizer or training.

Keep the frozen HPO/None replay unchanged. Append 16 copies of each train48
positive normalized surface/ID whose label is unambiguous across HPO+train48.
The fixed cardinalities belong to this approved fold0 control, not other folds.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil

import phenotagger_data as parent_recipe


REPEAT_REAL = 16
EXPECTED = {'documents': 48, 'parent_rows': 117118, 'hpo_rows': 58559,
            'none_rows': 58559, 'classes': 19120, 'linked_unique_pairs': 1824,
            'real_unique_pairs': 1749, 'rows': 145102}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(parent_recipe.tokens(text))


def group_labels(pairs):
    grouped = defaultdict(set)
    for surface, label in pairs:
        grouped[surface].add(label)
    return grouped


def exclusion_reason(identifier, terms, root):
    if identifier == root:
        return 'root'
    if terms.get(identifier, {}).get('obsolete'):
        return 'obsolete'
    return 'outside_branch' if identifier in terms else 'unknown_identifier'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--parent-data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hpo', help='Optional location override; bytes must equal the frozen parent HPO SHA.')
    args = parser.parse_args()
    parent = Path(args.parent_data_dir)
    out = Path(args.output_dir)
    assert out.resolve() != parent.resolve(), 'Parent data must remain read-only'
    manifest_path = parent / 'manifest.json'
    parent_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for filename in ['rows.jsonl.gz', 'labels.json']:
        assert sha(parent / filename) == parent_manifest['outputs'][filename], filename
    assert sha(args.train) == parent_manifest['train_sha256'], 'Wrong train48 bytes'
    hpo_path = Path(args.hpo or parent_manifest['config']['hpo'])
    assert sha(hpo_path) == parent_manifest['hpo_sha256'], 'Wrong designated HPO bytes'
    assert sha(parent_recipe.__file__) == parent_manifest['source_sha256'], 'Normalization recipe changed'
    assert sha(parent_recipe.baseline.__file__) == parent_manifest['baseline_source_sha256'], 'HPO/ID parser changed'

    train = parent_recipe.baseline.read_jsonl(args.train)
    train_ids = sorted(doc['pmc_id'] for doc in train)
    assert len(train_ids) == len(set(train_ids)) == EXPECTED['documents']
    assert train_ids == parent_manifest['train_ids'], 'Different document membership/order policy'
    labels = json.loads((parent / 'labels.json').read_text(encoding='utf-8'))
    terms, allowed, alt_ids, version = parent_recipe.baseline.load_hpo(hpo_path)
    assert labels == ['HP:None'] + sorted(allowed - {parent_recipe.baseline.ROOT})
    assert len(labels) == EXPECTED['classes']
    label_set = set(labels)
    head = label_set - {'HP:None'}
    with gzip.open(parent / 'rows.jsonl.gz', 'rt', encoding='utf-8') as handle:
        replay = [json.loads(line) for line in handle]
    assert len(replay) == parent_manifest['rows'] == EXPECTED['parent_rows']
    assert all(row['label'] in label_set for row in replay)
    hpo_rows = [row for row in replay if row['label'] != 'HP:None']
    none_rows = [row for row in replay if row['label'] == 'HP:None']
    hpo_pairs = {(row['text'], row['label']) for row in hpo_rows}
    assert len(hpo_rows) == len(hpo_pairs) == EXPECTED['hpo_rows']
    assert len(none_rows) == EXPECTED['none_rows']
    none_surfaces = {row['text'] for row in none_rows}

    counts = Counter()
    excluded = {'positive': Counter(), 'no': Counter()}
    positive_all_pairs = set()
    no_pairs = set()
    provenance = defaultdict(set)
    occurrences = Counter()
    physical_spans = set()
    canonical_spans = defaultdict(set)
    for doc in train:
        for entity in doc.get('entities', []):
            counts['all_entity_records'] += 1
            is_no = str(entity.get('note', '')).upper() == 'NO'
            kind = 'no' if is_no else 'positive'
            counts[kind + '_records'] += 1
            surface = normalize(entity['text'])
            assert surface, 'Empty real normalized surface'
            parts_raw = parent_recipe.baseline.identifier_parts(entity.get('identifier', ''))
            parts = tuple(sorted(set(parts_raw)))
            counts[kind + '_atomic_id_occurrences'] += len(parts)
            counts[kind + '_compound_identifier_records'] += len(parts) > 1
            counts[kind + '_duplicate_atomic_ids_within_record'] += len(parts_raw) - len(parts)
            counts[kind + '_canonical_duplicate_ids_within_record'] += len(parts) - len({alt_ids.get(part, part) for part in parts})
            offset, length = entity['offset'], entity['length']
            passage = next((p for p in doc['full_text']
                            if p['offset'] <= offset and offset + length <= p['offset'] + len(p['text'])), None)
            assert passage is not None, 'Entity outside passage'
            assert passage['text'][offset - passage['offset']:offset - passage['offset'] + length] == entity['text'], 'Raw slice mismatch'
            span_key = (doc['pmc_id'], offset, length)
            if not is_no:
                physical_spans.add(span_key)
            for raw_identifier in parts:
                identifier = alt_ids.get(raw_identifier, raw_identifier)
                counts[kind + '_alt_id_remapped_occurrences'] += identifier != raw_identifier
                if is_no:
                    no_pairs.add((surface, identifier))
                else:
                    positive_all_pairs.add((surface, identifier))
                    canonical_spans[span_key].add(identifier)
                if identifier == '-1':
                    excluded[kind]['unlinked_minus1'] += 1
                    continue
                if identifier not in head:
                    excluded[kind][exclusion_reason(identifier, terms, parent_recipe.baseline.ROOT)] += 1
                    continue
                counts[kind + '_in_head_atomic_occurrences'] += 1
                if is_no:
                    continue
                pair = (surface, identifier)
                provenance[pair].add(doc['pmc_id'])
                occurrences[pair] += 1

    linked_pairs = set(provenance)
    assert len(linked_pairs) == EXPECTED['linked_unique_pairs']
    real_labels = group_labels(linked_pairs)
    union_labels = group_labels(hpo_pairs | linked_pairs)
    ambiguous_real = {surface for surface, values in real_labels.items() if len(values) > 1}
    ambiguous_union = {surface for surface in real_labels if len(union_labels[surface]) > 1}
    selected = sorted((surface, label) for surface, label in linked_pairs
                      if len(union_labels[surface]) == 1)
    assert len(selected) == EXPECTED['real_unique_pairs']
    intersections = {
        'all_linked_pairs_with_none_surface': sum(surface in none_surfaces for surface, _ in linked_pairs),
        'selected1749_pairs_with_none_surface': sum(surface in none_surfaces for surface, _ in selected),
        'all_positive_including_unlinked_and_out_of_head_surfaces_with_none': len({s for s, _ in positive_all_pairs} & none_surfaces),
        'positive_minus1_surfaces_with_none': len({s for s, h in positive_all_pairs if h == '-1'} & none_surfaces),
        'no_surfaces_with_none': len({s for s, _ in no_pairs} & none_surfaces),
    }
    assert all(value == 0 for value in intersections.values()), intersections
    for kind in ['positive', 'no']:
        assert counts[kind + '_atomic_id_occurrences'] == counts[kind + '_in_head_atomic_occurrences'] + sum(excluded[kind].values())

    real_rows = [{'text': surface, 'label': label, 'source': 'train_real_positive'}
                 for surface, label in selected]
    real_evidence = [dict(row, pmc_ids=sorted(provenance[(row['text'], row['label'])]),
                          train_atomic_occurrences=occurrences[(row['text'], row['label'])])
                     for row in real_rows]
    # Objects in replay are neither edited nor reweighted; their original order
    # is retained. Each of 16 blocks contains sorted unique real pairs once.
    rows = replay + real_rows * REPEAT_REAL
    assert len(rows) == EXPECTED['rows']
    assert rows[:len(replay)] == replay
    assert Counter((r['text'], r['label']) for r in rows[len(replay):]) == Counter({pair: REPEAT_REAL for pair in selected})

    out.mkdir(parents=True, exist_ok=True)
    output_names = ['rows.jsonl.gz', 'labels.json', 'real-positives.jsonl', 'manifest.json']
    assert not any((out / name).exists() for name in output_names), 'Refuse to replace existing artifacts'
    with (out / 'rows.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='\n') as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copyfile(parent / 'labels.json', out / 'labels.json')
    assert sha(out / 'labels.json') == sha(parent / 'labels.json')
    with (out / 'real-positives.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for row in real_evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts.update(positive_unique_spans=len(physical_spans),
                  positive_unique_spans_multiple_ids=sum(len(ids) > 1 for ids in canonical_spans.values()),
                  linked_unique_pairs=len(linked_pairs), linked_unique_surfaces=len(real_labels),
                  linked_unique_classes=len({h for _, h in linked_pairs}),
                  selected_unique_pairs=len(selected), selected_unique_classes=len({h for _, h in selected}),
                  selected_pairs_already_hpo=sum(pair in hpo_pairs for pair in selected),
                  selected_pairs_not_hpo=sum(pair not in hpo_pairs for pair in selected),
                  selected_real_rows=REPEAT_REAL * len(selected),
                  parent_hpo_rows=len(hpo_rows), parent_none_rows=len(none_rows),
                  real_ambiguous_surfaces=len(ambiguous_real),
                  real_pairs_on_ambiguous_real_surfaces=sum(s in ambiguous_real for s, _ in linked_pairs),
                  union_ambiguous_real_surfaces=len(ambiguous_union),
                  excluded_real_pairs_on_union_ambiguous_surfaces=sum(s in ambiguous_union for s, _ in linked_pairs))
    result = {
        'config': vars(args), 'source_sha256': sha(__file__),
        'dependencies_sha256': {'phenotagger_data.py': sha(parent_recipe.__file__),
                                'ontology_rules/run.py': sha(parent_recipe.baseline.__file__)},
        'train_ids': train_ids, 'train_sha256': parent_manifest['train_sha256'],
        'hpo_sha256': parent_manifest['hpo_sha256'], 'hpo_version': version,
        'parent_data_manifest_sha256': sha(manifest_path),
        'parent_outputs': dict(parent_manifest['outputs']),
        'classes_with_reject': len(labels), 'rows': len(rows),
        'negative_rows': len(none_rows), 'real_repeat': REPEAT_REAL,
        'counts': dict(counts), 'excluded_atomic_occurrences': {key: dict(value) for key, value in excluded.items()},
        'none_exact_surface_intersections': intersections,
        'normalization': 'Exact parent_recipe.tokens; original real surface only, no extra lemma forms.',
        'positive_policy': 'note != NO; expand/deduplicate atomic IDs; official valid alt_id canonicalization; fixed-head linked positives only.',
        'unlinked_and_no_policy': 'Exclude -1 and NO from added positives; never convert them into HP:None.',
        'ambiguity_policy': 'Exclude every added real pair on a surface with multiple labels in HPO+train48 union. Preserve all parent rows, including existing HPO ambiguity.',
        'ordering': 'Original parent replay in original order, then 16 blocks of unique real pairs sorted by (normalized surface, canonical ID).',
        'replay_preserved': True, 'labels_bytes_preserved': True,
        'gzip': {'mtime': 0, 'filename': ''},
        'tokenization': 'Not performed here. Downstream retains parent truncation=True/max_length32 and must audit length/collisions; no row deletion.',
        'heldout_used': False, 'gpu_used': False,
        'outputs': {name: sha(out / name) for name in output_names if name != 'manifest.json'},
    }
    write_json(out / 'manifest.json', result)
    print(json.dumps({'rows': len(rows), 'real_unique_pairs': len(selected),
                      'none_exact_surface_intersections': intersections,
                      'outputs': result['outputs'], 'manifest_sha256': sha(out / 'manifest.json')}))


if __name__ == '__main__':
    main()
