"""Build public-candidate document relation inputs without held-out labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def relation_set(doc):
    pairs = set()
    for row in doc.get('association', []):
        for value in row['phenotype']:
            parts = value.split(';') if re.fullmatch(r'HP:\d{7}(;HP:\d{7})+', value) else [value]
            pairs.update((str(row['patient_id']), part) for part in parts)
    return pairs


def build(doc, candidates, training):
    # Preserve every passage in its provided order. Offset mappings do not assume
    # that gaps or repeated offsets can be reconstructed by padding characters.
    pieces, maps, cursor = [], [], 0
    for passage in doc['full_text']:
        text = passage['text']
        pieces.append(text)
        maps.append((int(passage['offset']), text, cursor))
        cursor += len(text) + 2
    body = '\n\n'.join(pieces)

    def locate(item):
        start, length, text = int(item['offset']), int(item['length']), item['text']
        matches = []
        for offset, passage, new_start in maps:
            local = start - offset
            if 0 <= local and local + length <= len(passage) and passage[local:local + length] == text:
                matches.append((new_start + local, new_start + local + length))
        if not matches:
            raise ValueError(f"Unmapped mention {doc['pmc_id']} {start}:{length}")
        # Repeated source passages do not create independent physical mentions.
        return matches[0]

    patients = [str(p['patient_id']) for p in doc['patient']]
    assert len(patients) == len(set(patients))
    concepts = sorted(candidates)
    entities, mention_count = [], 0
    for patient in doc['patient']:
        spans = sorted({locate(m) for m in patient['mention']})
        assert spans, 'A supplied patient has no mapped mention'
        entities.append({'kind': 'patient', 'id': str(patient['patient_id']), 'spans': spans})
        mention_count += len(spans)
    for concept in concepts:
        spans = sorted({locate(m) for m in candidates[concept].values()})
        assert spans
        entities.append({'kind': 'phenotype', 'id': concept, 'spans': spans})
        mention_count += len(spans)
    pairs = [[p, len(patients) + c] for p in range(len(patients)) for c in range(len(concepts))]
    keys = [[str(doc['pmc_id']), patients[p], concepts[t - len(patients)]] for p, t in pairs]
    result = {'pmc_id': str(doc['pmc_id']), 'text': body, 'entities': entities,
              'pairs': pairs, 'keys': keys, 'original_passages': len(maps),
              'unique_physical_mentions': mention_count}
    if training:
        gold = relation_set(doc)
        result['labels'] = [int((key[1], key[2]) in gold) for key in keys]
        result['training_gold_pairs'] = len(gold)
        result['training_reachable_positives'] = sum(result['labels'])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--public-cache', action='append', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    train, held = read(args.train), read(args.input)
    train_ids = {str(d['pmc_id']) for d in train}
    held_ids = {str(d['pmc_id']) for d in held}
    assert len(train_ids) == len(train) and len(held_ids) == len(held)
    assert train_ids.isdisjoint(held_ids)
    assert all('entities' not in d and 'association' not in d for d in held)
    candidates = defaultdict(lambda: defaultdict(dict))
    source_counts = {}
    for path in args.public_cache:
        count = 0
        for row in read(path):
            pmc = str(row['pmc_id'])
            if pmc not in train_ids | held_ids:
                continue
            concept = str(row['hpo_id'])
            assert re.fullmatch(r'HP:\d{7}', concept), concept
            key = (int(row['offset']), int(row['length']), row['text'])
            candidates[pmc][concept][key] = {k: row[k] for k in ['offset', 'length', 'text']}
            count += 1
        source_counts[path] = count
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, docs, is_train in [('train', train, True), ('held', held, False)]:
        rows = [build(doc, candidates[str(doc['pmc_id'])], is_train) for doc in docs]
        path = output / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        summaries[name] = {'documents': len(rows), 'characters': sum(len(x['text']) for x in rows),
                           'pairs': sum(len(x['pairs']) for x in rows),
                           'concepts': sum(sum(e['kind'] == 'phenotype' for e in x['entities']) for x in rows),
                           'unique_physical_mentions': sum(x['unique_physical_mentions'] for x in rows),
                           'sha256': sha(path), 'bytes': path.stat().st_size}
        if is_train:
            summaries[name]['positive_pairs'] = sum(sum(x['labels']) for x in rows)
            summaries[name]['training_gold_pairs'] = sum(x['training_gold_pairs'] for x in rows)
    write(output / 'preparation.json', {'status': 'complete_no_held_gold', 'configuration': vars(args),
          'train_ids': sorted(train_ids), 'held_ids': sorted(held_ids), 'source_counts': source_counts,
          'sources': {path: sha(path) for path in [args.train, args.input, *args.public_cache, __file__]},
          'summaries': summaries, 'policy': 'Only published fixed public-model candidates, no task-trained lower layer; no candidate cut or held label use.'})
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
