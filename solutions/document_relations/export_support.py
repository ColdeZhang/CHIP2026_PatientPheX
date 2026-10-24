"""Export exact public candidate mentions and provenance for score fusion."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from prepare import read, sha, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preparation', required=True)
    ap.add_argument('--prepared-held', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    metadata = json.loads(Path(args.preparation).read_text())
    config = metadata['configuration']
    source = config['input']
    assert sha(source) == metadata['sources'][source]
    assert sha(args.prepared_held) == metadata['summaries']['held']['sha256']
    original = {str(row['pmc_id']): row for row in read(source)}
    assert all('association' not in row and 'entities' not in row for row in original.values())
    prepared = read(args.prepared_held)
    support = defaultdict(lambda: defaultdict(set))
    for index, cache in enumerate(config['public_cache']):
        assert sha(cache) == metadata['sources'][cache]
        for item in read(cache):
            if str(item['pmc_id']) in original:
                support[(str(item['pmc_id']), item['hpo_id'])][
                    (item['offset'], item['length'], item['text'])].add(index)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, pairs, physical = [], 0, 0
    for document in prepared:
        pmc = document['pmc_id']
        orig = original[pmc]
        passage_map, current = [], 0
        for passage in orig['full_text']:
            passage_map.append((passage['offset'], passage['text'], current))
            current += len(passage['text']) + 2
        for entity in document['entities']:
            if entity['kind'] != 'phenotype':
                continue
            mentions = []
            for (offset, length, text), source_ids in sorted(support[(pmc, entity['id'])].items()):
                matches = [(new + offset - old, new + offset - old + length)
                           for old, passage, new in passage_map
                           if 0 <= offset - old and offset - old + length <= len(passage)
                           and passage[offset - old:offset - old + length] == text]
                assert matches
                first, end = matches[0]
                assert document['text'][first:end] == text
                mentions.append({'original_offset': offset, 'length': length, 'text': text,
                                 'prepared_span': [first, end], 'public_source_indices': sorted(source_ids)})
            assert {tuple(m['prepared_span']) for m in mentions} == {tuple(span) for span in entity['spans']}
            rows.append({'concept_key': [pmc, entity['id']], 'mentions': mentions})
            physical += len(mentions)
        pairs += len(document['keys'])
    path = output / 'held-candidate-support.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    result = {'status': 'all_held_candidate_concepts_exact_public_provenance_no_gold',
              'documents': len(prepared), 'patient_concept_pairs': pairs, 'doc_concepts': len(rows),
              'source_mentions': physical,
              'sources': [{'index': i, 'path': p, 'sha256': metadata['sources'][p]}
                          for i, p in enumerate(config['public_cache'])],
              'input': {'path': source, 'sha256': metadata['sources'][source]},
              'prepared': {'path': args.prepared_held, 'sha256': sha(args.prepared_held)},
              'support': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)},
              'join': 'Raw logit key [pmc_id, patient_id, HPO] joins concept_key [pmc_id,HPO]; patient mentions remain exact provided input metadata. Same candidate support for unmarked and marker-v2.'}
    write(output / 'manifest.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
