"""Training-only label strata for the actual fixed window support geometry."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def analyze(rows, training):
    counts = Counter()
    for document in rows:
        support = [set() for _ in document['entities']]
        for wi, window in enumerate(document['windows']):
            for entity, start, end in window['mentions']:
                support[entity].add(wi)
        for i, (patient, concept) in enumerate(document['pairs']):
            category = ('positive' if document['labels'][i] else 'negative') if training else 'unlabelled'
            counts[category + '_total'] += 1
            same_window = bool(support[patient] & support[concept])
            intersects = any(max(document['windows'][a]['start'], document['windows'][b]['start']) <
                             min(document['windows'][a]['end'], document['windows'][b]['end'])
                             for a in support[patient] for b in support[concept])
            counts[category + ('_same_owner_window' if same_window else '_different_owner_windows')] += 1
            counts[category + ('_overlapping_attention_support' if intersects else '_disjoint_attention_support')] += 1
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoded', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = {'scope': 'Exact input support geometry. Labels used for training32 strata only; no F1 gold.', 'splits': {}, 'sources': {}}
    for split in ['train', 'held']:
        path = Path(args.encoded) / (split + '.json.gz')
        with gzip.open(path, 'rt') as f:
            rows = json.load(f)
        assert all(('labels' in row) == (split == 'train') for row in rows)
        result['splits'][split] = analyze(rows, split == 'train')
        result['sources'][split] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
