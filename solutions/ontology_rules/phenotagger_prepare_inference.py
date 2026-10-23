#!/usr/bin/env python3
"""Normalize an already frozen, label-free candidate pool for HPO classification."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import nltk
from phenotagger_data import tokens
import phenotagger_data


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ['input', 'pool', 'pool-manifest', 'training-manifest', 'nltk-data', 'output-dir']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); nltk.data.path.insert(0, args.nltk_data)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    assert all('entities' not in d and 'association' not in d for d in docs)
    by_id = {str(d['pmc_id']): d for d in docs}
    training = json.loads(Path(args.training_manifest).read_text())
    pool_manifest = json.loads(Path(args.pool_manifest).read_text())
    assert set(by_id).isdisjoint(str(x) for x in training['train_ids'])
    assert pool_manifest['input_sha256'] == sha(args.input)
    assert pool_manifest['train_sha256'] == training['train_sha256']
    # The uncompressed frozen pool is bound by its original manifest.
    opener = gzip.open if str(args.pool).endswith('.gz') else open
    with opener(args.pool, 'rt') as handle: pool = [json.loads(line) for line in handle]
    assert len(pool) == pool_manifest['total']
    if not str(args.pool).endswith('.gz'):
        assert pool_manifest['output_sha256'] == sha(args.pool)
    result = []; cache = {}; keys = set()
    for row in pool:
        did = str(row['pmc_id']); offset = row['offset']; length = row['length']
        key = (did, offset, length); assert key not in keys; keys.add(key)
        passage = next(p for p in by_id[did]['full_text']
                       if p['offset'] <= offset and offset + length <= p['offset'] + len(p['text']))
        surface = passage['text'][offset - passage['offset']:offset - passage['offset'] + length]
        assert surface == row['span']
        if surface not in cache: cache[surface] = ' '.join(tokens(surface))
        result.append({'pmc_id': did, 'offset': offset, 'length': length, 'text': surface,
                       'model_text': cache[surface]})
    target = output / 'candidates.jsonl.gz'
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as z:
        with io.TextIOWrapper(z, encoding='utf-8') as handle:
            for row in result: handle.write(json.dumps(row) + '\n')
    manifest = {'source_sha256': sha(__file__), 'normalization_source_sha256': sha(phenotagger_data.__file__),
                'inputs': {key: sha(getattr(args, key)) for key in ['input', 'pool', 'pool_manifest', 'training_manifest']},
                'rows': len(result), 'unique_model_texts': len({r['model_text'] for r in result}),
                'input_ids': sorted(by_id), 'train_ids': training['train_ids'],
                'candidate_policy': pool_manifest['candidate_policy'], 'input_annotations_used': False,
                'normalization': 'same lower/Treebank tokenization with spaced hyphen/slash as distant training; no lookup label',
                'output_sha256': sha(target)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': manifest['rows'], 'unique': manifest['unique_model_texts']}))


if __name__ == '__main__': main()
