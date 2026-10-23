"""Small deterministic CPU-only evidence; does not change features or models."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def main():
    start = time.monotonic()
    plan = read('plan.json')
    prep = read('prepared/manifest.json')
    new = read('new-embeddings/manifest.json')
    features = read('features/manifest.json')
    assert features['prepared_sha256'] == sha('prepared/manifest.json')
    assert features['new_embedding_sha256'] == sha('new-embeddings/manifest.json')
    assert features['plan_sha256'] == sha('plan.json')
    assert sha('request.json.gz') == plan['request_sha256']
    with gzip.open('request.json.gz', 'rt') as stream:
        request = json.load(stream)
    assert sha('prepared/alias-indices.json') == prep['outputs']['alias-indices.json']['sha256']
    aliases = read('prepared/alias-indices.json')
    names = {text: index for index, text in enumerate(request['phrases'])}
    rank = lambda pair: hashlib.sha256((pair[0] + '\0' + pair[1]).encode()).hexdigest()
    ordered = sorted(map(tuple, request['pairs']), key=rank)
    groups = {
        'sha_first16': ordered[:16],
        'compound_first3': [p for p in ordered if ';' in p[1]][:3],
        'nil_first3': [p for p in ordered if p[1] == '-1'][:3],
        'missing_public_first3': [p for p in ordered if any(not aliases.get(a) for a in p[1].split(';'))][:3],
    }
    selected = sorted({pair for values in groups.values() for pair in values}, key=rank)
    references = [prep['outputs']['known-vectors.npy'], prep['public_bank'], new['vectors']]
    for item in references:
        assert sha(item['path']) == item['sha256']
    known = np.load(references[0]['path'], mmap_mode='r', allow_pickle=False)
    bank = np.load(references[1]['path'], mmap_mode='r', allow_pickle=False)
    added = np.load(references[2]['path'], mmap_mode='r', allow_pickle=False)
    missing = read('prepared/missing.json')
    assert sha('prepared/missing.json') == new['missing_texts_sha256']
    new_index = {text: i for i, text in enumerate(missing)}
    actual = {}
    assert sha('features/lookup.jsonl.gz') == features['lookup']['sha256']
    with gzip.open('features/lookup.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['text'], row['identifier']
            if key in selected:
                actual[key] = row
    assert len(actual) == len(selected)
    cases = []
    for text, identifier in selected:
        query = added[new_index[text]] if text in new_index else known[names[text]]
        atoms, maxima = [], []
        for atom in identifier.split(';'):
            indices = aliases.get(atom, [])
            values = bank[indices]
            maximum = float((values @ query).max()) if indices else None
            maxima.append(maximum)
            atoms.append(dict(identifier=atom, indices=indices,
                              public_alias_texts=[request['bank_labels'][i] for i in indices],
                              bank_vectors=values.tolist(), maximum=maximum))
        observed = bool(maxima) and all(value is not None for value in maxima)
        score = min(maxima) if observed else 0.
        target = actual[text, identifier]
        assert target['public_sap_observed'] == observed
        assert abs(target['public_sap_cosine'] - score) <= 1e-7
        cases.append(dict(text=text, identifier=identifier, pair_sha256=rank((text, identifier)),
                          query_source='new799' if text in new_index else 'known',
                          query_index=new_index[text] if text in new_index else names[text],
                          query_vector=query.tolist(), atoms=atoms, actual=target))
    result = dict(source_sha256=sha(__file__), plan=ref('plan.json'),
                  feature_manifest=ref('features/manifest.json'),
                  prepared_manifest=ref('prepared/manifest.json'),
                  new_manifest=ref('new-embeddings/manifest.json'), vectors=references,
                  groups=groups, cases=cases, no_gold=True, no_model=True, no_GPU=True,
                  seconds=time.monotonic()-start, finished_at=datetime.now(timezone.utc).isoformat())
    assert result['seconds'] < 60
    with Path('numeric-audit-sample.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as stream:
            stream.write(json.dumps(result, ensure_ascii=False).encode())
    print(json.dumps(ref('numeric-audit-sample.json.gz')))


if __name__ == '__main__':
    main()
