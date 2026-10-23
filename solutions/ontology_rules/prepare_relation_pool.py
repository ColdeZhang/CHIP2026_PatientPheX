#!/usr/bin/env python3
"""Make a high-recall evidence pool for relation reasoning, independent of entities.

This intermediate file is never an evaluated final entity prediction. Its
overlapping candidates supply possible concepts to the LLM, while merge_context
preserves the separately frozen entity output.
"""
import argparse
import gzip
import json
from pathlib import Path
import run as baseline


def main():
    ap = argparse.ArgumentParser()
    for k in ['input', 'base', 'candidates', 'output']:
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--probability', type=float, default=.4)
    ap.add_argument('--similarity', type=float, default=.85)
    a = ap.parse_args()
    docs = baseline.read_jsonl(a.input)
    base = {d['pmc_id']: d for d in baseline.read_jsonl(a.base)}
    with gzip.open(a.candidates, 'rt') as f:
        raw = {d['pmc_id']: d['candidates'] for d in map(json.loads, f)}
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert {d['pmc_id'] for d in docs} == set(base) == set(raw)
    output = []
    added = 0
    for doc in docs:
        bid = doc['pmc_id']
        pool = {(e['offset'], e['length'], e['identifier']): e for e in base[bid]['entities']}
        for c in raw[bid]:
            if c['probability'] < a.probability or c['similarity'] < a.similarity:
                continue
            key = c['offset'], c['length'], c['identifier']
            if key in pool:
                continue
            assert any(p['offset'] <= c['offset'] and
                       c['offset'] + c['length'] <= p['offset'] + len(p['text']) and
                       p['text'][c['offset']-p['offset']:c['offset']-p['offset']+c['length']] == c['text']
                       for p in doc['full_text']), 'candidate offset/text mismatch'
            pool[key] = {k: c[k] for k in ['identifier', 'offset', 'length', 'text']} | {'type': 'Phenotype', 'note': None}
            added += 1
        output.append(dict(base[bid], entities=sorted(pool.values(), key=lambda e: (e['offset'], e['length'], e['identifier']))))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
    meta = {'config': vars(a), 'source_sha256': baseline.sha256(__file__),
            'inputs': {k: baseline.sha256(getattr(a, k)) for k in ['input', 'base', 'candidates']},
            'added_candidate_spans': added, 'documents': len(docs),
            'purpose': 'relation evidence only; not final entity prediction',
            'input_annotations_used': False, 'output_sha256': baseline.sha256(path)}
    path.with_suffix('.manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
