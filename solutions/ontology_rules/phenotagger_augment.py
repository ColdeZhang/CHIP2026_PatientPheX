#!/usr/bin/env python3
"""Fixed p>=.95 additions, preserving all frozen base entities and relations."""
import argparse
import copy
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import run as baseline
import enhance
from merge_semantic import assign
import merge_semantic


def main():
    parser = argparse.ArgumentParser()
    for key in ['input', 'base', 'scored', 'hpo', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    docs = baseline.read_jsonl(args.input); bases = {str(d['pmc_id']): d for d in baseline.read_jsonl(args.base)}
    assert all('entities' not in d and 'association' not in d for d in docs)
    allowed = baseline.load_hpo(args.hpo)[1]; candidates = defaultdict(list)
    with gzip.open(args.scored, 'rt') as handle:
        for row in map(json.loads, handle): candidates[str(row['pmc_id'])].append(row)
    assert set(bases) == set(candidates) == {str(d['pmc_id']) for d in docs}
    predictions = []; counts = Counter()
    for doc in docs:
        did = str(doc['pmc_id']); pred = copy.deepcopy(bases[did]); additions = []
        for row in sorted(candidates[did], key=lambda r: (-r['probability'], -r['length'], r['offset'], r['identifier'])):
            counts['candidates'] += 1
            if row['identifier'] not in allowed or row['identifier'] == baseline.ROOT:
                counts['reject_none_or_disallowed'] += 1; continue
            if row['probability'] < .95: counts['reject_probability'] += 1; continue
            if row['overlength']: counts['reject_overlength'] += 1; continue
            _, passage, start, end = enhance.location(doc, row['offset'], row['offset'] + row['length'])
            assert passage['text'][start:end] == row['text']
            if baseline.is_negated(passage['text'], start, end): counts['reject_negated'] += 1; continue
            if any(row['offset'] < e['offset'] + e['length'] and e['offset'] < row['offset'] + row['length']
                   for e in pred['entities']): counts['reject_overlap'] += 1; continue
            entity = {key: row[key] for key in ['identifier', 'offset', 'length', 'text']}
            entity.update(type='Phenotype', note=None)
            pred['entities'].append(entity); additions.append(entity)
        new_associations = {row['patient_id']: set(row['phenotype']) for row in assign(doc, additions)}
        for row in pred['association']:
            original = set(row['phenotype']); row['phenotype'] = sorted(original | new_associations[row['patient_id']])
            counts['added_relations'] += len(set(row['phenotype']) - original)
        counts['added_entities'] += len(additions)
        pred['entities'].sort(key=lambda row: (row['offset'], row['length'], row['identifier']))
        predictions.append(pred)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in predictions))
    sources = {name: baseline.sha256(path) for name, path in
               [('assembler', __file__), ('baseline', baseline.__file__), ('enhance', enhance.__file__),
                ('association', merge_semantic.__file__)]}
    report = {'source_sha256': sources, 'inputs': {key: baseline.sha256(getattr(args, key))
                for key in ['input', 'base', 'scored', 'hpo']}, 'counts': dict(counts),
              'policy': 'Fixed p>=.95 top1 valid HPO; reject >32 tokens/negated; descending p,length,offset,ID nonoverlap; preserve base entities and all base associations; only new evidence adds associations.',
              'output_sha256': baseline.sha256(out), 'input_annotations_used': False}
    out.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report))


if __name__ == '__main__': main()
