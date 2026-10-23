#!/usr/bin/env python3
"""Fold-isolated occurrence bags for a patient-independent relation filter."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import diagnose_relation_bags as diagnostic


def bags(doc, prediction, terms, allowed):
    assert 'entities' not in doc and 'association' not in doc
    grouped = defaultdict(list)
    for e in prediction['entities']:
        concepts = diagnostic.atoms(e['identifier'], allowed)
        if not concepts:
            continue
        passages = [p for p in doc['full_text'] if int(p['offset']) <= e['offset']
                    and e['offset'] + e['length'] <= int(p['offset']) + len(p['text'])]
        assert len(passages) == 1
        p = passages[0]; offset = int(p['offset']); text = p['text']
        start, end = e['offset'] - offset, e['offset'] - offset + e['length']
        assert text[start:end] == e['text']
        left, right = max(0, start-220), min(len(text), end+220)
        mentions = set()
        for patient in doc['patient']:
            for m in patient.get('mention', []):
                a = int(m['offset'])-offset; b = a + int(m.get('length', len(m.get('text', ''))))
                if left <= a < b <= right:
                    mentions.add((a-left, b-left))
        occurrence = {'offset': e['offset'], 'length': e['length'],
                      'text': text[left:right], 'phenotype': [start-left, end-left],
                      'patients': sorted(mentions), 'section': str(p.get('section_type', ''))}
        for h in concepts:
            grouped[h].append(occurrence)
    rows = []
    for h, occurrences in sorted(grouped.items()):
        occurrences.sort(key=lambda o: (o['offset'], o['length']))
        assert len({(o['offset'], o['length']) for o in occurrences}) == len(occurrences)
        rows.append({'pmc_id': str(doc['pmc_id']), 'concept': h,
                     'name': terms[h]['name'], 'occurrences': occurrences})
    return rows


def save(path, rows):
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def main():
    ap = argparse.ArgumentParser()
    for k in ['train', 'input', 'base', 'base-manifest', 'hpo', 'output-dir']:
        ap.add_argument('--'+k, required=True)
    ap.add_argument('--candidate-dirs', nargs=3, required=True)
    args = ap.parse_args(); train = diagnostic.read(args.train); inputs = diagnostic.read(args.input)
    assert not set(train) & set(inputs)
    assert all('entities' not in d and 'association' not in d for d in inputs.values())
    terms, allowed, _, _ = diagnostic.ontology.load_hpo(args.hpo)
    training, seen, sources = [], set(), []
    for directory in args.candidate_dirs:
        directory = Path(directory); pred_path = directory/'candidates.jsonl'
        manifest = json.loads((directory/'candidates.manifest.json').read_text())
        assert manifest['output_sha256'] == diagnostic.sha(pred_path)
        config = manifest['config']
        for k in ['train', 'input', 'hpo', 'base', 'candidates', 'candidate_manifest']:
            assert manifest['inputs'][k] == diagnostic.sha(config[k])
        fold, generator, pred = diagnostic.read(config['input']), diagnostic.read(config['train']), diagnostic.read(pred_path)
        assert len(generator) == 32 and len(fold) == 16
        assert set(fold) == set(pred) and not set(fold) & set(generator)
        assert set(fold) | set(generator) == set(train) and not seen & set(fold)
        assert all(d == train[k] for k, d in generator.items())
        seen.update(fold)
        for did, doc in fold.items():
            assert doc == {k: v for k, v in train[did].items() if k not in ['entities', 'association']}
            positive = {h for _, h in diagnostic.links(train[did], allowed)}
            training.extend(dict(row, label=int(row['concept'] in positive)) for row in bags(doc, pred[did], terms, allowed))
        sources.append({'prediction': str(pred_path), 'prediction_sha256': diagnostic.sha(pred_path),
                        'manifest_sha256': diagnostic.sha(directory/'candidates.manifest.json'), 'inputs': manifest['inputs']})
    assert seen == set(train)
    base = diagnostic.read(args.base); assert set(base) == set(inputs)
    bm = json.loads(Path(args.base_manifest).read_text())
    assert bm['output_sha256'] == diagnostic.sha(args.base)
    assert bm['inputs']['input'] == diagnostic.sha(args.input)
    testing = [row for did, doc in inputs.items() for row in bags(doc, base[did], terms, allowed)]
    training.sort(key=lambda r: (r['pmc_id'], r['concept']))
    testing.sort(key=lambda r: (r['pmc_id'], r['concept']))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save(out/'train.jsonl.gz', training); save(out/'input.jsonl.gz', testing)
    result = {'config': vars(args), 'source_sha256': diagnostic.sha(__file__),
              'dependencies': {'diagnostic': diagnostic.sha(diagnostic.__file__), 'ontology': diagnostic.sha(diagnostic.ontology.__file__)},
              'inputs': {k: diagnostic.sha(getattr(args, k)) for k in ['train', 'input', 'base', 'base_manifest', 'hpo']},
              'training_sources': sources, 'train_documents': len(train), 'input_documents': len(inputs),
              'train_bags': len(training), 'positive': sum(r['label'] for r in training),
              'train_occurrences': sum(len(r['occurrences']) for r in training),
              'input_bags': len(testing), 'input_occurrences': sum(len(r['occurrences']) for r in testing),
              'outputs': {k: diagnostic.sha(out/k) for k in ['train.jsonl.gz', 'input.jsonl.gz']},
              'policy': 'All OOF predicted occurrences; doc-HPO any specified patient label; no per-occurrence positive labels; no patient IDs in model input; supplied mentions shared marker class; no NIL bags.',
              'input_annotations_used': False}
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['train_bags', 'positive', 'train_occurrences', 'input_bags', 'input_occurrences']}))


if __name__ == '__main__':
    main()
