#!/usr/bin/env python3
"""Keep ontology entities fixed and use an attributed frozen relation model."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    for name in ['source', 'pipeline', 'manifest', 'train', 'input', 'pred', 'hpo', 'output']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('frozen_relation', args.source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    origin = json.loads(Path(args.manifest).read_text())
    assert origin['source_sha256'] == module.sha(args.source)
    assert origin['train_sha256'] == module.sha(args.train)
    assert origin['input_sha256'] == module.sha(args.input)
    fingerprint = {k: origin[k] for k in ['train_sha256', 'hpo_sha256', 'source_sha256', 'iterations']}
    _, _, vectorizer, relation, _ = module.load_pipeline(
        args.pipeline, module.sha(args.train), module.sha(args.hpo), fingerprint)
    docs = {d['pmc_id']: d for d in module.read_jsonl(args.input)}
    predictions = module.read_jsonl(args.pred)
    assert set(docs) == {d['pmc_id'] for d in predictions}
    assert not set(docs) & {d['pmc_id'] for d in module.read_jsonl(args.train)}
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    output = []
    for pred in predictions:
        doc = docs[pred['pmc_id']]
        associations = {p['patient_id']: set() for p in doc.get('patient', [])}
        pairs = [(e, p) for e in pred['entities'] for p in doc.get('patient', [])]
        if pairs:
            x = vectorizer.transform([module.pair_features(doc, e, p) for e, p in pairs])
            for probability, (e, p) in zip(relation.predict_proba(x)[:, 1], pairs):
                if probability >= .55:
                    associations[p['patient_id']].update(module.association_keys(e))
        result = dict(pred, association=[{'patient_id': p['patient_id'],
            'phenotype': sorted(associations[p['patient_id']])} for p in doc.get('patient', [])])
        assert result['entities'] == pred['entities']
        output.append(result)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in output))
    manifest = {'config': vars(args), 'source_sha256': module.sha(__file__),
        'inputs': {name: module.sha(getattr(args, name)) for name in
                   ['source', 'pipeline', 'manifest', 'train', 'input', 'pred', 'hpo']},
        'entity_outputs_unchanged': True, 'relation_threshold': .55,
        'training_performed': False, 'input_annotations_used': False,
        'output_sha256': module.sha(dest)}
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
