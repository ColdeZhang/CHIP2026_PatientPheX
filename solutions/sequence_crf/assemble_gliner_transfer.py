#!/usr/bin/env python3
"""Two fixed original-dev transfer arms: entities only and the existing relation LR."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

import run as base
import combine
import assemble_gliner_linked as original


def associations(doc, entities, vectorizer, model):
    rows, pairs = [], []
    values = {p['patient_id']: set() for p in doc.get('patient', [])}
    for entity in entities:
        for patient in doc.get('patient', []):
            rows.append(base.pair_features(doc, entity, patient))
            pairs.append((entity, patient))
    probs = model.predict_proba(vectorizer.transform(rows))[:, 1] if rows else []
    for (entity, patient), probability in zip(pairs, probs):
        if probability >= .55:
            values[patient['patient_id']].update(base.association_keys(entity))
    return [{'patient_id': p['patient_id'], 'phenotype': sorted(values[p['patient_id']])}
            for p in doc.get('patient', [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'input', 'hpo', 'base', 'base-manifest', 'candidates',
                 'candidate-manifest', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    hashes = {k: base.sha(v) for k, v in vars(args).items() if k != 'output_dir'}
    prior_source = json.loads(Path(args.base_manifest).read_text())
    source = json.loads(Path(args.candidate_manifest).read_text())
    for key in ['train', 'input', 'hpo']:
        assert hashes[key] == prior_source['inputs'][key] == source['inputs'][key]
    assert prior_source['config']['relation_mode'] == 'lr'
    assert prior_source['config']['relation_threshold'] == .55
    assert hashes['base'] == prior_source['output_sha256']
    assert hashes['candidates'] == source['output']['sha256']
    assert source['span_training_provenance']['transfer_explicitly_enabled']
    train, docs = base.read_jsonl(args.train), base.read_jsonl(args.input)
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert not {str(d['pmc_id']) for d in train} & {str(d['pmc_id']) for d in docs}
    priors = {str(d['pmc_id']): d for d in base.read_jsonl(args.base)}
    candidates = {str(d['pmc_id']): d['candidates'] for d in base.read_jsonl(args.candidates)}
    assert set(priors) == set(candidates) == {str(d['pmc_id']) for d in docs}
    linker = base.Linker(train, args.hpo)
    vectorizer, relation, relation_training = base.fit_relation(train, linker)
    _, allowed, _, _ = base.parse_hpo(args.hpo)
    allowed.discard('HP:0000118')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Use a new immutable output directory'
    counts = Counter()
    outputs = {'baseline.jsonl': [], 'entity-only.jsonl': [], 'full-lr.jsonl': []}
    for doc in docs:
        prior = priors[str(doc['pmc_id'])]
        assert associations(doc, prior['entities'], vectorizer, relation) == prior['association'], 'Existing LR baseline not reproduced'
        outputs['baseline.jsonl'].append(prior)
        entities = list(prior['entities'])
        for candidate in sorted(candidates[str(doc['pmc_id'])],
                                key=lambda e: (-e['score'], -e['length'], e['offset'])):
            counts['all_candidates'] += 1
            if candidate['probability'] < .65 or candidate['similarity'] < .92:
                counts['semantic_filter'] += 1
                continue
            ids = candidate['identifier'].split(';')
            if not ids or not all(identifier in allowed for identifier in ids):
                counts['no_valid_hpo'] += 1
                continue
            _, passage = base.passage_for(doc, candidate['offset'])
            local = candidate['offset']-int(passage.get('offset', 0))
            assert passage['text'][local:local+candidate['length']] == candidate['text']
            if combine.ontology.is_negated(passage['text'], local, local+candidate['length']):
                counts['negated'] += 1
                continue
            if any(original.overlap(candidate, e) for e in prior['entities']):
                counts['overlap_original'] += 1
                continue
            if any(original.overlap(candidate, e) for e in entities):
                counts['overlap_added'] += 1
                continue
            entities.append({k: candidate[k] for k in ['identifier', 'offset', 'length', 'text']}
                            | {'type': 'Phenotype', 'note': None})
            counts['accepted'] += 1
        entities.sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
        outputs['entity-only.jsonl'].append(dict(prior, entities=entities))
        outputs['full-lr.jsonl'].append(dict(prior, entities=entities,
            association=associations(doc, entities, vectorizer, relation)))
    for name, predictions in outputs.items():
        (out/name).write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in predictions))
    assert base.sha(out/'baseline.jsonl') == hashes['base']
    manifest = {'configuration': vars(args), 'inputs': hashes,
                'source_sha256': base.sha(__file__),
                'dependencies': {name: base.sha(module.__file__) for name, module in
                                 [('sequence_base', base), ('combine', combine),
                                  ('ontology', combine.ontology), ('original_assembly', original)]},
                'policy': {'probability': .65, 'similarity': .92, 'relation_threshold': .55,
                           'order': 'GLiNER score descending,length descending,offset ascending',
                           'arms': ['same augmented entities, unchanged prior relations',
                                    'same augmented entities, existing LR all-pairs recomputation'],
                           'baseline': 'train64 CRF+Sap LR; exact no-op relation and full-byte reconstruction'},
                'relation_training': relation_training, 'counts': dict(counts),
                'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'outputs': {name: base.sha(out/name) for name in outputs}, 'gold_read': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
