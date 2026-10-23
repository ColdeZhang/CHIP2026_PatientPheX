#!/usr/bin/env python3
"""Fixed GLiNER/SapBERT additions; preserve base entities and use same patient rules."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

import run as base
import combine


def overlap(a, b):
    return a['offset'] < b['offset']+b['length'] and b['offset'] < a['offset']+a['length']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'input', 'hpo', 'base', 'base-manifest', 'candidates',
                 'candidate-manifest', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    sources = {name: base.sha(getattr(args, name)) for name in
               ['train', 'input', 'hpo', 'base', 'base_manifest', 'candidates', 'candidate_manifest']}
    prior_source = json.loads(Path(args.base_manifest).read_text())
    candidate_source = json.loads(Path(args.candidate_manifest).read_text())
    for name in ['train', 'input', 'hpo']:
        assert sources[name] == prior_source['inputs'][name] == candidate_source['inputs'][name]
    assert sources['base'] == prior_source['output_sha256']
    assert sources['candidates'] == candidate_source['output']['sha256']
    docs, train = base.read_jsonl(args.input), base.read_jsonl(args.train)
    assert all('entities' not in d and 'association' not in d for d in docs)
    assert not {str(d['pmc_id']) for d in docs} & {str(d['pmc_id']) for d in train}
    priors = {str(d['pmc_id']): d for d in base.read_jsonl(args.base)}
    candidates = {str(d['pmc_id']): d['candidates'] for d in base.read_jsonl(args.candidates)}
    assert set(priors) == set(candidates) == {str(d['pmc_id']) for d in docs}
    _, allowed, _, _ = base.parse_hpo(args.hpo)
    allowed.discard('HP:0000118')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Use a new output directory'
    counts = Counter()
    with (out/'baseline.jsonl').open('w') as control, (out/'augmented.jsonl').open('w') as result:
        for doc in docs:
            prior = priors[str(doc['pmc_id'])]
            assert combine.assign(doc, prior['entities']) == prior['association'], 'Baseline rules changed'
            control.write(json.dumps(prior, ensure_ascii=False)+'\n')
            entities = list(prior['entities'])
            for candidate in sorted(candidates[str(doc['pmc_id'])],
                                    key=lambda x: (-x['score'], -x['length'], x['offset'])):
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
                if any(overlap(candidate, e) for e in prior['entities']):
                    counts['overlap_original'] += 1
                    continue
                if any(overlap(candidate, e) for e in entities):
                    counts['overlap_added'] += 1
                    continue
                entities.append({k: candidate[k] for k in ['identifier', 'offset', 'length', 'text']}
                                | {'type': 'Phenotype', 'note': None})
                counts['accepted'] += 1
            entities.sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
            prediction = dict(prior, entities=entities, association=combine.assign(doc, entities))
            result.write(json.dumps(prediction, ensure_ascii=False)+'\n')
    manifest = {'configuration': vars(args), 'inputs': sources,
                'source_sha256': base.sha(__file__), 'dependencies':
                    {'sequence_base': base.sha(base.__file__), 'combine': base.sha(combine.__file__),
                     'ontology': base.sha(combine.ontology.__file__)},
                'policy': {'probability': .65, 'similarity': .92,
                           'overlap_order': 'GLiNER score descending,length descending,offset ascending',
                           'existing_entities': 'unchanged', 'relation': 'same fixed rules',
                           'unmapped_new_spans': 'discard; preserve original NIL'},
                'counts': dict(counts), 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'outputs': {name: base.sha(out/name) for name in ['baseline.jsonl', 'augmented.jsonl']},
                'gold_read': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
