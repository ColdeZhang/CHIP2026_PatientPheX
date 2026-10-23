#!/usr/bin/env python3
"""Add only distinct-concept properly nested spans to a frozen entity pool.

The existing entities and associations are preserved. This input-only adapter
implements the fixed training-side control; it never loads evaluation labels.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import combine
import run as base


def overlap(left, right):
    return (left['offset'] < right['offset'] + right['length'] and
            right['offset'] < left['offset'] + left['length'])


def proper_containment(left, right):
    a, b = left['offset'], left['offset'] + left['length']
    c, d = right['offset'], right['offset'] + right['length']
    return (a, b) != (c, d) and ((a <= c and b >= d) or (c <= a and d >= b))


def may_append(candidate, selected):
    ids = set(candidate['identifier'].split(';'))
    collisions = [entity for entity in selected if overlap(candidate, entity)]
    return bool(collisions) and all(
        proper_containment(candidate, entity) and
        not ids.intersection(entity['identifier'].split(';'))
        for entity in collisions)


def keyed(path):
    rows = base.read_jsonl(path)
    result = {str(row['pmc_id']): row for row in rows}
    assert len(result) == len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'hpo', 'baseline', 'baseline-manifest', 'candidates',
                 'candidate-manifest', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    hashes = {name: base.sha(value) for name, value in vars(args).items()
              if name != 'output_dir'}
    previous = json.loads(Path(args.baseline_manifest).read_text())
    source = json.loads(Path(args.candidate_manifest).read_text())
    assert hashes['baseline'] in previous['outputs'].values()
    assert hashes['candidates'] == source['output']['sha256']
    assert hashes['candidates'] == previous['inputs']['candidates']
    for key in ['input', 'hpo']:
        assert hashes[key] == previous['inputs'][key] == source['inputs'][key]
    assert previous['inputs']['train'] == source['inputs']['train']
    assert previous['gold_read'] is False
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Use a new immutable output directory'
    docs, priors, candidates = (keyed(path) for path in
                                [args.input, args.baseline, args.candidates])
    assert set(docs) == set(priors) == set(candidates)
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    _, allowed, _, _ = base.parse_hpo(args.hpo)
    allowed.discard('HP:0000118')
    counts, result = Counter(), []
    for pmc, prior in priors.items():
        entities = list(prior['entities'])
        rows = candidates[pmc]['candidates']
        assert len({(c['offset'], c['length']) for c in rows}) == len(rows)
        for candidate in sorted(rows, key=lambda c: (-c['score'], -c['length'], c['offset'])):
            counts['all_candidates'] += 1
            assert all(math.isfinite(candidate[key]) for key in ['score', 'probability', 'similarity'])
            if candidate['probability'] < .65 or candidate['similarity'] < .92:
                counts['semantic_filter'] += 1
                continue
            ids = set(candidate['identifier'].split(';'))
            if not ids or not ids <= allowed:
                counts['invalid_hpo'] += 1
                continue
            _, passage = base.passage_for(docs[pmc], candidate['offset'])
            local = candidate['offset'] - int(passage.get('offset', 0))
            assert passage['text'][local:local+candidate['length']] == candidate['text']
            if combine.ontology.is_negated(passage['text'], local, local+candidate['length']):
                counts['negated'] += 1
                continue
            if not may_append(candidate, entities):
                counts['not_distinct_proper_nesting'] += 1
                continue
            entities.append({key: candidate[key] for key in ['identifier', 'offset', 'length', 'text']}
                            | {'type': 'Phenotype', 'note': None})
            counts['added_rows'] += 1
        entities.sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
        prediction = dict(prior, entities=entities)
        assert prediction['association'] == prior['association']
        assert all(entity in entities for entity in prior['entities'])
        result.append(prediction)
    pred = out/'pred.jsonl'
    pred.write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in result))
    manifest = {
        'configuration': vars(args), 'inputs': hashes,
        'source_sha256': base.sha(__file__),
        'dependencies': {name: base.sha(module.__file__) for name, module in
                         [('sequence', base), ('combine', combine), ('ontology', combine.ontology)]},
        'policy': {'probability_gte': .65, 'similarity_gte': .92,
                   'order': 'GLiNER score descending,length descending,offset ascending',
                   'new': 'At least one overlap; proper containment and disjoint HPO IDs against every overlapping selected entity',
                   'rejected': 'crossing, identical boundaries, shared HPO IDs, non-overlap',
                   'relations_unchanged': True, 'all_prior_entities_preserved': True},
        'counts': dict(counts), 'gold_read': False,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'output_sha256': base.sha(pred)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'counts': dict(counts), 'prediction_sha256': manifest['output_sha256']}))


if __name__ == '__main__':
    main()
