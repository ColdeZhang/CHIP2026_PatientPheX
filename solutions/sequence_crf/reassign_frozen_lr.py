#!/usr/bin/env python3
"""Apply the unchanged train-fold LR to a frozen, additive entity-pool control."""
import argparse
import datetime
import json
from pathlib import Path

import run as base
import assemble_gliner_transfer as transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'input', 'hpo', 'baseline', 'candidates',
                 'candidate-manifest', 'handoff', 'output-dir']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    hashes = {k: base.sha(v) for k, v in vars(args).items() if k != 'output_dir'}
    handoff = json.loads(Path(args.handoff).read_text())
    assert handoff['gold_read'] is False and handoff['frozen_at_utc']
    for name in ['train', 'input', 'hpo', 'baseline', 'candidates', 'candidate_manifest']:
        assert hashes[name] == handoff['sha256'][name], name
    train, docs = base.read_jsonl(args.train), base.read_jsonl(args.input)
    prior_rows, candidate_rows = base.read_jsonl(args.baseline), base.read_jsonl(args.candidates)
    priors = {str(d['pmc_id']): d for d in prior_rows}
    candidates = {str(d['pmc_id']): d for d in candidate_rows}
    assert len(priors) == len(prior_rows) == len(docs) == len(candidates) == len(candidate_rows)
    assert set(priors) == set(candidates) == {str(d['pmc_id']) for d in docs}
    assert not set(priors) & {str(d['pmc_id']) for d in train}
    assert all('entities' not in d and 'association' not in d for d in docs)
    linker = base.Linker(train, args.hpo)
    vectorizer, relation, training = base.fit_relation(train, linker)
    predictions, added = [], 0
    for doc in docs:
        prior, candidate = priors[str(doc['pmc_id'])], candidates[str(doc['pmc_id'])]
        assert transfer.associations(doc, prior['entities'], vectorizer, relation) == prior['association']
        assert {k: v for k, v in prior.items() if k != 'entities'} == {
            k: v for k, v in candidate.items() if k != 'entities'}, 'Entity-only arm changed nonentity fields'
        for entity in prior['entities']:
            assert entity in candidate['entities'], 'Existing entity was removed or altered'
        spans = {(e['offset'], e['length'], e['identifier']) for e in candidate['entities']}
        assert len(spans) == len(candidate['entities'])
        for entity in candidate['entities']:
            start, length = int(entity['offset']), int(entity['length'])
            matches = [p for p in doc['full_text'] if int(p.get('offset', 0)) <= start
                       and start+length <= int(p.get('offset', 0))+len(p.get('text', ''))]
            assert length > 0 and len(matches) == 1
            local = start-int(matches[0].get('offset', 0))
            assert matches[0]['text'][local:local+length] == entity['text']
        associations = transfer.associations(doc, candidate['entities'], vectorizer, relation)
        old_relations = {(r['patient_id'], x) for r in prior['association'] for x in r['phenotype']}
        new_relations = {(r['patient_id'], x) for r in associations for x in r['phenotype']}
        assert old_relations <= new_relations
        added += len(candidate['entities'])-len(prior['entities'])
        predictions.append(dict(candidate, association=associations))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Use a new immutable output directory'
    (out/'baseline.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in prior_rows))
    assert base.sha(out/'baseline.jsonl') == hashes['baseline']
    (out/'full-lr.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in predictions))
    manifest = {'configuration': vars(args), 'inputs': hashes,
                'source_sha256': base.sha(__file__), 'dependencies': {
                    'sequence_base': base.sha(base.__file__), 'transfer_relation': base.sha(transfer.__file__)},
                'policy': 'unchanged train-fold LR>=.55; exact baseline replay; preserve frozen entity pool and old relations',
                'relation_training': training, 'added_entities': added,
                'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'outputs': {name: base.sha(out/name) for name in ['baseline.jsonl', 'full-lr.jsonl']},
                'gold_read': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'added_entities': added, 'outputs': manifest['outputs']}))


if __name__ == '__main__':
    main()
