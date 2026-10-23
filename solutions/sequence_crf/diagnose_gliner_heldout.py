#!/usr/bin/env python3
"""Span-only diagnostics for a frozen outer-fold GLiNER control, after prediction."""
import argparse
from collections import Counter
import json
from pathlib import Path

import run as base
import diagnose_gliner_pilot as helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['gold', 'input', 'train', 'base', 'base-manifest', 'candidates',
                 'freeze', 'training-manifest', 'data-manifest', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    paths = {name: base.sha(value) for name, value in vars(args).items() if name != 'output'}
    frozen = json.loads(Path(args.freeze).read_text())
    assert frozen['gold_read'] is False and frozen['frozen_at_utc']
    assert frozen['candidate_sha256'] == paths['candidates']
    assert frozen['input_sha256'] == paths['input']
    training = json.loads(Path(args.training_manifest).read_text())
    prepared = json.loads(Path(args.data_manifest).read_text())
    assert training['fingerprint']['configuration']['training_part'] == 'train48'
    assert training['fingerprint']['data_manifest_sha256'] == paths['data_manifest']
    assert training['fingerprint']['excluded_input_sha256'] == paths['input']
    assert training['fingerprint']['train_sha256'] == prepared['artifacts']['train48']['sha256']
    assert prepared['files']['train'] == paths['train']
    role = ('base' if frozen['model_sha256'] == training['fingerprint']['base_weight_sha256']
            else 'finetuned')
    if role == 'finetuned':
        assert frozen['model_sha256'] == training['epoch_results'][-1]['model_sha256']
    prior_source = json.loads(Path(args.base_manifest).read_text())
    assert prior_source['output_sha256'] == paths['base']
    assert prior_source['inputs']['train'] == paths['train']
    assert prior_source['inputs']['input'] == paths['input']
    def index(path):
        rows = base.read_jsonl(path)
        mapping = {str(d['pmc_id']): d for d in rows}
        assert len(mapping) == len(rows), 'Duplicate document ID'
        return mapping

    docs, truth, prior, candidates = map(index, [args.input, args.gold, args.base, args.candidates])
    assert set(docs) == set(truth) == set(prior) == set(candidates)
    assert not set(docs) & {str(d['pmc_id']) for d in base.read_jsonl(args.train)}
    totals = {name: Counter() for name in ['gliner', 'crf_sap', 'nonoverlap_union']}
    errors = {name: Counter() for name in totals}
    marginal = Counter()
    per_document = []
    for key, doc in docs.items():
        assert doc == {k: v for k, v in truth[key].items() if k not in {'entities', 'association'}}
        gold = set(helpers.evaluate.positive_units(truth[key], truth[key], 'all'))
        before = set(helpers.evaluate.positive_units(prior[key], truth[key], 'all'))
        negative = {(int(e['offset']), int(e['length'])) for e in truth[key]['entities']
                    if str(e.get('note') or '').upper() == 'NO'}
        rows = candidates[key]['candidates']
        predicted = {(int(e['offset']), int(e['length'])) for e in rows}
        assert len(predicted) == len(rows), 'Duplicate candidate span'
        for entity in rows:
            offset, length = int(entity['offset']), int(entity['length'])
            matches = [p for p in doc['full_text'] if int(p.get('offset', 0)) <= offset
                       and offset+length <= int(p.get('offset', 0))+len(p.get('text', ''))]
            assert length > 0 and len(matches) == 1
            local = offset-int(matches[0].get('offset', 0))
            assert matches[0]['text'][local:local+length] == entity['text']
        added = set()
        for entity in sorted(rows, key=lambda e: (-e['score'], -e['length'], e['offset'])):
            span = int(entity['offset']), int(entity['length'])
            if not any(helpers.overlaps(span, x) for x in before | added):
                added.add(span)
        local = {}
        for name, spans in [('gliner', predicted), ('crf_sap', before), ('nonoverlap_union', before | added)]:
            count = helpers.counts(gold, spans)
            totals[name].update(count)
            errors[name].update(helpers.error_types(gold, spans, negative))
            local[name] = helpers.metric(count)
        addition = {'accepted_tp': len(added & gold), 'accepted_fp': len(added-gold)}
        marginal.update(addition)
        per_document.append({'pmc_id': key, 'metrics': local, 'marginal': addition})
    report = {'scope': 'outer fold0 development; pure spans, not HPO or task Score',
              'model_role': role, 'configuration': vars(args), 'input_shas': paths,
              'source_sha256': base.sha(__file__), 'dependencies':
                  {'base': base.sha(base.__file__), 'helper': base.sha(helpers.__file__),
                   'evaluator': base.sha(helpers.evaluate.__file__)},
              'metrics': {k: helpers.metric(v) for k, v in totals.items()},
              'marginal': dict(marginal), 'error_types': {k: dict(v) for k, v in errors.items()},
              'per_document': per_document,
              'limitations': ['Positive NIL retained, HPO IDs collapsed, NO excluded.',
                              'The fold is held out of this model training but has been used for development.',
                              'No prediction, threshold or checkpoint choice is modified by this diagnostic.']}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'model_role': role, 'metrics': report['metrics'], 'marginal': report['marginal']}))


if __name__ == '__main__':
    main()
