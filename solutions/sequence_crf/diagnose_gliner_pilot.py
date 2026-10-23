#!/usr/bin/env python3
"""Training-only span diagnostic on already frozen GLiNER pilot candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import run as base
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import evaluate


def overlaps(left, right):
    return max(left[0], right[0]) < min(sum(left), sum(right))


def counts(gold, pred):
    return {'tp': len(gold & pred), 'fp': len(pred - gold), 'fn': len(gold - pred)}


def metric(count):
    tp, fp, fn = (count[k] for k in ('tp', 'fp', 'fn'))
    return dict(count, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def error_types(gold, predicted, negative):
    result = Counter()
    for span in predicted - gold:
        name = ('fp_exact_no' if span in negative else
                'fp_overlap_positive' if any(overlaps(span, x) for x in gold) else
                'fp_no_positive_overlap')
        result[name] += 1
    for span in gold - predicted:
        name = ('fn_overlap_prediction' if any(overlaps(span, x) for x in predicted)
                else 'fn_no_prediction_overlap')
        result[name] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['gold', 'input', 'candidates', 'frozen-evidence', 'oof-manifest', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--expected-documents', type=int, default=3)
    args = parser.parse_args()
    frozen = json.loads(Path(args.frozen_evidence).read_text())
    if 'candidate_sha256' in frozen:
        assert frozen['gold_read'] is False and frozen['frozen_at_utc']
        candidate_sha, input_sha = frozen['candidate_sha256'], frozen['input_sha256']
    else:
        recorded = next(r for r in frozen['artifacts'] if Path(r['path']).name == 'candidates.jsonl')
        candidate_sha, input_sha = recorded['sha256'], frozen['selection']['input_sha256']
    assert base.sha(args.candidates) == candidate_sha
    assert base.sha(args.input) == input_sha
    gold = {str(d['pmc_id']): d for d in base.read_jsonl(args.gold)}
    inputs = {str(d['pmc_id']): d for d in base.read_jsonl(args.input)}
    assert len(inputs) == args.expected_documents and set(inputs) <= set(gold)
    for key, doc in inputs.items():
        assert doc == {k: v for k, v in gold[key].items() if k not in {'entities', 'association'}}
    manifest = json.loads(Path(args.oof_manifest).read_text())
    assert base.sha(args.gold) == manifest['files']['train']
    baseline, provenance = {}, []
    for item in manifest['provenance']:
        path = item['augment_command'][item['augment_command'].index('--output')+1]
        assert base.sha(path) == item['candidate_sha256']
        assert not set(item['oof_job']['train_ids']) & set(item['oof_job']['input_ids'])
        for doc in base.read_jsonl(path):
            key = str(doc['pmc_id'])
            if key in inputs:
                assert key not in baseline and key not in item['oof_job']['train_ids']
                baseline[key] = doc
                provenance.append({'pmc_id': key, 'inner_fold': item['inner_fold'],
                                   'candidate_path': path, 'candidate_sha256': item['candidate_sha256'],
                                   'training_sha256': item['oof_job']['train_sha256']})
    assert set(baseline) == set(inputs)
    totals = {name: Counter() for name in ['gliner', 'oof_crf_sap', 'nonoverlap_union']}
    errors = {name: Counter() for name in totals}
    added_errors = Counter()
    marginal = Counter()
    per_document = []
    candidates = base.read_jsonl(args.candidates)
    assert {str(d['pmc_id']) for d in candidates} == set(inputs)
    for row in candidates:
        key = str(row['pmc_id'])
        truth = set(evaluate.positive_units(gold[key], gold[key], 'all'))
        negative = {(int(e['offset']), int(e['length'])) for e in gold[key]['entities']
                    if str(e.get('note') or '').upper() == 'NO'}
        prior = set(evaluate.positive_units(baseline[key], gold[key], 'all'))
        predicted = {(int(e['offset']), int(e['length'])) for e in row['candidates']}
        eligible = {s for s in predicted if not any(overlaps(s, x) for x in prior)}
        added = set()
        # Same predeclared policy: preserve base, greedily resolve new overlap.
        for e in sorted(row['candidates'], key=lambda e: (-e['score'], -e['length'], e['offset'])):
            span = int(e['offset']), int(e['length'])
            if span in eligible and not any(overlaps(span, x) for x in added):
                added.add(span)
        local = {name: counts(truth, spans) for name, spans in
                 [('gliner', predicted), ('oof_crf_sap', prior), ('nonoverlap_union', prior | added)]}
        for name, count in local.items():
            totals[name].update(count)
        for name, spans in [('gliner', predicted), ('oof_crf_sap', prior),
                            ('nonoverlap_union', prior | added)]:
            errors[name].update(error_types(truth, spans, negative))
        added_errors.update({k: v for k, v in error_types(truth, added, negative).items()
                             if k.startswith('fp_')})
        more = {'eligible_tp': len(eligible & truth), 'eligible_fp': len(eligible - truth),
                'accepted_tp': len(added & truth), 'accepted_fp': len(added - truth),
                'overlaps_original': len(predicted - eligible),
                'new_overlap_rejected': len(eligible - added)}
        marginal.update(more)
        per_document.append({'pmc_id': key, 'gold_spans': len(truth),
                             'metrics': {k: metric(v) for k, v in local.items()}, 'marginal': more})
    report = {'scope': f'{len(inputs)} outer-training documents, span-only diagnostic; no HPO or association score',
              'configuration': vars(args), 'source_sha256': base.sha(__file__),
              'dependencies': {'base': base.sha(base.__file__), 'evaluator': base.sha(evaluate.__file__)},
              'sources': {k: base.sha(getattr(args, k)) for k in
                          ['gold', 'input', 'candidates', 'frozen_evidence', 'oof_manifest']},
              'metrics': {k: metric(v) for k, v in totals.items()},
              'error_types': {k: dict(v) for k, v in errors.items()},
              'accepted_addition_error_types': dict(added_errors),
              'marginal': dict(marginal), 'per_document': per_document, 'oof_provenance': provenance,
              'limitations': ['Span-only collapses atomic HPO IDs and includes positive NIL; no linking claim.',
                              'Within-outer-training diagnostic; no original dev or outerfold0 score.',
                              'Model training scope is recorded in the frozen candidate manifest; OOF base uses the appropriate train32.',
                              'No gold rule, candidate, label threshold, or patient relation was changed.']}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'metrics': report['metrics'], 'marginal': report['marginal']}))


if __name__ == '__main__':
    main()
