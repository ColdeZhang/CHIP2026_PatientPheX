#!/usr/bin/env python3
"""Read-only marginal entity diagnostics; gold never affects prediction creation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate import positive_units, mention_counts, prf
from run import read_jsonl, sha256


def span_metrics(gold, predicted):
    return prf(len(gold & predicted), len(predicted - gold), len(gold - predicted))


def summarize(gold_rows, predicted_rows, baseline_rows, linked):
    gold = {str(d['pmc_id']): d for d in gold_rows}
    pred = {str(d['pmc_id']): d for d in predicted_rows}
    base = {str(d['pmc_id']): d for d in baseline_rows}
    assert set(gold) == set(pred) == set(base), 'All document IDs must match'
    totals = {name: Counter() for name in ['baseline', 'extracted', 'novel_coordinates', 'nonoverlapping', 'union']}
    full = {name: Counter() for name in totals}
    per_document = []
    for pmc, doc in gold.items():
        truth = positive_units(doc, doc, 'all')
        extracted = positive_units(pred[pmc], doc, 'all')
        previous = positive_units(base[pmc], doc, 'all')
        novel = {s: ids for s, ids in extracted.items() if s not in previous}
        nonoverlap = {s: ids for s, ids in novel.items() if not any(
            s[0] < b[0] + b[1] and b[0] < s[0] + s[1] for b in previous)}
        union = {s: set(ids) for s, ids in previous.items()}
        for s, ids in extracted.items():
            union.setdefault(s, set()).update(ids)
        views = dict(baseline=previous, extracted=extracted,
                     novel_coordinates=novel, nonoverlapping=nonoverlap, union=union)
        row = {'pmc_id': pmc}
        for name, value in views.items():
            stats = span_metrics(set(truth), set(value))
            totals[name].update({k: stats[k] for k in ['tp', 'fp', 'fn']})
            if linked or name == 'baseline':
                full[name].update(dict(zip(['tp', 'fp', 'fn'], mention_counts(truth, value))))
            row[name] = {k: stats[k] for k in ['tp', 'fp']}
        per_document.append(row)
    return {'span_only': {n: prf(**v) for n, v in totals.items()},
            'span_and_hpo_v2': {n: prf(**v) for n, v in full.items() if v},
            'per_document': per_document,
            'definitions': {
                'span_only': 'Exact (pmc_id, offset, length), excludes note=NO; ignores all identifiers; not Score.',
                'novel_coordinates': 'Exact coordinates absent from baseline; may overlap a different baseline span.',
                'nonoverlapping': 'No character overlap with any baseline entity; half-open raw-text offsets.',
                'span_and_hpo_v2': 'Shared frozen evaluator mention_counts, including compound identifiers and -1 wildcard.',
                'subset_recall': 'Subset FN/recall still use all gold; use subset TP/FP/precision to assess marginal quality.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--baseline', required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--pred')
    group.add_argument('--items', help='Unlinked normalization-items.json; only span diagnostics')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    gold, base = read_jsonl(args.gold), read_jsonl(args.baseline)
    if args.pred:
        pred = read_jsonl(args.pred)
    else:
        occurrences = json.loads(Path(args.items).read_text())['occurrences']
        pred = [{'pmc_id': d['pmc_id'], 'entities': [dict(e, identifier='-1', note=None)
                for e in occurrences.get(str(d['pmc_id']), [])]} for d in base]
    result = summarize(gold, pred, base, bool(args.pred))
    result.update(purpose='Read-only evaluation; never creates or changes predictions',
                  files={k: sha256(getattr(args, k)) for k in ['gold', 'baseline', 'pred', 'items'] if getattr(args, k)},
                  code_sha256=sha256(__file__), evaluator_sha256=sha256(Path(__file__).resolve().parents[2] / 'scripts/evaluate.py'))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['span_only', 'span_and_hpo_v2']}))


if __name__ == '__main__':
    main()
