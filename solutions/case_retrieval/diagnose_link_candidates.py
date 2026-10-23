#!/usr/bin/env python3
"""Read-only error and candidate-coverage audit after frozen quote linking."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate import positive_units
from normalize_extraction import parse_json
from run import read_jsonl, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['gold', 'baseline', 'items', 'linked', 'prompts', 'responses', 'output']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    gold = {str(d['pmc_id']): d for d in read_jsonl(args.gold)}
    base = {str(d['pmc_id']): d for d in read_jsonl(args.baseline)}
    linked = {str(d['pmc_id']): d for d in read_jsonl(args.linked)}
    assert set(gold) == set(base) == set(linked)
    data = json.loads(Path(args.items).read_text())
    items = {i['code']: i for i in data['items']}
    rows = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    prompts, responses = rows(args.prompts), rows(args.responses)
    lookup = {p['row_id']: p for p in prompts}
    assert len(lookup) == len(prompts) == len(responses)
    assert {r['row_id'] for r in responses} == set(lookup)
    errors = Counter(); failed = []
    for response in responses:
        expected = set(lookup[response['row_id']]['item_codes'])
        result = parse_json(response['response']); row_errors = Counter()
        if response.get('hit_token_limit'):
            row_errors['token_limit'] += 1
        if not isinstance(result, dict):
            row_errors['invalid_json_object'] += 1
            row_errors['unanswered_item_codes'] += len(expected)
        else:
            row_errors['unanswered_item_codes'] += len(expected - set(result))
            row_errors['unexpected_item_codes'] += len(set(result) - expected)
            for key, value in result.items():
                if key not in expected:
                    continue
                if isinstance(value, list):
                    value = ';'.join(str(part) for part in value)
                candidates = {part for c in items[key]['candidates'] for part in c['identifier'].split(';')}
                if not isinstance(value, str) or (value != '-1' and not set(value.split(';')) <= candidates):
                    row_errors['invalid_or_out_of_candidate_values'] += 1
        row_errors += Counter()  # Drop zero entries; this never changes predictions.
        errors.update(row_errors)
        if row_errors:
            failed.append({'row_id': response['row_id'], 'errors': dict(row_errors)})
    views = {name: Counter() for name in ['all_extracted', 'nonoverlapping_baseline']}
    for pmc, doc in gold.items():
        truth = positive_units(doc, doc, 'all')
        old = positive_units(base[pmc], doc, 'all')
        pred = positive_units(linked[pmc], doc, 'all')
        for occurrence in data['occurrences'].get(pmc, []):
            span = (occurrence['offset'], occurrence['length'])
            names = ['all_extracted']
            if not any(span[0] < s[0]+s[1] and s[0] < span[0]+span[1] for s in old):
                names.append('nonoverlapping_baseline')
            item = items[occurrence['item_code']]
            available = {part for c in item['candidates'] for part in c['identifier'].split(';')}
            for name in names:
                c = views[name]; c['raw_spans'] += 1
                if span not in truth:
                    overlaps = any(span[0] < s[0]+s[1] and s[0] < span[0]+span[1] for s in truth)
                    c['incorrect_span_overlaps_gold' if overlaps else 'incorrect_span_no_gold_overlap'] += 1
                    continue
                c['correct_spans'] += 1
                targets = truth[span]
                unknown = '-1' in targets
                c['gold_unknown_spans'] += int(unknown)
                c['all_gold_ids_in_candidates_or_unknown'] += int(unknown or targets <= available)
                c['candidate_set_missing_gold_id'] += int(not unknown and not targets <= available)
                if span not in pred:
                    c['correct_span_unresolved_link'] += 1
                elif unknown or targets <= pred[span]:
                    c['all_gold_ids_selected_or_unknown'] += 1
                else:
                    c['correct_span_wrong_or_incomplete_hpo'] += 1
    output = {'purpose': 'Read-only diagnostics; gold does not create or alter predictions',
              'response_errors': dict(errors), 'failed_rows': failed,
              'candidate_coverage': {name: dict(value) for name, value in views.items()},
              'limitations': ['Candidate coverage is a diagnostic upper bound on correct extracted spans, not an achievable Score.',
                             'Gold -1 follows evaluator wildcard semantics and is reported separately.',
                             'Span boundary errors are separated from HPO errors; no thresholds or predictions change.'],
              'files': {name: sha256(getattr(args, name)) for name in
                        ['gold', 'baseline', 'items', 'linked', 'prompts', 'responses']},
              'source_sha256': sha256(__file__)}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['response_errors', 'candidate_coverage']}))


if __name__ == '__main__':
    main()
