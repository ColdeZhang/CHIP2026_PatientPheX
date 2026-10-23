#!/usr/bin/env python3
"""Aggregate paired pilot diagnostics on selected candidate-patient pairs only."""
import argparse
import json
from pathlib import Path
import run as base
from merge_pair_rag import parse_json


def counts(predicted, gold):
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    p = argparse.ArgumentParser()
    for name in ['prompts', 'control', 'thinking', 'gold', 'baseline', 'output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    prompts = base.read_jsonl(a.prompts)
    gold = {str(d['pmc_id']): base.gold_associations(d) for d in base.read_jsonl(a.gold)}
    baseline = {str(d['pmc_id']): base.gold_associations(d) for d in base.read_jsonl(a.baseline)}
    records, diagnostics, positives = {}, {}, set()
    for mode in ['control', 'thinking']:
        response = {x['row_id']: x for x in base.read_jsonl(getattr(a, mode))}
        assert set(response) == {x['row_id'] for x in prompts}
        valid, predicted, effective = set(), set(), set()
        for row in prompts:
            pmc = str(row['pmc_id'])
            r = response[row['row_id']]
            parsed = parse_json(r['response']) if not r.get('hit_token_limit') else None
            for code, hp in row['code_map'].items():
                key = (row['row_id'], code)
                value = parsed.get(code) if isinstance(parsed, dict) else None
                good = isinstance(value, list) and all(str(pid) in row['patient_ids'] for pid in value)
                if good:
                    valid.add(key)
                for pid in row['patient_ids']:
                    pair = (row['row_id'], code, pid)
                    if hp in gold[pmc].get(pid, set()):
                        positives.add(pair)
                    if good and pid in value:
                        predicted.add(pair)
                        effective.add(pair)
                    elif not good and hp in baseline[pmc].get(pid, set()):
                        effective.add(pair)
        records[mode] = (valid, predicted, effective)
        diagnostics[mode] = {'prompts': len(response), 'hit_token_limit': sum(bool(x.get('hit_token_limit')) for x in response.values()),
                             'output_tokens': sum(x['output_tokens'] for x in response.values()),
                             'valid_candidate_decisions': len(valid), 'effective_with_predefined_lr_fallback': counts(effective, positives)}
    common = records['control'][0] & records['thinking'][0]
    common_gold = {x for x in positives if x[:2] in common}
    for mode in diagnostics:
        diagnostics[mode]['common_resolved_pairs'] = counts({x for x in records[mode][1] if x[:2] in common}, common_gold)
    result = {'scope': 'input-selected 12 prompts in 3 documents; conditional candidate-patient pair diagnostics, not competition Score',
              'candidate_decisions': sum(len(x['code_map']) for x in prompts), 'positive_candidate_patient_pairs': len(positives),
              'common_resolved_candidate_decisions': len(common), 'metrics': diagnostics,
              'files': {k: base.sha(getattr(a, k)) for k in ['prompts', 'control', 'thinking', 'gold', 'baseline']},
              'source_sha256': base.sha(__file__), 'parser_sha256': base.sha(__import__('merge_pair_rag').__file__)}
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
