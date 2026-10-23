#!/usr/bin/env python3
"""Retry failed HPO-link JSON requests once, preserving original context/candidates."""
import argparse
import copy
import json
from pathlib import Path
import re
from normalize_extraction import parse_json
from run import sha256


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def errors(prompt, response, items):
    problems = []
    if response.get('hit_token_limit'):
        problems.append('token_limit')
    parsed = parse_json(response['response'])
    if not isinstance(parsed, dict):
        return problems + ['not_json_object']
    if set(parsed) != set(prompt['item_codes']):
        problems.append('item_key_mismatch')
    for code, value in parsed.items():
        if code not in prompt['item_codes']:
            continue
        if isinstance(value, list):
            value = ';'.join(str(x) for x in value)
        allowed = {part for c in items[code]['candidates'] for part in c['identifier'].split(';')}
        if value != '-1' and (not isinstance(value, str) or
                not re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})*', value) or
                not set(value.split(';')) <= allowed):
            problems.append('invalid_candidate:' + code)
    return problems


def split_request(prompt, count):
    if prompt.get('retry_number') or len(prompt['messages']) != 2:
        raise ValueError('Only original zero-shot requests may receive one retry')
    blocks = prompt['messages'][1]['content'].split('\n\n')
    codes = prompt['item_codes']
    assert len(blocks) == len(codes)
    assert all(block.startswith(code + ': quote=') for block, code in zip(blocks, codes))
    rows = []
    for begin in range(0, len(codes), count):
        row = copy.deepcopy(prompt)
        row.update(row_id=prompt['row_id'] + ':retry1:' + str(begin),
                   source_row_id=prompt['row_id'], retry_number=1,
                   item_codes=codes[begin:begin + count])
        row['messages'][1]['content'] = '\n\n'.join(blocks[begin:begin + count])
        rows.append(row)
    return rows


def write_rows(path, rows):
    data = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows)
    if path.exists() and path.read_text() != data:
        raise ValueError('Refusing to replace a different frozen output: ' + str(path))
    path.write_text(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['prompts', 'responses', 'items', 'output-dir']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--items-per-retry', type=int, default=6)
    ap.add_argument('--retry-responses')
    args = ap.parse_args()
    if args.items_per_retry < 1:
        ap.error('--items-per-retry must be positive')
    prompts, responses = read_rows(args.prompts), read_rows(args.responses)
    lookup = {x['row_id']: x for x in responses}
    assert len(lookup) == len(responses) == len(prompts)
    assert set(lookup) == {p['row_id'] for p in prompts}
    items = {x['code']: x for x in json.loads(Path(args.items).read_text())['items']}
    failures = {p['row_id']: errors(p, lookup[p['row_id']], items) for p in prompts}
    failures = {k: v for k, v in failures.items() if v}
    retries = [r for p in prompts if p['row_id'] in failures for r in split_request(p, args.items_per_retry)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / 'retry-prompts.jsonl', retries)
    report = {'selection': 'Only JSON/expected-item/candidate validity or output token limit; no gold input',
              'failed_rows': failures, 'retry_requests': len(retries),
              'files': {k: sha256(getattr(args, k)) for k in ['prompts', 'responses', 'items']},
              'code_sha256': sha256(__file__), 'configuration': vars(args)}
    name = 'prepare-manifest.json'
    if args.retry_responses:
        rr = read_rows(args.retry_responses)
        by_id = {x['row_id']: x for x in rr}
        assert len(by_id) == len(rr) == len(retries)
        assert set(by_id) == {p['row_id'] for p in retries}
        report['remaining_failures'] = {p['row_id']: problems for p in retries
            if (problems := errors(p, by_id[p['row_id']], items))}
        write_rows(out / 'combined-prompts.jsonl', [p for p in prompts if p['row_id'] not in failures] + retries)
        write_rows(out / 'combined-responses.jsonl', [r for r in responses if r['row_id'] not in failures] + rr)
        report['files']['retry_responses'] = sha256(args.retry_responses)
        name = 'combined-manifest.json'
    report['outputs'] = {p.name: sha256(p) for p in out.glob('*.jsonl')}
    (out / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
