#!/usr/bin/env python3
"""Ground coreference strings; preserve original anchors and all non-patient input."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re

from run import read_jsonl, sha
from merge_pair_rag import parse_json


def locate(prompt, text, occurrence):
    """Count in displayed TEXT; reject an occurrence crossing passage boundaries."""
    if not isinstance(text, str) or not text or type(occurrence) is not int or occurrence < 1:
        return None
    joined = '\n\n'.join(segment['text'] for segment in prompt['segments'])
    assert joined == prompt['text']
    matches = list(re.finditer(re.escape(text), joined))
    if occurrence > len(matches):
        return None
    match = matches[occurrence-1]
    cursor = 0
    for segment in prompt['segments']:
        end = cursor + len(segment['text'])
        if cursor <= match.start() < match.end() <= end:
            return int(segment['offset']) + match.start() - cursor
        cursor = end + 2
    return None


def locate_context(prompt, text, context):
    """Unique verbatim context + exactly one complete-word reference, no guessed offset."""
    if not isinstance(text, str) or not text or not isinstance(context, str) or not 20 <= len(context) <= 300:
        return None
    joined = '\n\n'.join(segment['text'] for segment in prompt['segments'])
    assert joined == prompt['text']
    contexts = list(re.finditer(re.escape(context), joined))
    refs = list(re.finditer(r'(?<!\w)'+re.escape(text)+r'(?!\w)', context))
    if len(contexts) != 1 or len(refs) != 1:
        return None
    begin = contexts[0].start()+refs[0].start()
    end = begin+len(text)
    if (begin and joined[begin-1].isalnum()) or (end < len(joined) and joined[end].isalnum()):
        return None
    cursor = 0
    for segment in prompt['segments']:
        bound = cursor+len(segment['text'])
        if cursor <= contexts[0].start() < contexts[0].end() <= bound and cursor <= begin < end <= bound:
            return int(segment['offset'])+begin-cursor
        cursor = bound+2
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'prompts', 'responses', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--grounding', choices=['occurrence', 'context'], default='occurrence')
    args = parser.parse_args()
    inputs, prompt_rows, responses = map(read_jsonl, [args.input, args.prompts, args.responses])
    assert all('entities' not in doc and 'association' not in doc for doc in inputs)
    docs = {str(doc['pmc_id']): doc for doc in inputs}
    prompts = {p['row_id']: p for p in prompt_rows}
    assert len(prompts) == len(prompt_rows)
    assert len({r['row_id'] for r in responses}) == len(responses)
    assert {r['row_id'] for r in responses} == set(prompts)
    found = defaultdict(lambda: defaultdict(dict))
    stats = Counter()
    evidence = []
    for response in responses:
        prompt = prompts[response['row_id']]
        assert str(response['pmc_id']) == str(prompt['pmc_id'])
        doc = docs[str(prompt['pmc_id'])]
        assert prompt['patient_ids'] == [str(p['patient_id']) for p in doc['patient']]
        parsed = None if response.get('hit_token_limit') else parse_json(response['response'])
        if not isinstance(parsed, list):
            stats['invalid_or_truncated_responses'] += 1
            continue
        stats['valid_responses'] += 1
        accepted_per_patient = Counter()
        for item in parsed:
            if not isinstance(item, dict) or item.get('patient_id') not in prompt['patient_ids']:
                stats['invalid_item_or_patient_id'] += 1
                continue
            text, pid = item.get('text'), item['patient_id']
            if accepted_per_patient[pid] >= 25:
                stats['over_patient_window_cap'] += 1
                continue
            offset = (locate_context(prompt, text, item.get('context')) if args.grounding == 'context'
                      else locate(prompt, text, item.get('occurrence')))
            if offset is None:
                stats['unmatched_or_cross_passage_quote'] += 1
                continue
            assert any(int(p['offset']) <= offset < offset+len(text) <= int(p['offset'])+len(p['text'])
                       and p['text'][offset-int(p['offset']):offset-int(p['offset'])+len(text)] == text
                       for p in doc['full_text'])
            found[str(prompt['pmc_id'])][pid][(offset, len(text))] = {
                'offset': offset, 'length': len(text), 'text': text}
            accepted_per_patient[pid] += 1
            evidence.append({'pmc_id': prompt['pmc_id'], 'row_id': prompt['row_id'],
                             'patient_id': pid, 'text': text, 'offset': offset,
                             'occurrence': item.get('occurrence'), 'context': item.get('context')})
            stats['grounded_items_including_duplicates'] += 1
    outputs = copy.deepcopy(inputs)
    for doc in outputs:
        for patient in doc['patient']:
            old = {(int(m['offset']), int(m['length'])) for m in patient.get('mention', [])}
            new = found[str(doc['pmc_id'])][str(patient['patient_id'])]
            additions = [new[k] for k in sorted(new) if k not in old]
            # Preserve trusted original identity anchors first for global snippet selection.
            patient['mention'].extend(additions)
            stats['added_mentions'] += len(additions)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in outputs))
    out.with_suffix('.evidence.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in evidence))
    manifest = {'configuration': vars(args), 'stats': dict(stats),
                'source_sha256': sha(__file__), 'parser_sha256': sha(Path(__file__).with_name('merge_pair_rag.py')),
                'files': {k: sha(getattr(args, k)) for k in ['input', 'prompts', 'responses', 'output']},
                'policy': 'only grounded same-passage spans; preserve all original mentions in original order, append sorted novel spans; no generated patient IDs; invalid/truncated replies rejected; context mode requires unique 20-300-character context and a single complete-word reference'}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
