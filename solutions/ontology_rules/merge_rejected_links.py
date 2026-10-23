#!/usr/bin/env python3
"""Strict finite-choice HPO linking; preserve a frozen baseline in full."""
import argparse
from collections import Counter, defaultdict
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_choice(response):
    if response.get('hit_token_limit') is not False:
        return None, 'missing_or_true_token_limit_flag'
    if not isinstance(response.get('response'), str):
        return None, 'response_not_string'
    try:
        value = json.loads(response['response'], object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError):
        return None, 'invalid_json'
    if not isinstance(value, dict) or set(value) != {'choice'}:
        return None, 'invalid_schema'
    choice = value['choice']
    if not isinstance(choice, str) or choice not in {'C1', 'C2', 'C3', 'C4', 'C5', 'ABSTAIN'}:
        return None, 'invalid_choice'
    return choice, None


def overlap(a, b):
    return a['offset'] < b['offset'] + b['length'] and b['offset'] < a['offset'] + a['length']


def load_relations(source, expected):
    source = Path(source).resolve()
    assert sha(source) == expected['combine']
    assert sha(source.parent / 'run.py') == expected['sequence_base']
    assert sha(source.parent.parent / 'ontology_rules/run.py') == expected['ontology']
    # Fresh CLI process: load the precise sequence dependency as its expected
    # module name, so combine.py cannot accidentally import ontology/run.py.
    assert 'run' not in sys.modules, 'Dependency already imported ambiguously'
    spec = importlib.util.spec_from_file_location('run', source.parent / 'run.py')
    sequence = importlib.util.module_from_spec(spec)
    sys.modules['run'] = sequence
    spec.loader.exec_module(sequence)
    spec = importlib.util.spec_from_file_location('fixed_relation_rules', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'hpo', 'base', 'base-manifest', 'prompts', 'prompt-manifest',
                 'responses', 'response-manifest', 'relation-source', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    baseline_manifest = json.loads(Path(args.base_manifest).read_text())
    prompt_manifest = json.loads(Path(args.prompt_manifest).read_text())
    response_manifest = json.loads(Path(args.response_manifest).read_text())
    assert baseline_manifest['outputs']['augmented.jsonl'] == sha(args.base)
    assert baseline_manifest['inputs']['input'] == sha(args.input)
    assert baseline_manifest['inputs']['hpo'] == sha(args.hpo)
    assert prompt_manifest['prompts_sha256'] == sha(args.prompts)
    assert prompt_manifest['inputs']['hpo'] == sha(args.hpo)
    assert response_manifest['output_sha256'] == sha(args.responses)
    assert response_manifest['fingerprint']['prompts_sha256'] == sha(args.prompts)
    assert response_manifest['expected_rows'] == prompt_manifest['requests']
    relation = load_relations(args.relation_source, baseline_manifest['dependencies'])
    allowed = relation.ontology.load_hpo(args.hpo)[1] - {relation.ontology.ROOT}
    docs = {d['pmc_id']: d for d in read(args.input)}
    bases = read(args.base)
    assert len(docs) == len(bases) and set(docs) == {d['pmc_id'] for d in bases}
    assert all('entities' not in d and 'association' not in d for d in docs.values())
    prompts = read(args.prompts)
    by_id = {r['row_id']: r for r in prompts}
    responses = read(args.responses)
    assert len(by_id) == len(prompts) == len(responses) == response_manifest['expected_rows']
    assert {r['row_id'] for r in responses} == set(by_id), 'Incomplete/unexpected/duplicate rows'
    selected, counts, decisions = defaultdict(list), Counter(), []
    for response in responses:
        prompt = by_id[response['row_id']]
        assert response['pmc_id'] == prompt['pmc_id']
        mapping = prompt['code_to_hpo']
        assert set(mapping) == {'C1', 'C2', 'C3', 'C4', 'C5'}
        assert len(set(mapping.values())) == 5 and set(mapping.values()) <= allowed
        choice, failure = parse_choice(response)
        counts['responses'] += 1
        decisions.append({'row_id': response['row_id'], 'choice': choice, 'failure': failure})
        if failure:
            counts['format_failure'] += 1
            counts['failure_' + failure] += 1
            continue
        if choice == 'ABSTAIN':
            counts['abstain'] += 1
            continue
        candidate = copy.deepcopy(prompt['candidate'])
        candidate['identifier'] = mapping[choice]
        selected[prompt['pmc_id']].append(candidate)
        counts['selected_valid_choice'] += 1
    output_rows = []
    for base in bases:
        doc = docs[base['pmc_id']]
        pred = copy.deepcopy(base)
        additions = []
        for candidate in sorted(selected[base['pmc_id']], key=lambda c: (-c['score'], -c['length'], c['offset'])):
            _, passage = relation.sequence.passage_for(doc, candidate['offset'])
            start = candidate['offset'] - int(passage.get('offset', 0))
            end = start + candidate['length']
            assert end <= len(passage['text']) and passage['text'][start:end] == candidate['text']
            assert not any(overlap(candidate, e) for e in base['entities'])
            assert not relation.ontology.is_negated(passage['text'], start, end)
            if any(overlap(candidate, e) for e in additions):
                counts['overlap_new_rejected'] += 1
                continue
            entity = {key: candidate[key] for key in ('identifier', 'offset', 'length', 'text')}
            entity.update(type='Phenotype', note=None)
            additions.append(entity)
        if additions:
            pred['entities'].extend(additions)
            pred['entities'].sort(key=lambda e: (e['offset'], e['length'], e['identifier']))
            added_relations = {r['patient_id']: set(r['phenotype']) for r in relation.assign(doc, additions)}
            for association in pred['association']:
                old = set(association['phenotype'])
                extra = added_relations[association['patient_id']] - old
                if extra:
                    association['phenotype'] = sorted(old | extra)
                    counts['added_relations'] += len(extra)
        assert all(e in pred['entities'] for e in base['entities'])
        new_map = {a['patient_id']: set(a['phenotype']) for a in pred['association']}
        assert all(set(a['phenotype']) <= new_map[a['patient_id']] for a in base['association'])
        counts['added_entities'] += len(additions)
        output_rows.append(pred)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output_rows))
    manifest = {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_sha256': sha(__file__), 'relation_dependencies': baseline_manifest['dependencies'],
                'inputs': {name: {'path': str(Path(getattr(args, name)).resolve()), 'sha256': sha(getattr(args, name))}
                           for name in ('input', 'hpo', 'base', 'base_manifest', 'prompts', 'prompt_manifest',
                                        'responses', 'response_manifest', 'relation_source')},
                'counts': dict(counts), 'output_sha256': sha(out), 'gold_read': False,
                'base_fully_preserved': True, 'NIL_added': False,
                'policy': 'Strict finite choice or explicit format failure; fixed GLiNER overlap order; preserve base and union only same frozen rules on new additions.'}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    out.with_suffix('.decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
