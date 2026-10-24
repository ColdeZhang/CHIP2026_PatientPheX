#!/usr/bin/env python3
"""Frozen train-only CPU contracts. No torch, weights, forward, fit or evaluator."""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append('/usr/lib/python3/dist-packages')
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import tokenizers
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def save_rows(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def functions_only(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def reverse_training(row, reverse):
    """Reuse upstream prompt transform and rename target keys by the same map."""
    result, mapping = reverse(row)
    answer = json.loads(row['answer'])
    assert list(answer) == list(row['code_map'])
    assert row['answer'] == json.dumps(answer, ensure_ascii=False, separators=(',', ':'))
    inverse = {new: old for old, new in mapping.items()}
    target = {new: answer[inverse[new]] for new in result['code_map']}
    result['answer'] = json.dumps(target, ensure_ascii=False, separators=(',', ':'))
    assert all(isinstance(value, list) and len(value) == len(set(value))
               and set(value) <= set(row['patient_ids']) for value in answer.values())
    assert {(row['code_map'][cid], tuple(value)) for cid, value in answer.items()} == {
        (result['code_map'][cid], tuple(value)) for cid, value in target.items()}
    return result, mapping


class FrozenTokenizer:
    """Same fixed JSON tokenizer + Jinja template; old native audit is exact oracle."""
    def __init__(self, tokenizer, template):
        self.backend = Tokenizer.from_file(str(tokenizer))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def raise_exception(message):
            raise ValueError(message)
        env.globals['raise_exception'] = raise_exception
        self.template = env.from_string(Path(template).read_text())
        self.eos_token = '<|im_end|>'
        self.eos_token_id = self.backend.token_to_id(self.eos_token)
        assert self.eos_token_id == 248046

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking, return_dict=False):
        assert enable_thinking is False and return_dict is False
        text = self.template.render(messages=messages, tools=None,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=False)
        return self.backend.encode(text, add_special_tokens=False).ids if tokenize else text


def token_audit(row, encoded):
    return dict(row_id=row['row_id'], pmc_id=str(row['pmc_id']),
                full_tokens=len(encoded['input_ids']),
                prefix_tokens=encoded['prefix_tokens'],
                supervised_tokens=encoded['supervised_tokens'],
                eos_token_id=encoded['eos_token_id'],
                rendered_prefix_sha256=encoded['rendered_prefix_sha256'],
                rendered_full_sha256=encoded['rendered_full_sha256'],
                input_ids_sha256=digest(encoded['input_ids']),
                labels_sha256=digest(encoded['labels']))


def aggregate(audit):
    lengths = [x['full_tokens'] for x in audit]
    return dict(rows=len(audit), tokens=sum(lengths), maximum=max(lengths),
                minimum=min(lengths), median=statistics.median(lengths),
                supervised_tokens=sum(x['supervised_tokens'] for x in audit),
                prefix_tokens=sum(x['prefix_tokens'] for x in audit),
                over6144=sum(x > 6144 for x in lengths), truncated=0)


def synthetic_contracts(reverse, contracts):
    result = {'upstream_exact_prompt_reverse_cases': contracts()}
    tests = []
    for n in [1, 2, 3, 8, 11]:
        candidates = [dict(code=f'C{i:03d}', meaning='same literal C000',
                           evidence=[dict(context='[PATIENT P1: literal C010] P10 C001')])
                      for i in range(n)]
        answer = {c['code']: (['P1', 'P10'] if i % 3 == 0 else ['P10'] if i % 3 == 1 else [])
                  for i, c in enumerate(candidates)}
        row = dict(row_id='synthetic', pmc_id='synthetic',
                   code_map={c['code']: f'HP:{i:07d}' for i, c in enumerate(candidates)},
                   patient_ids=['P1', 'P10'],
                   messages=[dict(role='system', content='literal C000'),
                             dict(role='user', content='literal C010\n\nCandidate phenotypes:\n' + json.dumps(candidates))],
                   answer=json.dumps(answer, separators=(',', ':')))
        new, mapping = reverse_training(row, reverse)
        back, inverse = reverse_training(new, reverse)
        assert back == row and back['answer'].encode() == row['answer'].encode()
        assert all(inverse[value] == key for key, value in mapping.items())
        tests.append(dict(candidates=n, answer_inverse_exact=True, literal_CIDs_unchanged=True))
    result['training_target_cases'] = tests
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory
    plan = load(directory / 'plan.json')
    assert sha(__file__) == plan['source_sha256']
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU limit')))
    signal.alarm(plan['cpu_cap_seconds'])
    start = time.monotonic()
    references = plan['references']
    for value in references.values():
        assert ref(value['path']) == value
    train = [json.loads(line) for line in Path(references['train']['path']).read_text().splitlines()]
    original_audit = {x['row_id']: x for x in load(references['native_token_audit']['path'])['rows']}
    provenance = load(references['preparation_manifest']['path'])
    docs = sorted({str(x['pmc_id']) for x in train})
    assert len(train) == len({x['row_id'] for x in train}) == 124 and len(docs) == 32
    assert docs == provenance['training_document_ids']
    assert not set(docs) & set(provenance['excluded_document_ids'])
    assert provenance['training_sha256'] == references['train']['sha256']
    assert provenance['all_target_documents_excluded_from_own_generators']
    assert provenance['no_F1_or_outer0_in_any_generator_or_SFT_training']
    for chain in provenance['provenance']:
        generator, target = set(chain['generator_ids']), set(chain['target_ids'])
        assert len(generator) == len(target) == 16 and not generator & target
        assert generator | target == set(docs)
    upstream = functions_only(references['reverse_source']['path'],
        ['parts', 'replace_top_code', 'reverse', 'contracts'],
        dict(copy=copy, json=json, re=re, DIVIDER='\n\nCandidate phenotypes:\n'))
    encode = functions_only(references['encode_source']['path'], ['encode'], {'hashlib': hashlib})['encode']
    batch_plan = functions_only(references['trainer_source']['path'], ['batch_plan'], {'random': random})['batch_plan']
    synthetic = synthetic_contracts(upstream['reverse'], upstream['contracts'])
    tokenizer = FrozenTokenizer(references['tokenizer']['path'], references['template']['path'])
    old_audit, reversed_audit, reversed_rows, sidecar = [], [], [], []
    for row in train:
        new, mapping = reverse_training(row, upstream['reverse'])
        back, inverse = reverse_training(new, upstream['reverse'])
        assert back == row and json.dumps(back, ensure_ascii=False) == json.dumps(row, ensure_ascii=False)
        old = token_audit(row, encode(row, tokenizer))
        assert old == original_audit[row['row_id']], ('native oracle mismatch', row['row_id'])
        new_tokens = token_audit(new, encode(new, tokenizer))
        assert max(old['full_tokens'], new_tokens['full_tokens']) <= 6144
        old_audit.append(old)
        reversed_audit.append(new_tokens)
        sidecar.append(dict(original_row_id=row['row_id'], pmc_id=row['pmc_id'],
                            augmented_row_id=row['row_id'] + ':reverse-candidates',
                            old_to_new=mapping, new_to_old=inverse,
                            original_row_sha256=digest(row), reversed_row_sha256=digest(new),
                            inverse_all_fields_exact=True, target_inverse_byteexact=True,
                            original=old, reversed=new_tokens))
        # Unique training-row ID is required by the old trainer. The content and IDs
        # of original rows remain exact; remove this explicit suffix on inverse.
        new['row_id'] += ':reverse-candidates'
        assert new['row_id'] not in {x['row_id'] for x in train}
        reversed_rows.append(new)
    save_rows(directory / 'reverse124.jsonl', reversed_rows)
    save_rows(directory / 'augmented248.jsonl', train + reversed_rows)
    save(directory / 'row-inverse-and-token-audit.json', sidecar)
    old_stats, new_stats = aggregate(old_audit), aggregate(reversed_audit)
    calls = []
    for logical_epoch in range(4):
        groups = batch_plan(124, 1, 8, plan['schedule']['seed'], logical_epoch)
        assert len(groups) == 16 and [len(g) for g in groups] == [8] * 15 + [4]
        assert sorted(index for group in groups for batch in group for index in batch) == list(range(124))
        variant = ['O', 'R', 'O', 'R'][logical_epoch]
        for group_index, group in enumerate(groups):
            indices = [batch[0] for batch in group]
            step = logical_epoch * 16 + group_index
            warmup, total = 4, 64
            lr = 1e-4 * (step / warmup if step < warmup else max(0, (total-step) / (total-warmup)))
            calls.append(dict(optimizer_call=step+1, logical_epoch=logical_epoch+1,
                              shared_sample_ids=[train[i]['row_id'] for i in indices],
                              group_rows=len(indices), control_variant='O', augmented_variant=variant,
                              proposed_lr=lr,
                              control_tokens=sum(old_audit[i]['full_tokens'] for i in indices),
                              augmented_tokens=sum((old_audit if variant == 'O' else reversed_audit)[i]['full_tokens'] for i in indices)))
    assert len(calls) == 64 and sum(x['group_rows'] for x in calls) == 496
    save(directory / 'matched-call-schedule.json', calls)
    old_complete = load(references['old_training_complete']['path'])
    full_complete = load(references['full64_training_complete']['path'])
    public_complete = load(references['public212_training_complete']['path'])
    control_tokens = old_stats['tokens'] * 4
    augmented_tokens = (old_stats['tokens'] + new_stats['tokens']) * 2
    projections = {}
    for name, complete, observed_tokens in [
        ('same_A800_old124x2', old_complete, old_stats['tokens'] * 2),
        ('same_A800_full265x2', full_complete, 864937 * 2),
        ('same_A800_public212x2', public_complete, 651512 * 2)]:
        projections[name] = dict(observed_seconds=complete['elapsed_seconds'], observed_tokens=observed_tokens,
                                control_seconds=complete['elapsed_seconds']*control_tokens/observed_tokens,
                                augmented_seconds=complete['elapsed_seconds']*augmented_tokens/observed_tokens)
    report = dict(status='CPU_feasibility_complete_no_training_or_quality',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=ref(__file__), plan=ref(directory/'plan.json'),
        runtime=dict(python=sys.version, tokenizers=tokenizers.__version__, jinja2=jinja2.__version__,
                     torch_imported='torch' in sys.modules, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),
        cpu_seconds=time.monotonic()-start, training_documents=32,
        excluded_document_overlap=0, existing_train_label_only=True,
        original=old_stats, reverse=new_stats,
        original124_all_fields_equal_saved_native_tokenizer_audit=True,
        reversed_changed_rows=sum(x['original_row_sha256'] != x['reversed_row_sha256'] for x in sidecar),
        token_changed_rows=sum(x['original']['full_tokens'] != x['reversed']['full_tokens'] for x in sidecar),
        target_changed_rows=sum(train[i]['answer'] != reversed_rows[i]['answer'] for i in range(124)),
        all124_prompt_and_target_inverse_exact=True,
        candidate_values=sum(len(x['code_map']) for x in train),
        positive_patient_concept_pairs=sum(len(ps) for x in train for ps in json.loads(x['answer']).values()),
        total_patient_concept_pairs=sum(len(x['code_map'])*len(x['patient_ids']) for x in train),
        synthetic=synthetic,
        matched_schedule=dict(logical_epochs=4, control='O/O/O/O', augmented='O/R/O/R',
            optimizer_calls=64, nonzero_lr_calls=63, warmup_calls=4, example_exposures=496,
            full_groups=60, short4_groups=4, identical_original_sample_ids_per_call=True,
            control_tokens=control_tokens, augmented_tokens=augmented_tokens,
            control_supervised_tokens=old_stats['supervised_tokens']*4,
            augmented_supervised_tokens=(old_stats['supervised_tokens']+new_stats['supervised_tokens'])*2,
            ordinary248x2_calls=math.ceil(248/8)*2,
            ordinary248x2_does_not_match64=True,
            schedule_is_CPU_proposal_not_a_training_launch=True),
        throughput_projections=projections,
        projection_caveat='Token-linear estimates include old loading/checkpoint overhead; no new GPU speed guarantee, floating-point equivalence, or model quality.',
        no_F1_gold=True, no_original_dev=True, no_model_weights=True, no_fit=True,
        artifacts={name:ref(directory/name) for name in ['reverse124.jsonl','augmented248.jsonl','row-inverse-and-token-audit.json','matched-call-schedule.json']})
    save(directory/'summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['artifacts','synthetic','runtime']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
