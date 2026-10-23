#!/usr/bin/env python3
"""Input-only one-example retrieval and actual native-template token inventory."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('/usr/lib/python3/dist-packages')  # existing read-only Jinja2
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import tokenizers
from tokenizers import Tokenizer

HEADER = 'Worked example from a different report; EX_ identifiers apply only to this example.\n\n'
DIVIDER = '\n\nCandidate phenotypes:\n'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
def load(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_bytes().splitlines()]
def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def replace_user(text, codes, patients):
    """Only JSON code fields, PATIENT marker IDs and line-start anchor IDs."""
    def code(match):
        old = match.group(2)
        return match.group(1) + codes.get(old, old) + match.group(3)
    text = re.sub(r'("code"\s*:\s*")([^"\\]+)(")', code, text)
    # Simultaneous regex replacement prevents P1/P10 and generated-ID cascades.
    alternatives = '|'.join(re.escape(k) for k in sorted(patients, key=lambda k: (-len(k), k)))
    assert alternatives
    text = re.sub(r'(\[PATIENT )(' + alternatives + r')(?=:)',
                  lambda m: m.group(1) + patients[m.group(2)], text)
    text = re.sub(r'^(' + alternatives + r')(?=:)', lambda m: patients[m.group(1)], text, flags=re.M)
    return text


def replace_answer(answer, mapping):
    # Preserve every non-ID byte including whitespace and key/list ordering.
    def repl(match):
        value = json.loads(match.group())
        return json.dumps(mapping[value], ensure_ascii=False) if value in mapping else match.group()
    return re.sub(r'"(?:\\.|[^"\\])*"', repl, answer)


def example(row):
    assert [m['role'] for m in row['messages']] == ['system', 'user']
    user, answer = row['messages'][1]['content'], row['answer']
    assert 'EX_' not in user and 'EX_' not in answer
    codes = {c: 'EX_' + c for c in row['code_map']}
    patients = {p: 'EX_P%03d' % i for i, p in enumerate(row['patient_ids'])}
    assert len(set(codes.values()) | set(patients.values())) == len(codes) + len(patients)
    renamed = replace_user(user, codes, patients)
    renamed_answer = replace_answer(answer, codes | patients)
    assert replace_user(renamed, {v: k for k, v in codes.items()},
                        {v: k for k, v in patients.items()}) == user
    assert replace_answer(renamed_answer, {v: k for k, v in (codes | patients).items()}) == answer
    parsed = json.loads(answer)
    assert set(parsed) == set(codes)
    assert all(isinstance(v, list) and set(v) <= set(patients) for v in parsed.values())
    parsed_new = json.loads(renamed_answer)
    assert set(parsed_new) == set(codes.values())
    assert all(set(v) <= set(patients.values()) for v in parsed_new.values())
    original_candidates = json.loads(user.split(DIVIDER, 1)[1])
    new_candidates = json.loads(renamed.split(DIVIDER, 1)[1])
    assert {c['code'] for c in original_candidates} == set(codes)
    assert {c['code'] for c in new_candidates} == set(codes.values())
    return dict(user=HEADER + renamed, answer=renamed_answer, codes=codes, patients=patients,
                user_inverse_byteexact=True, answer_inverse_byteexact=True)


def lexical_input(row):
    # Answers and candidate/patient identifier metadata are never consulted here.
    user = row['messages'][1]['content']
    anchors, raw = user.split(DIVIDER, 1)
    anchors = re.sub(r'^Patient identities and global anchors:\n', '', anchors)
    anchors = re.sub(r'^\S+: ', '', anchors, flags=re.M)
    anchors = re.sub(r'\[PATIENT [^\]]*?: ', '', anchors).replace(']', '')
    texts = [anchors]
    for c in json.loads(raw):
        texts.append(c['meaning'])
        for e in c['evidence']:
            texts.extend([e['quote'], re.sub(r'\[PATIENT [^\]]*?: ', '', e['context']).replace(']', '')])
    return '\n'.join(dict.fromkeys(texts))


def describe(values):
    values = sorted(values)
    return dict(rows=len(values), total=sum(values), minimum=values[0],
                median=values[math.ceil(len(values)*.5)-1], p95=values[math.ceil(len(values)*.95)-1], maximum=values[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', type=Path, required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU >120s')))
    signal.alarm(120)
    started = time.monotonic()
    b = a.directory
    plan = load(b/'plan.json')
    for r in plan['references'].values(): assert sha(r['path']) == r['sha256']
    refs = plan['references']
    train, query = rows(refs['train']['path']), rows(refs['query']['path'])
    prep = load(refs['preparation']['path'])
    assert len(train) == 124 and len(query) == 80
    train_ids, query_ids = {r['pmc_id'] for r in train}, {r['pmc_id'] for r in query}
    assert len(train_ids) == 32 and len(query_ids) == 16 and not train_ids & query_ids
    assert train_ids == set(prep['training_document_ids']) and query_ids <= set(prep['excluded_document_ids'])
    assert all([m['role'] for m in r['messages']] == ['system', 'user'] for r in train + query)
    assert all('answer' not in r for r in query)
    assert len({r['row_id'] for r in train + query}) == 204
    assert len({r['messages'][0]['content'] for r in train + query}) == 1
    assert all('EX_' not in m['content'] for r in query for m in r['messages'])
    def exception(message): raise ValueError(message)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals['raise_exception'] = exception
    template = env.from_string(Path(refs['template']['path']).read_text())
    tokenizer = Tokenizer.from_file(refs['tokenizer']['path'])
    def render(messages):
        return template.render(messages=messages, tools=None, add_generation_prompt=True, enable_thinking=False)
    def encoded_length(text): return len(tokenizer.encode(text, add_special_tokens=False).ids)
    audit = {r['row_id']: r for r in load(refs['old_token_audit']['path'])['rows']}
    for r in train:
        prefix = render(r['messages'])
        assert digest(prefix) == audit[r['row_id']]['rendered_prefix_sha256']
        assert encoded_length(prefix) == audit[r['row_id']]['prefix_tokens']
    # Inspect only archived input-token counts/times, never response text/quality.
    old_response = {r['row_id']: {k: r[k] for k in ['input_tokens', 'output_tokens', 'batch_seconds']}
                    for r in rows(refs['old_responses_for_token_and_time_only']['path'])}
    query_lengths = {}
    for r in query:
        n = encoded_length(render(r['messages']))
        assert n == old_response[r['row_id']]['input_tokens']
        query_lengths[r['row_id']] = n
    # Boundary fixture proves no P1/P10 or ordinary C000/P1 substring mutation.
    fixture = 'P1: [PATIENT P10: foo] P1 C000\nP10: [PATIENT P1: bar]\n{"code": "C000"}'
    renamed = replace_user(fixture, {'C000': 'EX_C000'}, {'P1': 'EX_P000', 'P10': 'EX_P001'})
    assert 'foo] P1 C000' in renamed
    assert replace_user(renamed, {'EX_C000': 'C000'}, {'EX_P000': 'P1', 'EX_P001': 'P10'}) == fixture
    examples = {r['row_id']: example(r) for r in train}
    # The budget only sees user input; answer text is not used to admit/rank rows.
    lengths = {r['row_id']: encoded_length(examples[r['row_id']]['user']) for r in train}
    eligible = sorted([r for r in train if lengths[r['row_id']] <= 2048], key=lambda r: r['row_id'])
    assert eligible
    texts = [lexical_input(r) for r in eligible]
    for r, text in zip(eligible, texts):
        poison = dict(r, answer='POISON ANSWER MUST NOT AFFECT RETRIEVAL')
        assert lexical_input(poison) == text
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words='english', norm='l2', min_df=1)
    matrix = vec.fit_transform(texts)
    similarities = (vec.transform([lexical_input(r) for r in query]) @ matrix.T).toarray()
    prepared, selections = [], []
    for i, row in enumerate(query):
        ranked = sorted(range(len(eligible)), key=lambda j: (-float(similarities[i, j]), eligible[j]['row_id']))
        chosen = None
        overflow = []
        for rank, j in enumerate(ranked):
            exrow, sim = eligible[j], float(similarities[i, j])
            ex = examples[exrow['row_id']]
            assert not set(ex['codes'].values()) & set(row['code_map'])
            assert not set(ex['patients'].values()) & set(row['patient_ids'])
            messages = [copy.deepcopy(row['messages'][0]), dict(role='user', content=ex['user']),
                        dict(role='assistant', content=ex['answer']), copy.deepcopy(row['messages'][1])]
            native = render(messages)
            n = encoded_length(native)
            if n > 14000:
                overflow.append(dict(row_id=exrow['row_id'], combined_tokens=n))
                continue
            chosen = dict(row_id=row['row_id'], pmc_id=row['pmc_id'], example_row_id=exrow['row_id'],
                          example_pmc_id=exrow['pmc_id'], retrieval_cosine=sim, chosen_rank=rank+1,
                          example_user_tokens=lengths[exrow['row_id']], example_answer_tokens=encoded_length(ex['answer']),
                          original_input_tokens=query_lengths[row['row_id']], combined_input_tokens=n,
                          output_reserve=1600, context_with_reserve=n+1600, overflow_skipped=overflow,
                          mappings=dict(codes=ex['codes'], patients=ex['patients']),
                          full_render_sha256=digest(native), original_user_sha256=digest(row['messages'][1]['content']),
                          original_system_sha256=digest(row['messages'][0]['content']))
            changed = copy.deepcopy(row)
            changed['messages'] = messages
            assert {k:v for k,v in changed.items() if k != 'messages'} == {k:v for k,v in row.items() if k != 'messages'}
            assert messages[0] == row['messages'][0] and messages[-1] == row['messages'][-1]
            prepared.append(changed)
            selections.append(chosen)
            break
        assert chosen is not None, 'No complete example fits; no silent zero-shot fallback'
    with (b/'prompts.jsonl').open('x') as f:
        for row in prepared: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(b/'selections.json', selections)
    save(b/'eligible-examples.json', [dict(row_id=r['row_id'], pmc_id=r['pmc_id'], input_tokens=lengths[r['row_id']]) for r in eligible])
    old = load(refs['old_runtime']['path'])
    orig_tokens = sum(query_lengths.values())
    total = sum(s['combined_input_tokens'] for s in selections)
    output_tokens = sum(v['output_tokens'] for v in old_response.values())
    ratio = (total + output_tokens) / (orig_tokens + output_tokens)
    report = dict(completed_at_utc=now(), scope='CPU readiness only, no model loading/forward/new quality',
        source=ref(__file__), plan=ref(b/'plan.json'), references=refs,
        runtime=dict(python=sys.version, tokenizers=tokenizers.__version__, jinja2=jinja2.__version__, sklearn=sklearn.__version__),
        exact_native_checks=dict(archived124_prefix_text_hash_and_tokens=True, actual80_query_tokens=True,
                                 template_file_equals_config=Path(refs['template']['path']).read_text()==load(refs['tokenizer_config']['path'])['chat_template']),
        isolation=dict(train_documents=32, query_documents=16, intersection=0, all_generator_excludes_query=True),
        retrieval=dict(all_examples=124, eligible_examples=len(eligible), eligible_documents=len({r['pmc_id'] for r in eligible}),
                       example_short_cut=2048, target_used_in_filter_or_TFIDF=False, target_poison_input_test=True,
                       selected_unique_examples=len({s['example_row_id'] for s in selections}),
                       top1_count=sum(s['chosen_rank']==1 for s in selections),
                       overflow_fallback_count=sum(s['chosen_rank']>1 for s in selections),
                       zero_similarity_count=sum(s['retrieval_cosine']==0 for s in selections),
                       selected_similarity=describe([s['retrieval_cosine'] for s in selections])),
        example_integrity=dict(all124_user_and_target_inverse_byteexact=True, boundary_fixture_pass=True,
                               query_all_fields_except_messages_unchanged=True, query_system_user_exact=True,
                               full_targets_retained_including_empty=True, IDs_disjoint_all80=True),
        tokens=dict(original=describe(list(query_lengths.values())), new=describe([s['combined_input_tokens'] for s in selections]),
                    example_user=describe([s['example_user_tokens'] for s in selections]),
                    example_answer=describe([s['example_answer_tokens'] for s in selections]),
                    above14000=0, truncated=0, max_with_output_reserve=max(s['context_with_reserve'] for s in selections)),
        cost=dict(old_actual_same80_seconds=old['elapsed_seconds'], old_batch_seconds=sum(v['batch_seconds'] for v in old_response.values()),
                  old_actual_output_tokens=output_tokens, input_token_ratio=total/orig_tokens,
                  input_plus_old_output_ratio=ratio, all_wall_linear_token_proxy_seconds=old['elapsed_seconds']*ratio,
                  caveat='Proxy only: output behavior/hardware unchanged is unproven; prefill and autoregressive costs differ. No GPU timing here.'),
        artifacts={n:ref(b/n) for n in ['prompts.jsonl','selections.json','eligible-examples.json']},
        no_new_quality=True, no_GPU=True, no_task_fit=True, originaldev_read=False, seconds=time.monotonic()-started)
    save(b/'report.json', report)
    print(json.dumps({k:report[k] for k in ['retrieval','tokens','cost','seconds']}, ensure_ascii=False))


if __name__ == '__main__': main()
