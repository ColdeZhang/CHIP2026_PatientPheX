"""Derive frozen train124 value-token positions and exact alpha=.5 coefficients."""
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jhash(value):
    # Preserve the existing native audit's default json.dumps spacing.
    return sha(json.dumps(value).encode())


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=sha(data))


def ratio(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def sync(path):
    while True:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if path.parent == path:
            break
        path = path.parent


def save(path, value, jsonl=False):
    data = (''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in value) if jsonl
            else json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    sync(path.parent)


def extract(path, name, kind, namespace):
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, kind) and n.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + ':' + name, 'exec'), namespace)
    return namespace[name]


def patient_key(pmc_id, patient_id):
    return sha(('patient-loss-census-v1:' + str(pmc_id) + ':' + patient_id).encode())


def value_owners(row):
    """The original census's compact JSON cursor and native character spans."""
    answer = row['answer']
    obj = json.loads(answer)
    assert answer == compact(obj)
    assert list(obj) == row['patient_ids'] and len(obj) == len(set(row['patient_ids']))
    owners = [None] * len(answer)
    ranges = {}
    cursor = 1
    assert answer.startswith('{') and answer.endswith('}')
    for index, (pid, codes) in enumerate(obj.items()):
        if index:
            assert answer[cursor] == ','
            cursor += 1
        assert isinstance(pid, str) and isinstance(codes, list)
        assert all(isinstance(code, str) and code in row['code_map'] for code in codes)
        key = json.dumps(pid, ensure_ascii=False)
        assert answer[cursor:cursor + len(key)] == key
        cursor += len(key)
        assert answer[cursor] == ':'
        cursor += 1
        value = compact(codes)
        start = cursor
        assert answer[start:start + len(value)] == value
        cursor += len(value)
        for position in range(start, cursor):
            owners[position] = pid
        ranges[pid] = (start, cursor)
    assert cursor == len(answer) - 1
    return obj, owners, ranges


def derive_row(index, row, audit, tokenizer, counts):
    answer = row['answer']
    obj, owners, ranges = value_owners(row)
    prefix = tokenizer.apply_chat_template(row['messages'], False, True, False)
    full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': answer}], False, False, False)
    assert full.startswith(prefix) and full[len(prefix):] == answer + '<|im_end|>\n'
    prefix_encoding = tokenizer.backend.encode(prefix, add_special_tokens=False)
    encoding = tokenizer.backend.encode(full, add_special_tokens=False)
    prefix_length = len(prefix_encoding.ids)
    assert encoding.ids[:prefix_length] == prefix_encoding.ids
    assert len(encoding.ids) == len(encoding.offsets)
    labels = [-100] * prefix_length + encoding.ids[prefix_length:]
    positions = list(range(prefix_length, len(encoding.ids)))
    assert prefix_length > 0 and positions
    actual = dict(row_id=row['row_id'], pmc_id=str(row['pmc_id']),
        full_tokens=len(encoding.ids), prefix_tokens=prefix_length, supervised_tokens=len(positions),
        eos_token_id=tokenizer.eos_token_id, rendered_prefix_sha256=sha(prefix.encode()),
        rendered_full_sha256=sha(full.encode()), input_ids_sha256=jhash(encoding.ids), labels_sha256=jhash(labels))
    assert actual == audit['patient_first'], (row['row_id'], 'original native audit differs')
    assert audit['row_id'] == row['row_id']
    assert [q for q, value in enumerate(labels) if value != -100] == positions
    assert actual['full_tokens'] <= 6144
    values = {pid: [] for pid in obj}
    covered = {pid: set() for pid in obj}
    eos_positions, newline_positions = [], []
    for q in positions:
        token_id = encoding.ids[q]
        start, end = encoding.offsets[q]
        start -= len(prefix)
        end -= len(prefix)
        assert end > start
        if token_id == tokenizer.eos_token_id:
            eos_positions.append(q)
            continue
        if start >= len(answer):
            assert full[len(prefix) + start:len(prefix) + end] == '\n'
            newline_positions.append(q)
            continue
        assert 0 <= start < end <= len(answer)
        intersected = set(owners[start:end]) - {None}
        # A whole BPE token can belong to one value, including its boundary punctuation.
        # Never split its CE by characters or silently share it across two values.
        assert len(intersected) <= 1, (row['row_id'], q, 'token intersects multiple patient values')
        if intersected:
            pid = next(iter(intersected))
            values[pid].append(q)
            covered[pid].update(z for z in range(start, end) if owners[z] == pid)
    assert len(eos_positions) == len(newline_positions) == 1
    assert eos_positions[0] < newline_positions[0] and newline_positions[0] == positions[-1]
    patients = []
    all_value_positions = []
    for patient_index, (pid, codes) in enumerate(obj.items()):
        qvalues = values[pid]
        start, end = ranges[pid]
        assert qvalues and len(qvalues) == len(set(qvalues))
        assert covered[pid] == set(range(start, end))
        assert all(q in positions and q >= 1 for q in qvalues)
        key = patient_key(row['pmc_id'], pid)
        n_p = counts[key]
        assert n_p > 0
        patients.append(dict(patient_index=patient_index, opaque_patient_key=key, n_p=n_p,
            value_label_positions=qvalues, value_logit_positions=[q - 1 for q in qvalues],
            value_token_count=len(qvalues), prepared_value_empty=(len(codes) == 0),
            patient_chunk_mean_weight=ratio(Fraction(1, n_p))))
        all_value_positions.extend(qvalues)
    assert len(all_value_positions) == len(set(all_value_positions))
    return dict(row_index=index, row_id=row['row_id'], pmc_id=str(row['pmc_id']), native_audit=actual,
        sequence_length=len(encoding.ids), prefix_length=prefix_length,
        supervised_label_positions=positions, supervised_logit_positions=[q - 1 for q in positions],
        eos_label_positions=eos_positions, post_eos_newline_label_positions=newline_positions,
        patients=patients, target_position_coordinate='full labels index q; causal logit index q-1')


def groups_and_contract(records, counts, batch_plan, expected):
    alpha = Fraction(1, 2)
    P, S = expected['patients'], expected['groups_per_epoch']
    epochs, contracts = [], []
    for epoch_index in range(2):
        original_groups = batch_plan(len(records), 1, 8, 20261023, epoch_index)
        assert [len(g) for g in original_groups] == [8] * 15 + [4]
        groups = []
        coefficients = defaultdict(Fraction)
        mixed_coefficients = defaultdict(Fraction)
        appearances = Counter()
        order = []
        for group_index, original_group in enumerate(original_groups):
            assert all(len(batch) == 1 for batch in original_group)
            indices = [batch[0] for batch in original_group]
            order.extend(indices)
            size = len(indices)
            terms = []
            for index in indices:
                patients = []
                for item in records[index]['patients']:
                    key, n_p, value_count = item['opaque_patient_key'], item['n_p'], item['value_token_count']
                    unscaled = Fraction(S * size, P * n_p)
                    mixed = alpha * unscaled
                    # Along the unchanged old loss/len(group) backward, group_size
                    # cancels exactly; averaging the S objectives yields 1/(P*n_p).
                    token_unscaled = unscaled / value_count
                    token_mixed = mixed / value_count
                    coefficients[key] += token_unscaled * value_count / size / S
                    mixed_coefficients[key] += token_mixed * value_count / size / S
                    appearances[key] += 1
                    patients.append(dict(opaque_patient_key=key, n_p=n_p, value_token_count=value_count,
                        unscaled_value_mean_coefficient_before_outer_group_division=ratio(unscaled),
                        mixed_value_mean_coefficient_before_outer_group_division=ratio(mixed),
                        mixed_per_value_token_coefficient_before_outer_group_division=ratio(token_mixed)))
                terms.append(dict(row_index=index, old_mean_coefficient_before_outer_group_division=ratio(1 - alpha), patients=patients))
            groups.append(dict(group_index=group_index, group_size=size, row_indices=indices,
                outer_group_divisor=size, row_coefficients=terms))
        assert sorted(order) == list(range(len(records)))
        assert appearances == counts and len(coefficients) == P
        assert all(value == Fraction(1, P) for value in coefficients.values())
        assert all(value == alpha / P for value in mixed_coefficients.values())
        assert sum(coefficients.values()) == 1 and sum(mixed_coefficients.values()) == alpha
        contracts.append(dict(epoch=epoch_index + 1, seed=20261023 + epoch_index,
            all_patients_exact_uniform_unscaled=True, all_patients_exact_uniform_after_alpha=True,
            mean_group_unscaled_coefficient_per_patient=ratio(Fraction(1, P)),
            mean_group_mixed_coefficient_per_patient=ratio(alpha / P),
            patient_coefficients=[dict(opaque_patient_key=key, n_p=counts[key],
                unscaled=ratio(coefficients[key]), mixed=ratio(mixed_coefficients[key])) for key in sorted(counts)],
            tail_group_size=len(groups[-1]['row_indices']),
            tail_group_row_id_sha256=[sha(records[i]['row_id'].encode()) for i in groups[-1]['row_indices']]))
        epochs.append(dict(epoch=epoch_index + 1, seed=20261023 + epoch_index, groups=groups))
    return epochs, contracts


def run(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    assert plan['schema'] == 'patient-balanced-alpha05-preparation-v1'
    assert plan['source'] == ref(__file__)
    assert plan['alpha'] == {'numerator': 1, 'denominator': 2}
    assert plan['expected']['rows'] == 124 and plan['expected']['patients'] == 87
    assert 'torch' not in sys.modules
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    train_path = Path(plan['references']['train']['path']).resolve()
    derived_path = (Path(plan['output_directory']) / 'row-weights.jsonl').resolve()
    def audit(event, values):
        if event == 'import' and (values[0] == 'torch' or values[0].startswith(('torch.', 'transformers'))):
            raise PermissionError('No Torch or model imports in value-position preparation')
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            mode = values[1]
            if str(path).endswith('.jsonl') and (mode is None or 'r' in mode):
                if path not in {train_path, derived_path}:
                    raise PermissionError('Only frozen training JSONL and own derived positions may be read')
    sys.addaudithook(audit)
    for item in plan['references'].values():
        assert ref(item['path']) == item
    # Merely extract the two frozen pure helpers; never import their training modules.
    sys.path.insert(0, plan['cpu_libraries'])
    tokenizer_type = extract(plan['references']['tokenizer_helper']['path'], 'FrozenTokenizer', ast.ClassDef, {'Path': Path})
    batch_plan = extract(plan['references']['trainer']['path'], 'batch_plan', ast.FunctionDef, {'random': random})
    tokenizer = tokenizer_type(plan['references']['tokenizer']['path'], plan['references']['template']['path'])
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    audits = json.loads(Path(plan['references']['token_audit']['path']).read_text())
    expected = plan['expected']
    assert len(train) == len(audits) == expected['rows']
    assert len({row['row_id'] for row in train}) == len(train)
    assert [row['row_id'] for row in train] == [item['row_id'] for item in audits]
    provenance = json.loads(Path(plan['references']['provenance']['path']).read_text())
    documents = {str(row['pmc_id']) for row in train}
    assert len(documents) == expected['documents'] == 32
    assert documents == set(provenance['training_document_ids'])
    assert not documents & set(provenance['excluded_document_ids'])
    counts = Counter(patient_key(row['pmc_id'], pid) for row in train for pid in row['patient_ids'])
    assert len(counts) == expected['patients'] == 87
    assert sum(counts.values()) == expected['patient_row_occurrences'] == 346
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'launch.json', dict(plan=ref(args.plan), source=ref(__file__), cpu_only=True,
         no_model_or_Torch=True, originals_unchanged=True))
    records = [derive_row(i, row, audit, tokenizer, counts) for i, (row, audit) in enumerate(zip(train, audits))]
    assert sum(item['sequence_length'] for item in records) == expected['full_tokens'] == 400614
    assert sum(item['prefix_length'] for item in records) == expected['prefix_tokens'] == 395920
    assert sum(len(item['supervised_label_positions']) for item in records) == expected['supervised_tokens'] == 4694
    assert max(item['sequence_length'] for item in records) == expected['max_sequence_length'] == 5556
    assert sum(p['prepared_value_empty'] for item in records for p in item['patients']) == expected['empty_patient_values'] == 106
    epochs, contracts = groups_and_contract(records, counts, batch_plan, expected)
    assert [item['tail_group_row_id_sha256'] for item in contracts] == plan['original_tail_group_row_id_sha256']
    save(output / 'row-weights.jsonl', records, jsonl=True)
    save(output / 'epoch-groups.json', dict(alpha=plan['alpha'], patients=87, groups_per_epoch=16,
         original_outer_backward_divides_by_group_size=True, epochs=epochs))
    save(output / 'patient-component-contract.json', dict(epochs=contracts,
         exact_rational_arithmetic=True, no_character_fractional_assignment=True,
         token_crossing_two_values_policy='reject', observed_cross_value_tokens=0,
         original_token_component_retains_all_supervised_positions=True,
         prepared_empty_does_not_mean_complete_gold_empty=True,
         coefficients_not_gradients_or_optimizer_effects=True))
    save(output / 'patient-counts.json', [dict(opaque_patient_key=key, n_p_per_epoch=counts[key]) for key in sorted(counts)])
    assert 'torch' not in sys.modules
    # Inputs remain immutable, including native target tokens and both original pure helpers.
    for item in plan['references'].values():
        assert ref(item['path']) == item
    save(output / 'manifest.json', dict(status='PASS_FIXED_ALPHA05_TRAIN124_POSITION_AND_COEFFICIENT_PREPARATION',
         plan=ref(args.plan), source=ref(__file__), references=plan['references'],
         outputs={name: ref(output / name) for name in ['row-weights.jsonl', 'epoch-groups.json', 'patient-component-contract.json', 'patient-counts.json']},
         rows=124, documents=32, patients=87, full_tokens=400614, prefix_tokens=395920,
         supervised_tokens=4694, max_sequence_length=5556, all_original_native_audit_fields_exact=True,
         supervised_positions_are_full_label_indices=True, causal_logits_are_q_minus_one=True,
         original_input_order_and_target_bytes_unchanged=True, epochs=2, groups_per_epoch=16,
         alpha=plan['alpha'], no_NN_or_fit_or_quality_score=True, no_heldout_gold_or_predictions=True,
         torch_imported=False, python_version=sys.version,
         tokenizer_version=sys.modules['tokenizers'].__version__,
         jinja2_version=sys.modules['jinja2'].__version__, seconds=time.monotonic() - started))
    print(json.dumps(ref(output / 'manifest.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    assert __debug__
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    assert plan['cpu_timeout_seconds'] == 120
    assert not Path(plan['output_directory']).exists(), 'Existing preparation is immutable; do not rerun'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU preparation budget exceeded')))
    signal.alarm(plan['cpu_timeout_seconds'])
    try:
        run(args)
    except BaseException:
        output = Path(plan['output_directory'])
        if output.exists() and not (output / 'failure.json').exists():
            save(output / 'failure.json', dict(traceback=traceback.format_exc(), no_automatic_retry=True))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
