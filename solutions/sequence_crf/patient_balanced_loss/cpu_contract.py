"""CPU-only small-tensor contract; no model, optimizer or real target IDs."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

NATIVE_SHA = '83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'


def ref(path):
    p = Path(path).resolve()
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(value):
    assert ref(value['path']) == value
    return Path(value['path'])


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


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    with path.open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    sync(path.parent)


class Observations:
    """Persist every actual comparison/error before its acceptance assertion."""
    def __init__(self, output):
        self.path = output / 'observations.jsonl'
        self.stream = self.path.open('x')
        self.count = 0

    def write(self, name, passed, **details):
        self.count += 1
        record = dict(index=self.count, name=name, passed=bool(passed), **details)
        self.stream.write(json.dumps(clean(record), allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        sync(self.path.parent)
        return bool(passed)

    def require(self, name, passed, **details):
        recorded = self.write(name, passed, **details)
        assert recorded, name + ': actual observation saved before assertion'

    def compare(self, torch, name, actual, expected, atol, rtol, **details):
        a = actual.detach().to(dtype=torch.float64, device='cpu')
        b = expected.detach().to(dtype=torch.float64, device='cpu')
        if a.shape != b.shape:
            self.require(name + '/shape', False, actual_shape=list(a.shape), expected_shape=list(b.shape))
        finite = torch.isfinite(a) & torch.isfinite(b)
        delta = (a - b).abs()
        allowed = atol + rtol * b.abs()
        ok = bool(torch.all(finite & (delta <= allowed)))
        finite_delta = delta[finite]
        relative = finite_delta / b.abs()[finite].clamp_min(1e-30)
        self.require(name, ok, shape=list(a.shape), absolute_tolerance=atol, relative_tolerance=rtol,
            max_absolute_error=float(finite_delta.max()) if finite_delta.numel() else None,
            max_relative_error=float(relative.max()) if relative.numel() else None,
            actual_nonfinite=int((~torch.isfinite(a)).sum()), expected_nonfinite=int((~torch.isfinite(b)).sum()),
            mismatched_elements=int((~(finite & (delta <= allowed))).sum()), **details)


def plan_check(path):
    plan = json.loads(path.read_text())
    assert plan['schema'] == 'patient-balanced-alpha05-numeric-contract-v1'
    assert plan['execution_ready'] is True, 'Draft plan cannot execute'
    assert plan['budget_seconds'] == 60 and plan['term_grace_seconds'] == 5
    assert plan['expected_torch_version'] == '2.6.0+cu124'
    assert plan['threads'] == 1 and plan['vocabulary_size'] == 11
    assert plan['fp32_tolerance'] == {'atol': 1e-6, 'rtol': 1e-5}
    assert plan['BF16_gradient_tolerance'] == {'atol': 1e-4, 'rtol': .02}
    assert plan['source']['sha256'] == ref(__file__)['sha256'] and plan['source']['bytes'] == ref(__file__)['bytes']
    for value in plan['references'].values():
        checked(value)
    assert plan['references']['native_loss_utils']['sha256'] == NATIVE_SHA
    manifest = json.loads(Path(plan['references']['preparation_manifest']['path']).read_text())
    assert manifest['status'] == 'PASS_FIXED_ALPHA05_TRAIN124_POSITION_AND_COEFFICIENT_PREPARATION'
    assert (manifest['rows'], manifest['documents'], manifest['patients']) == (124, 32, 87)
    assert manifest['full_tokens'] == 400614 and manifest['supervised_tokens'] == 4694
    for key, name in [('row_weights', 'row-weights.jsonl'), ('fraction_contract', 'patient-component-contract.json'),
                      ('epoch_groups', 'epoch-groups.json')]:
        for field in ['bytes', 'sha256']:
            assert plan['references'][key][field] == manifest['outputs'][name][field]
    return plan, manifest


def native_functions(path, torch):
    tree = ast.parse(path.read_text())
    names = ['fixed_cross_entropy', 'ForCausalLMLoss']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert [n.name for n in nodes] == names
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path) + ':two-functions-only', 'exec'), namespace)
    return namespace['fixed_cross_entropy'], namespace['ForCausalLMLoss']


def load_loss(path):
    spec = importlib.util.spec_from_file_location('fixed_alpha05_loss_under_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_layouts(records):
    assert len(records) == 124 and len({r['row_index'] for r in records}) == 124
    selected, used = [], set()
    criteria = [
        ('longest', sorted(records, key=lambda r: (-r['sequence_length'], r['row_index']))),
        ('most_patients', sorted(records, key=lambda r: (-len(r['patients']), -r['sequence_length'], r['row_index']))),
        ('single_patient', sorted([r for r in records if len(r['patients']) == 1], key=lambda r: (-r['sequence_length'], r['row_index']))),
    ]
    for name, candidates in criteria:
        unique = [r for r in candidates if r['row_index'] not in used]
        assert unique, 'No distinct real prepared layout for ' + name
        row = unique[0]
        used.add(row['row_index'])
        # Only position/count metadata and audit hashes are retained. Actual target IDs are absent.
        selected.append((name, copy.deepcopy(row)))
    assert len(selected) == 3
    return selected


def toy_row(pattern=('A', 'B', 'C'), counts=None):
    counts = counts or {'A': 1, 'B': 3, 'C': 7}
    slots = {'A': [4, 5], 'B': [8, 9, 10], 'C': [12]}
    q = list(range(3, 15))
    return dict(row_index=-1, sequence_length=16, supervised_label_positions=q,
        supervised_logit_positions=[x - 1 for x in q], eos_label_positions=[13],
        post_eos_newline_label_positions=[14],
        patients=[dict(patient_index=i, opaque_patient_key='synthetic-' + key, n_p=counts[key],
            value_label_positions=slots[key], value_logit_positions=[x - 1 for x in slots[key]],
            value_token_count=len(slots[key]), prepared_value_empty=(key == 'A'))
            for i, key in enumerate(pattern)])


def tensors(torch, row, dtype, salt, vocab=11):
    length = row['sequence_length'] + 3  # Explicit ignored trailing padding as well as last causal logit.
    arange = torch.arange(length * vocab, dtype=torch.int64, device='cpu')
    values = (((arange * 17 + salt * 13 + 11) % 101) - 50).to(torch.float32) / 17
    logits = values.reshape(1, length, vocab).to(dtype=dtype)
    labels = torch.full((1, length), -100, dtype=torch.long, device='cpu')
    for q in row['supervised_label_positions']:
        labels[0, q] = (q * 3 + salt) % vocab
    return logits, labels


def manual_reference(torch, logits, labels, row, group_size, P, S, alpha):
    """Independent explicit patient/chunk means and analytic softmax derivatives.

    This computes the contribution after the original outer /group_size.
    It never calls cross_entropy, native loss, value_coefficients or autograd.
    """
    q = row['supervised_label_positions']
    selected = logits.detach()[0, [x - 1 for x in q], :].to(torch.float64)
    targets = labels[0, q]
    nll = torch.logsumexp(selected, dim=-1) - selected[torch.arange(len(q)), targets]
    rank = {position: i for i, position in enumerate(q)}
    old = nll.mean()
    patient_terms = []
    for patient in row['patients']:
        indices = [rank[x] for x in patient['value_label_positions']]
        patient_terms.append(nll[indices].mean() / patient['n_p'])
    objective = (1 - alpha) * old / group_size + alpha * S / P * sum(patient_terms)
    coefficient = torch.full((len(q),), (1 - alpha) / (group_size * len(q)), dtype=torch.float64)
    for patient in row['patients']:
        for position in patient['value_label_positions']:
            coefficient[rank[position]] += alpha * S / (P * patient['n_p'] * len(patient['value_label_positions']))
    probabilities = torch.softmax(selected, dim=-1)
    probabilities[torch.arange(len(q)), targets] -= 1
    expected = torch.zeros(logits.shape, dtype=torch.float64, device='cpu')
    expected[0, [x - 1 for x in q], :] = probabilities * coefficient[:, None]
    return objective, expected


def loss_grad(torch, loss_module, native_fixed, logits, labels, row, group_size, P, S, alpha):
    value = logits.detach().clone().requires_grad_(True)
    row_loss = loss_module.mixed_causal_loss(value, labels, row, group_size, P, S, native_fixed, alpha=alpha)
    gradient, = torch.autograd.grad(row_loss / group_size, value)
    return row_loss.detach(), gradient.detach()


def grad_tolerance(torch, dtype):
    return (1e-4, .02) if dtype == torch.bfloat16 else (1e-6, 1e-5)


def compare_gradient(torch, observations, name, actual, analytic, dtype, **details):
    # Exported BF16 input gradients must first be compared to a BF16-rounded expectation.
    expected = analytic.to(dtype=torch.bfloat16) if dtype == torch.bfloat16 else analytic
    atol, rtol = grad_tolerance(torch, dtype)
    observations.compare(torch, name, actual, expected, atol, rtol,
        expectation_rounded_to_BF16=(dtype == torch.bfloat16), **details)


def one_layout(torch, observations, module, native_fixed, native_full, name, row, dtype, salt):
    P, S = 87, 16
    logits, labels = tensors(torch, row, dtype, salt)
    used = set(row['supervised_logit_positions'])
    ignored = [x for x in range(logits.shape[1]) if x not in used]
    observations.require(name + '/ignored-shape', len(ignored) >= 3 and bool(used),
        sequence_length=row['sequence_length'], padded_length=logits.shape[1], ignored_logits=len(ignored),
        supervised_logits=len(used), dtype=str(dtype))
    w4 = module.value_coefficients(row, 4, P, S)
    w8 = module.value_coefficients(row, 8, P, S)
    observations.compare(torch, name + '/value-coefficient-outer-g4-g8',
        torch.tensor(w4, dtype=torch.float64) / 4, torch.tensor(w8, dtype=torch.float64) / 8, 1e-12, 1e-12)
    for group_size in [4, 8]:
        for alpha in [0., .5]:
            tag = name + '/' + str(dtype) + '/g' + str(group_size) + '/a' + str(alpha)
            actual_loss, actual_grad = loss_grad(torch, module, native_fixed, logits, labels, row, group_size, P, S, alpha)
            expected_loss, expected_grad = manual_reference(torch, logits, labels, row, group_size, P, S, alpha)
            observations.compare(torch, tag + '/explicit-patient-chunk-loss', actual_loss / group_size,
                expected_loss, 1e-6, 1e-5, outer_group_division_applied=True)
            compare_gradient(torch, observations, tag + '/analytic-softmax-gradient', actual_grad, expected_grad, dtype,
                             original_q_minus_one_positions=True)
            if alpha == 0.:
                native_input = logits.detach().clone().requires_grad_(True)
                native_loss = native_full(native_input, labels, logits.shape[-1])
                native_grad, = torch.autograd.grad(native_loss / group_size, native_input)
                observations.compare(torch, tag + '/original-full-native-causal-loss', actual_loss, native_loss, 1e-6, 1e-5)
                atol, rtol = grad_tolerance(torch, dtype)
                observations.compare(torch, tag + '/original-full-native-causal-gradient', actual_grad, native_grad, atol, rtol)
                value_positions = {q for patient in row['patients'] for q in patient['value_label_positions']}
                structures = set(row['supervised_label_positions']) - value_positions
                eos_newline = set(row['eos_label_positions'] + row['post_eos_newline_label_positions'])
                nonzero = all(bool(actual_grad[0, q - 1].abs().sum() > 0) for q in structures)
                observations.require(tag + '/all-structure-EOS-newline-retained', nonzero and eos_newline <= structures,
                    structure_tokens=len(structures), eos_and_newline_tokens=len(eos_newline))
            poisoned = logits.detach().clone()
            for i, position in enumerate(ignored):
                poisoned[0, position, :] = [float('nan'), float('inf'), float('-inf')][i % 3]
            poison_loss, poison_grad = loss_grad(torch, module, native_fixed, poisoned, labels, row, group_size, P, S, alpha)
            observations.compare(torch, tag + '/ignored-NaN-Inf-loss-vs-finite-baseline', poison_loss, actual_loss, 1e-6, 1e-5,
                                 reference_is_new_loss_on_finite_input=True, original_full_CE_poison_not_used=True)
            atol, rtol = grad_tolerance(torch, dtype)
            observations.compare(torch, tag + '/ignored-NaN-Inf-gradient-vs-finite-baseline', poison_grad, actual_grad, atol, rtol)
            ignored_values = poison_grad[0, ignored, :]
            observations.require(tag + '/ignored-gradients-exact-zero',
                bool(torch.isfinite(ignored_values).all()) and int(torch.count_nonzero(ignored_values)) == 0,
                nonzero_ignored_gradient_elements=int(torch.count_nonzero(ignored_values)),
                nonfinite_ignored_gradient_elements=int((~torch.isfinite(ignored_values)).sum()))


def rejected_metadata(torch, observations, module, native_fixed):
    base = toy_row()
    logits, labels = tensors(torch, base, torch.float32, 91)
    cases = {}
    shared = copy.deepcopy(base)
    shared['patients'][1]['value_label_positions'] = [4, 9, 10]
    shared['patients'][1]['value_logit_positions'] = [3, 8, 9]
    cases['shared_value_token'] = shared
    empty = copy.deepcopy(base)
    empty['patients'][0]['value_label_positions'] = []
    empty['patients'][0]['value_logit_positions'] = []
    empty['patients'][0]['value_token_count'] = 0
    cases['empty_value_positions'] = empty
    unsupervised = copy.deepcopy(base)
    unsupervised['patients'][0]['value_label_positions'] = [1, 5]
    unsupervised['patients'][0]['value_logit_positions'] = [0, 4]
    cases['unsupervised_value_position'] = unsupervised
    for name, row in cases.items():
        rejected, error_type = False, None
        try:
            module.mixed_causal_loss(logits, labels, row, 4, 87, 16, native_fixed, alpha=.5)
        except (AssertionError, ValueError) as error:
            rejected, error_type = True, type(error).__name__
        observations.require('reject/' + name, rejected, error_type=error_type)
    mismatch = labels.clone()
    mismatch[0, base['supervised_label_positions'][0]] = -100
    rejected, error_type = False, None
    try:
        module.mixed_causal_loss(logits, mismatch, base, 4, 87, 16, native_fixed, alpha=.5)
    except (AssertionError, ValueError) as error:
        rejected, error_type = True, type(error).__name__
    observations.require('reject/metadata-vs-label-supervision-mask', rejected, error_type=error_type)
    weights = module.value_coefficients(base, 4, 87, 16)
    rank = {q: i for i, q in enumerate(base['supervised_label_positions'])}
    empty_item = base['patients'][0]
    observations.require('synthetic-empty-list-value-retained',
        empty_item['prepared_value_empty'] and all(weights[rank[q]] > 0 for q in empty_item['value_label_positions']),
        empty_list_token_positions=empty_item['value_label_positions'])


def real_coefficients(torch, observations, module, records, epoch_groups, fraction):
    """Connect actual loss coefficients to the existing complete Fraction proof."""
    P, S = epoch_groups['patients'], epoch_groups['groups_per_epoch']
    observations.require('real-coefficients/frozen-dimensions', P == 87 and S == 16
        and len(records) == 124 and len(epoch_groups['epochs']) == len(fraction['epochs']) == 2,
        patients=P, groups_per_epoch=S, rows=len(records))
    for epoch, proof in zip(epoch_groups['epochs'], fraction['epochs']):
        accumulated, alpha_accumulated, seen = {}, {}, []
        observations.require('real-coefficients/epoch-identity-' + str(epoch['epoch']),
            epoch['epoch'] == proof['epoch'] and epoch['seed'] == proof['seed']
            and [g['group_size'] for g in epoch['groups']] == [8] * 15 + [4],
            seed=epoch['seed'], group_sizes=[g['group_size'] for g in epoch['groups']])
        for group in epoch['groups']:
            g = group['group_size']
            observations.require('real-coefficients/group-layout-' + str(epoch['epoch']) + '-' + str(group['group_index']),
                g == group['outer_group_divisor'] == len(group['row_indices']),
                group_size=g, outer_group_divisor=group['outer_group_divisor'])
            for index in group['row_indices']:
                row = records[index]
                seen.append(index)
                weights = module.value_coefficients(row, g, P, S)
                rank = {q: i for i, q in enumerate(row['supervised_label_positions'])}
                for patient in row['patients']:
                    key = patient['opaque_patient_key']
                    # Actual loss function's weights, original outer /g, then epoch /S.
                    mass = math.fsum(weights[rank[q]] for q in patient['value_label_positions']) / g / S
                    accumulated.setdefault(key, []).append(mass)
                    alpha_accumulated.setdefault(key, []).append(.5 * mass)
        expected = {p['opaque_patient_key']: p for p in proof['patient_coefficients']}
        observations.require('real-coefficients/epoch-coverage-' + str(epoch['epoch']),
            sorted(seen) == list(range(124)) and set(accumulated) == set(expected)
            and len(expected) == P, actual_rows=len(seen), actual_patients=len(accumulated))
        keys = sorted(expected)
        actual = torch.tensor([math.fsum(accumulated[k]) for k in keys], dtype=torch.float64)
        mixed = torch.tensor([math.fsum(alpha_accumulated[k]) for k in keys], dtype=torch.float64)
        desired = torch.tensor([expected[k]['unscaled']['numerator'] / expected[k]['unscaled']['denominator'] for k in keys], dtype=torch.float64)
        desired_mixed = torch.tensor([expected[k]['mixed']['numerator'] / expected[k]['mixed']['denominator'] for k in keys], dtype=torch.float64)
        observations.compare(torch, 'real-coefficients/actual-loss-vs-Fraction-' + str(epoch['epoch']),
            actual, desired, 1e-12, 0., all_87_patients=True, original_census_not_reexecuted=True)
        observations.compare(torch, 'real-coefficients/actual-half-loss-vs-Fraction-' + str(epoch['epoch']),
            mixed, desired_mixed, 1e-12, 0., all_87_patients=True)
        observations.compare(torch, 'real-coefficients/uniform-1-over-87-' + str(epoch['epoch']),
            actual, torch.full_like(actual, 1 / 87), 1e-12, 0.)
        observations.compare(torch, 'real-coefficients/uniform-1-over-174-' + str(epoch['epoch']),
            mixed, torch.full_like(mixed, 1 / 174), 1e-12, 0.)


def repeated_chunk_group(torch, observations, module, native_fixed, dtype):
    # Four original synthetic rows: A has three chunks, B two, C one.
    # Repeating the rows gives an eight-row group with twice each n_p.
    patterns = [('A', 'B'), ('A',), ('A', 'C'), ('B',)]
    group_results = {}
    for group_size in [4, 8]:
        repetition = group_size // 4
        counts = {'A': 3 * repetition, 'B': 2 * repetition, 'C': repetition}
        rows = [toy_row(pattern, counts) for pattern in patterns * repetition]
        variables, losses, expected_grads = [], [], []
        ce_by_patient = {key: [] for key in ['A', 'B', 'C']}
        old_means = []
        for i, row in enumerate(rows):
            logits, labels = tensors(torch, row, dtype, i % 4 + 101)
            value = logits.clone().requires_grad_(True)
            variables.append(value)
            losses.append(module.mixed_causal_loss(value, labels, row, group_size, 3, 1, native_fixed, alpha=.5) / group_size)
            _, expected_grad = manual_reference(torch, logits, labels, row, group_size, 3, 1, .5)
            expected_grads.append(expected_grad)
            q = row['supervised_label_positions']
            z = logits[0, [x - 1 for x in q], :].to(torch.float64)
            targets = labels[0, q]
            ce = torch.logsumexp(z, -1) - z[torch.arange(len(q)), targets]
            old_means.append(ce.mean())
            rank = {position: index for index, position in enumerate(q)}
            for patient in row['patients']:
                key = patient['opaque_patient_key'].removeprefix('synthetic-')
                ce_by_patient[key].append(ce[[rank[x] for x in patient['value_label_positions']]].mean())
        actual = sum(losses)
        explicit = .5 * sum(old_means) / group_size + .5 / 3 * sum(sum(values) / len(values) for values in ce_by_patient.values())
        gradients = torch.autograd.grad(actual, variables)
        tag = 'repeated-chunks/' + str(dtype) + '/g' + str(group_size)
        observations.compare(torch, tag + '/explicit-patient-then-chunk-mean', actual, explicit, 1e-6, 1e-5,
                             actual_chunks_per_patient=counts, outer_group_division_applied=True)
        for i, (actual_grad, expected_grad) in enumerate(zip(gradients, expected_grads)):
            compare_gradient(torch, observations, tag + '/row' + str(i) + '/analytic-gradient', actual_grad, expected_grad, dtype)
        group_results[group_size] = (actual.detach(), [g.detach() for g in gradients])
    observations.compare(torch, 'repeated-chunks/' + str(dtype) + '/g4-g8-objective-identical',
        group_results[4][0], group_results[8][0], 1e-6, 1e-5)
    atol, rtol = grad_tolerance(torch, dtype)
    for i in range(4):
        observations.compare(torch, 'repeated-chunks/' + str(dtype) + '/g4-g8-aggregated-gradient-' + str(i),
            group_results[8][1][i] + group_results[8][1][i + 4], group_results[4][1][i], atol, rtol)


def worker(args):
    started = time.monotonic()
    plan, manifest = plan_check(args.plan)
    output = Path(plan['output_directory'])
    observations = Observations(output)
    observations.require('CPU-environment-before-Torch', os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), model_imports_forbidden=True)
    def forbid_models(event, values):
        if event == 'import' and values[0].split('.')[0] in {'transformers', 'peft', 'accelerate', 'datasets'}:
            raise PermissionError('Model/framework import forbidden in pure CPU loss contract')
    sys.addaudithook(forbid_models)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    observations.require('original-native-runtime', str(torch.__version__) == plan['expected_torch_version']
        and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
        actual_torch=str(torch.__version__), expected_torch=plan['expected_torch_version'],
        actual_threads=torch.get_num_threads(), actual_interop_threads=torch.get_num_interop_threads())
    native_fixed, native_full = native_functions(Path(plan['references']['native_loss_utils']['path']), torch)
    module = load_loss(Path(plan['references']['mixed_loss']['path']))
    records = [json.loads(line) for line in Path(plan['references']['row_weights']['path']).read_text().splitlines()]
    epoch_groups = json.loads(Path(plan['references']['epoch_groups']['path']).read_text())
    fraction = json.loads(Path(plan['references']['fraction_contract']['path']).read_text())
    real = select_layouts(records)
    observations.require('real-layout-selection-bound-to-plan',
        [dict(reason=name, row_index=row['row_index']) for name, row in real] == plan['selected_layouts_expected'],
        actual=[dict(reason=name, row_index=row['row_index']) for name, row in real])
    save(output / 'selected-layouts.json', dict(selection='longest, most patients, single patient; distinct rows, stable original index tie-break',
        rows=[dict(reason=name, row_index=row['row_index'], sequence_length=row['sequence_length'],
            patient_count=len(row['patients']), supervised_tokens=len(row['supervised_label_positions']),
            native_audit_sha256=hashlib.sha256(json.dumps(row['native_audit'], sort_keys=True).encode()).hexdigest()) for name, row in real],
        actual_training_target_ids_read=False, logits_and_labels_synthetic=True))
    real_coefficients(torch, observations, module, records, epoch_groups, fraction)
    rejected_metadata(torch, observations, module, native_fixed)
    cases = [('synthetic-multiple-patients-unequal-chunks', toy_row())] + real
    for dtype in [torch.float32, torch.bfloat16]:
        for index, (name, row) in enumerate(cases):
            one_layout(torch, observations, module, native_fixed, native_full, name, row, dtype, index + 1)
        repeated_chunk_group(torch, observations, module, native_fixed, dtype)
    observations.require('existing-real124-Fraction-proof-only-reference',
        fraction['exact_rational_arithmetic'] and len(fraction['epochs']) == 2
        and all(e['mean_group_unscaled_coefficient_per_patient'] == {'numerator': 1, 'denominator': 87}
                and e['mean_group_mixed_coefficient_per_patient'] == {'numerator': 1, 'denominator': 174}
                for e in fraction['epochs']), original_census_not_reexecuted=True,
        proof=plan['references']['fraction_contract'])
    observations.require('no-model-framework-loaded',
        not any(name.split('.')[0] in {'transformers', 'peft', 'accelerate', 'datasets'} for name in sys.modules),
        no_model_or_optimizer_created=True, no_actual_target_values_loaded=True)
    observations.stream.close()
    save(output / 'result.json', dict(status='PASS_CPU_SYNTHETIC_LOSS_AND_LOGITS_GRADIENT_CONTRACT',
        plan=ref(args.plan), source=ref(__file__), references=plan['references'],
        actual_torch=str(torch.__version__), threads=1, CUDA_VISIBLE_DEVICES='', observations=ref(output / 'observations.jsonl'),
        selected_layouts=ref(output / 'selected-layouts.json'), comparisons=observations.count,
        alpha_values=[0., .5], alpha_zero_is_contract_anchor_only=True, group_sizes=[4, 8],
        prepared_layouts=3, vocabulary_size=11, tensors_device='cpu', native_source_AST_functions_only=True,
        fp32_tolerance=dict(atol=1e-6, rtol=1e-5), BF16_gradient_tolerance=dict(atol=1e-4, rtol=.02),
        BF16_expected_gradient_rounded_before_comparison=True, all_actual_errors_saved_before_assert=True,
        no_actual_training_target_ids=True, no_model_forward_or_training=True, no_GPU_calls=True,
        no_claim_about_LoRA_updates_or_actual_gradient_causality_or_quality_or_GPU_contract=True,
        original_Fraction_proof_referenced_not_recomputed=plan['references']['fraction_contract'],
        actual_loss_coefficient_bridge_both_epochs_all124_and87_patients=True,
        seconds=time.monotonic() - started))
    print(json.dumps(ref(output / 'result.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    assert __debug__
    os.umask(0o077)
    if args.worker:
        try:
            worker(args)
        except BaseException:
            plan = json.loads(args.plan.read_text())
            output = Path(plan['output_directory'])
            save(output / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
            raise
        return
    plan, _ = plan_check(args.plan)
    output = Path(plan['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'launch.json', dict(plan=ref(args.plan), source=ref(__file__),
        budget_seconds=60, term_grace_seconds=5, CUDA_VISIBLE_DEVICES='', threads=1, automatic_retry=False))
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--plan', str(args.plan.resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    timed_out = False
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            save(output / 'child-started.json', dict(child_pid=process.pid, started_unix=time.time(), budget_seconds=60))
            rc = process.wait(timeout=max(.001, started + 60 - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                rc = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait()
    save(output / 'terminal.json', dict(returncode=rc, timed_out=timed_out, seconds=time.monotonic() - started,
        budget_seconds=60, term_grace_seconds=5, child_pid=process.pid, automatic_retry=False,
        result=ref(output / 'result.json') if (output / 'result.json').exists() else None))
    print(json.dumps(ref(output / 'terminal.json')))
    if rc or timed_out:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
