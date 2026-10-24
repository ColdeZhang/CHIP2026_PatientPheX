#!/usr/bin/env python3
"""Pure synthetic CPU interface replay; no project gold, model or scoring calls."""
import argparse
import ast
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve().with_name('evaluate_paired.py')
spec = importlib.util.spec_from_file_location('paired_contract_cpu', ENTRY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
TRAINER = 'a' * 64
CASES = []


def mark(name):
    CASES.append(name)


def rejects(name, function):
    try:
        function()
    except (ValueError, KeyError):
        mark(name)
    else:
        raise AssertionError(name)


def fixture(arm):
    variants = ['O', 'O'] if arm == 'original' else ['R', 'O']
    warm = dict(kind='new_stage_adapter_warm_start_fresh_optimizer_RNG',
                source_checkpoint_sha256=m.WARM_CP, adapter_files=m.WARM_FILES,
                source_epochs=2, source_optimizer_calls=32, old_optimizer_loaded=False,
                old_training_state_opened=False, not_exact_old_stage_resume=True)
    fp = dict(schema=m.STAGE_SCHEMA, source_sha256=TRAINER, order_arm=arm, epoch_variants=variants,
              train_sha256=m.TRAIN_SHA, excluded_input_sha256=m.EXCLUDED_SHA,
              reverse_train_sha256=m.REVERSE_SHA, training_document_ids=[str(i) for i in range(32)],
              excluded_document_ids=[str(i) for i in range(32, 64)], warm_start=warm,
              configuration={'fixed': {'seed': 20261023},
                             'paths_and_device': {'output_dir': arm, 'order_arm': arm, 'gpu_uuid': 'same'}})
    cp = dict(fingerprint=fp, completed_epochs=2, global_step=32, smoke_only=False,
              files={'adapter/adapter_model.safetensors': ('b' if arm == 'original' else 'c') * 64,
                     'adapter/adapter_config.json': 'd' * 64, 'training_state.pt': 'e' * 64})
    manifest = dict(fingerprint=fp, warm_start=warm, examples=124, steps_per_epoch=16, epochs=2,
                    new_stage_epochs=2, old_stage_completed_epochs=2, total_optimizer_steps=32,
                    warmup_steps=2, smoke_only=False, validation_labels_used=False,
                    fresh_optimizer_on_initial_run=True, fresh_base_on_initial_run=False, epoch_variants=variants)
    terminal = dict(source_sha256=TRAINER, order_arm=arm, warm_start=warm, epochs=2, new_stage_epochs=2,
                    old_stage_epochs=2, global_step=32, smoke_only=False, base_parameters=8953803264,
                    trainable_adapter_parameters=43278336, total_parameters=8997081600,
                    started_completed_epochs=1, elapsed_scope='current_invocation_only_not_full_stage',
                    full_stage_elapsed_seconds=None)
    return cp, manifest, terminal


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    started = time.monotonic()
    ast.parse(ENTRY.read_text())
    for arm in m.ARMS:
        cp, manifest, terminal = fixture(arm)
        assert 'partial_not_for_inference' not in cp
        m.verify_training(arm, cp, manifest, terminal, TRAINER)
        mark('complete_old_schema_no_partial_' + arm)
    invalid = [('epoch', 'cp', 'completed_epochs', 1), ('calls', 'cp', 'global_step', 31),
               ('smoke', 'cp', 'smoke_only', True), ('labels', 'manifest', 'validation_labels_used', True),
               ('optimizer', 'manifest', 'fresh_optimizer_on_initial_run', False),
               ('terminal_calls', 'terminal', 'global_step', 64),
               ('elapsed_scope', 'terminal', 'elapsed_scope', 'full_stage')]
    for label, section, key, value in invalid:
        cp, manifest, terminal = fixture('original')
        {'cp': cp, 'manifest': manifest, 'terminal': terminal}[section][key] = value
        rejects('reject_' + label, lambda: m.verify_training('original', cp, manifest, terminal, TRAINER))
    cp, manifest, terminal = fixture('original')
    cp['fingerprint']['warm_start']['old_optimizer_loaded'] = True
    rejects('reject_old_optimizer', lambda: m.verify_training('original', cp, manifest, terminal, TRAINER))
    cp, manifest, terminal = fixture('original')
    cp['files']['adapter/adapter_model.safetensors'] = m.WARM_FILES['adapter_model.safetensors']
    rejects('reject_unchanged_adapter', lambda: m.verify_training('original', cp, manifest, terminal, TRAINER))
    cp, manifest, terminal = fixture('original')
    cp['fingerprint']['training_document_ids'][0] = '32'
    rejects('reject_overlap', lambda: m.verify_training('original', cp, manifest, terminal, TRAINER))
    left, right = fixture('original')[0]['fingerprint'], fixture('reverse-first')[0]['fingerprint']
    assert m.comparable_fingerprint(left) == m.comparable_fingerprint(right)
    mark('matched_arms_only_order_output_differ')
    right['configuration']['fixed']['seed'] = 1
    assert m.comparable_fingerprint(left) != m.comparable_fingerprint(right)
    mark('detect_seed_drift')
    base = {k: .5 for k in m.METRICS}
    candidate = {**base, 'Score': .51}
    assert m.keep_suggestion(candidate, base) and not m.keep_suggestion(base, base)
    assert not m.keep_suggestion({**candidate, 'macF1': .49}, base)
    mark('strict_score_and_nondecreasing_macro')
    rejects('reject_nonfinite', lambda: m.delta({**base, 'Score': float('nan')}, base))
    original_sha, original_shared, original_gold = m.sha, m.SHARED_SHA.copy(), m.GOLD_SHA
    original_argv = list(sys.argv)
    with tempfile.TemporaryDirectory(prefix='paired-source-contract-') as temp:
        root = Path(temp)
        data = root / 'synthetic'
        data.mkdir()
        events = []

        def dump(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return m.ref(path)

        def dumprows(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row) + '\n' for row in value))
            return m.ref(path)

        baseline = [dict(pmc_id=str(i), pmid=None, entities=[{'synthetic': True}] * (74 if i < 15 else 79),
                         association=[]) for i in range(16)]
        shared = {k: dump(data / (k + '.json'), {}) for k in original_shared}
        shared['parser'], shared['parser_base'] = dump(data / 'merge.py', {}), dump(data / 'run.py', {})
        shared['old_baseline_pred'] = dumprows(data / 'oldpred.jsonl', baseline)
        shared['old_baseline_metrics'] = dump(data / 'oldmetrics.json', {'metrics': base})
        prompts = [{'code_map': {str(j): 'x' for j in range(8 if i < 27 else 7)}} for i in range(80)]
        assert sum(len(p['code_map']) for p in prompts) == 587
        shared['prompts'] = dumprows(data / 'prompts.jsonl', prompts)
        # These source pins apply ONLY to this manufactured in-process fixture.
        m.SHARED_SHA = {k: record['sha256'] for k, record in shared.items()}
        gold = data / 'synthetic-gold.txt'
        gold.write_text('synthetic sentinel, never actual labels')
        m.GOLD_SHA = original_sha(gold)
        arms = {}
        for arm in m.ARMS:
            directory = data / arm
            cp, manifest, terminal = fixture(arm)
            refs = {'training_manifest': dump(directory / 'run_manifest.json', manifest),
                    'training_complete': dump(directory / 'training_complete.json', terminal),
                    'checkpoint': dump(directory / 'epoch-002/checkpoint.json', cp)}
            files = {name: {'bytes': 1, 'sha256': digest} for name, digest in cp['files'].items()}
            files.update({name: {'bytes': refs[key]['bytes'], 'sha256': refs[key]['sha256']}
                          for name, key in [('checkpoint.json', 'checkpoint'),
                                            ('_run/run_manifest.json', 'training_manifest')]})
            receipt = dict(status='persistent_checkpoint_verified', persistent_mount_id=m.MOUNT_ID,
                           fsync_all_files=True, completed_epochs=2, global_step=32,
                           source_checkpoint_directory='/dev/shm/synthetic/' + arm + '/epoch-002',
                           persistent_directory='case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/' + arm + '/epoch-002',
                           checkpoint_sha256=refs['checkpoint']['sha256'],
                           run_manifest_sha256=refs['training_manifest']['sha256'], files=files)
            refs['export_ack'] = dump(directory / 'ack.json', receipt)
            m.verify_export(receipt, cp, refs)
            mark('export_positive_' + arm)
            bad_receipt = copy.deepcopy(receipt)
            bad_receipt['files']['adapter/adapter_model.safetensors']['sha256'] = '0' * 64
            rejects('reject_actual_adapter_receipt_' + arm, lambda: m.verify_export(bad_receipt, cp, refs))
            if arm == 'original':
                m.verify_export(receipt, cp, refs)
                mark('allow_actual_mount_relative_prefix')
                paths = [('absolute', '/' + receipt['persistent_directory']), ('wrong_prefix', 'elsewhere/epoch-002'),
                         ('parent_traversal', 'case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/../outside'),
                         ('no_descendant', 'case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints')]
                for name, path in paths:
                    bad_receipt = copy.deepcopy(receipt)
                    bad_receipt['persistent_directory'] = path
                    rejects('reject_path_' + name, lambda: m.verify_export(bad_receipt, cp, refs))
                bad_receipt = copy.deepcopy(receipt)
                bad_receipt['persistent_mount_id'] = 'other'
                rejects('reject_other_mount', lambda: m.verify_export(bad_receipt, cp, refs))
            adapter = {name: cp['files']['adapter/' + name] for name in m.WARM_FILES}
            infer = dict(checkpoint_sha256=refs['checkpoint']['sha256'],
                         training_complete_sha256=refs['training_complete']['sha256'], adapter_files=adapter,
                         adapter_directory='/synthetic/adapter/' + arm, prompts_sha256=shared['prompts']['sha256'],
                         runner_sha256=shared['runner']['sha256'], model_source_sha256=shared['model_source']['sha256'])
            refs['inference_plan'] = dump(directory / 'inference.json', infer)
            refs['responses'] = dumprows(directory / 'responses.jsonl', [{'row_id': str(i)} for i in range(80)])
            refs['response_manifest'] = dump(directory / 'responses.manifest.json',
                                            {'fingerprint': {'configuration': {'adapter': infer['adapter_directory']}}})
            refs['response_freeze'] = dump(directory / 'responses-freeze.json',
                                          dict(plan_sha256=refs['inference_plan']['sha256'],
                                               raw_sha256=refs['responses']['sha256'],
                                               manifest_sha256=refs['response_manifest']['sha256'], rows=80))
            arms[arm] = {'references': refs}
        plan = dict(schema=m.SCHEMA, control_sha256=original_sha(ENTRY), python=sys.executable,
                    references=shared, gold_path=str(gold), historical_F1_gold_sha256=m.GOLD_SHA,
                    trainer_sha256=TRAINER, arms=arms)
        m.plan_references(plan)
        mark('reference_whitelist_positive')
        bad_plan = copy.deepcopy(plan)
        bad_plan['references']['input']['path'] = str(gold)
        rejects('reject_gold_as_reference', lambda: m.plan_references(bad_plan))
        hardlink = data / 'gold-hardlink'
        hardlink.hardlink_to(gold)
        bad_plan = copy.deepcopy(plan)
        bad_plan['references']['input']['path'] = str(hardlink)
        rejects('reject_gold_hardlink', lambda: m.plan_references(bad_plan))

        def traced_sha(path):
            if Path(path) == gold:
                assert (current / 'paired-predictions-freeze.json').exists()
                assert all((current / arm / 'pred.jsonl').exists() for arm in m.ARMS)
                events.append('gold_bytes')
            return original_sha(path)

        m.sha = traced_sha
        tree = types.SimpleNamespace(verify_parser=lambda path: None, pair_rows=lambda prompts: [{}] * 1557,
                                     llm_features=lambda *args: (None, {'valid_candidate_decisions': 587,
                                                                       'unresolved_candidate_decisions': 0}))
        fusion = types.SimpleNamespace(verify_sft_response=lambda *args: None)
        m.module = lambda name, path: tree if 'tree' in name else fusion

        def fake_command(argv, output):
            target = Path(argv[argv.index('--output') + 1])
            if argv[1] == shared['parser']['path']:
                events.append('assembly')
                dumprows(target, baseline)
                dump(target.with_suffix('.manifest.json'), dict(code_sha256=shared['parser']['sha256'],
                                                                stats={'valid_candidate_decisions': 587}))
            elif argv[1] == shared['validator']['path']:
                events.append('format')
                dump(target, {'valid': True})
            elif argv[1] == shared['evaluator']['path']:
                assert 'gold_bytes' in events
                events.append('v2')
                assert argv[argv.index('--body-scope') + 1] == 'all'
                assert argv[argv.index('--no-id-doc') + 1] == 'text'
                dump(target, dict(documents=16, patients=39, metrics={**base, 'Score': .51}, counts={'synthetic': 1}))
            else:
                raise AssertionError(argv)
            m.save(output, {'synthetic_stub_no_actual_command': True})

        m.command = fake_command
        current = root / 'positive'
        current.mkdir()
        dump(current / 'run-plan.json', plan)
        sys.argv = ['evaluate_paired.py', '--directory', str(current)]
        with contextlib.redirect_stdout(io.StringIO()):
            m.main()
        assert events == ['assembly', 'assembly', 'format', 'format', 'gold_bytes', 'v2', 'v2'], events
        assert m.read(current / 'summary.json')['automatic_migration'] is False
        event_trace = list(events)
        mark('synthetic_pair_barrier_exact_two_format_two_v2')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.main()
        except FileExistsError:
            pass
        else:
            raise AssertionError('Repeat invocation accepted')
        assert events == event_trace
        mark('repeat_directory_refused_before_new_commands')
        current = root / 'incomplete-second'
        current.mkdir()
        bad_plan = copy.deepcopy(plan)
        refs = bad_plan['arms']['reverse-first']['references']
        refs['responses'] = dumprows(data / 'incomplete.jsonl', [{'row_id': str(i)} for i in range(79)])
        freeze = m.read(refs['response_freeze']['path'])
        freeze['raw_sha256'] = refs['responses']['sha256']
        refs['response_freeze'] = dump(data / 'badfreeze.json', freeze)
        dump(current / 'run-plan.json', bad_plan)
        sys.argv = ['evaluate_paired.py', '--directory', str(current)]
        events.clear()
        rejects('incomplete_second_blocks_both_assemblies_and_gold', m.main)
        assert not events
        m.sha = original_sha
    sys.argv = original_argv
    m.SHARED_SHA, m.GOLD_SHA = original_shared, original_gold
    assert len(CASES) == 32
    result = dict(status='CPU_synthetic_32_contracts_pass', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256=original_sha(ENTRY), source_bytes=ENTRY.stat().st_size,
                  harness_sha256=original_sha(__file__), cases=CASES, count=len(CASES), synthetic_event_trace=event_trace,
                  seconds=time.monotonic() - started, actual_quality_calls=0, actual_gold_bytes_read=0,
                  actual_parser_calls=0, actual_model_loads=0, actual_network_calls=0,
                  synthetic_helpers_and_subprocesses_stubbed=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
