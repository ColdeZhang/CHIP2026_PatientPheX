"""Resume only the uncalled F0 evaluation; reuse immutable F1 metrics."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ORIGINAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ORIGINAL))
from common import (load_plan, check_acceptance, ref, checked, read, save, now,
                    load_module, rows, evaluation_keys, METRICS, COMPOUND)
from score import validate_freeze


def truth_keys(evaluator, gold):
    """Use the frozen evaluator's exact per-patient relation-set semantics."""
    return {(str(doc['pmc_id']), pid, value)
            for doc in gold for pid, values in evaluator.associations(doc).items()
            for value in values}


def preflight(args):
    recovery = read(args.plan)
    assert recovery['schema'] == 'majority-gold-diagnostic-recovery-v1'
    assert recovery['source'] == ref(__file__)
    assert recovery['budget_seconds'] == 29 and recovery['term_grace_seconds'] == 1
    assert recovery['new_evaluator_calls'] == {'F1': 0, 'F0': 1}
    assert recovery['original_sources_unchanged'] and recovery['no_prediction_changes']
    original_path = checked(recovery['original_plan'])
    plan, guard = load_plan(original_path)
    assert not guard['enabled']
    accepted = check_acceptance(recovery['postaccept_receipt']['path'], recovery['postaccept_receipt']['sha256'])
    assert accepted == recovery['postaccept_receipt']
    freeze_path, frozen = validate_freeze(original_path, plan)
    assert ref(freeze_path) == recovery['common_freeze']
    assert frozen['predictions'] == recovery['frozen_predictions']
    old = Path(plan['output_directory']) / 'evaluation'
    assert {p.name for p in old.iterdir()} == set(recovery['original_evaluation_files'])
    for name, value in recovery['original_evaluation_files'].items():
        assert Path(value['path']) == old / name
        checked(value)
    assert not list(old.glob('F0.*')) and not (old / 'result.json').exists()
    previous = read(old / 'terminal.json')
    assert previous['returncode'] == 1 and not previous['timed_out']
    assert previous['seconds'] == recovery['prior_actual_seconds']
    assert previous['seconds'] + recovery['budget_seconds'] <= 30
    assert previous['postaccept_receipt'] == accepted and previous['freeze'] == ref(freeze_path)
    failure = read(old / 'failure.json')['traceback']
    assert 'evaluation_keys(gold)' in failure and 'AssertionError: Duplicate votes must not count' in failure
    metrics = read(old / 'F1.metrics.json')
    assert metrics['evaluator_version'] == 'patientphex-public-approx-v2'
    assert metrics['configuration']['body_scope'] == 'all' and metrics['configuration']['unlinked_document_mode'] == 'text'
    assert metrics['prediction_sha256'] == frozen['predictions']['F1']['sha256']
    assert metrics['gold_sha256'] == plan['folds']['F1']['gold']['sha256']
    assert metrics['metrics']['Score'] == recovery['already_observed_F1_Score']
    assert metrics['metrics']['Score'] < plan['baseline_scores']['F1']
    call = read(old / 'F1.call-start.json')
    assert call['call_number'] == 1 and call['prediction'] == frozen['predictions']['F1']
    assert call['evaluator'] == plan['references']['evaluator'] and call['gold'] == plan['folds']['F1']['gold']
    assert read(old / 'F1.validation.json')['valid']
    push = read(args.source_push_receipt)
    assert push['schema'] == 'majority-recovery-source-push-receipt-v1'
    assert push['plan'] == ref(args.plan) and push['source'] == ref(__file__)
    assert push['pushed'] is True and push['ls_remote_exact'] is True
    assert len(push['commit']) == 40 and all(c in '0123456789abcdef' for c in push['commit'])
    return recovery, plan, guard, accepted, frozen


def worker(args):
    started = time.monotonic()
    recovery, plan, guard, accepted, frozen = preflight(args)
    output = Path(recovery['output_directory'])
    launch = read(output / 'launch.json')
    assert launch['recovery_plan'] == ref(args.plan) and launch['source_push_receipt'] == ref(args.source_push_receipt)
    assert launch['postaccept_receipt'] == accepted
    # No evaluator module or label is loaded until all original and recovery gates pass.
    validator = load_module('validate_predictions', plan['references']['validator'])
    evaluator = load_module('majority_recovery_frozen_evaluator', plan['references']['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION == 'patientphex-public-approx-v2' and evaluator.COMPOUND_HPO.pattern == COMPOUND.pattern
    hpo = validator.read_hpo(plan['references']['hpo']['path'])
    assert hpo['version'] == 'hp/releases/2026-06-23'
    results = {}
    for fold in plan['fold_order']:
        spec = plan['folds'][fold]
        refs = spec['references']
        _, prediction, _ = rows(frozen['predictions'][fold]['path'])
        if fold == 'F1':
            metric_ref = recovery['original_evaluation_files']['F1.metrics.json']
            metrics = read(checked(metric_ref))
            validation_ref = recovery['original_evaluation_files']['F1.validation.json']
            save(output / 'F1.reuse.json', dict(at_utc=now(), metrics=metric_ref,
                 validation=validation_ref, evaluate_report_called=False, original_call_count=1))
        else:
            assert fold == 'F0'
            _, inputs, _ = rows(refs['input']['path'])
            validation = validator.validate_predictions(inputs, prediction, hpo)
            save(output / 'F0.validation.json', validation)
            assert validation['valid']
            validation_ref = ref(output / 'F0.validation.json')
        guard['enabled'] = True
        checked(spec['gold'])
        gold = validator.load_jsonl(spec['gold']['path'])
        if fold == 'F0':
            save(output / 'F0.call-start.json', dict(at_utc=now(), prediction=frozen['predictions']['F0'],
                 gold=spec['gold'], evaluator=plan['references']['evaluator'],
                 recovery_call_number=1, cumulative_original_control_call_number=2))
            metrics = evaluator.evaluate_report(gold, prediction, 'all', 'text')
            metrics['gold_sha256'] = spec['gold']['sha256']
            metrics['prediction_sha256'] = frozen['predictions'][fold]['sha256']
            save(output / 'F0.metrics.json', metrics)
            metric_ref = ref(output / 'F0.metrics.json')
        baseline = read(refs['baseline_metrics']['path'])['metrics']
        rules = read(refs['rules_metrics']['path'])['metrics']
        delta = {key: metrics['metrics'][key] - baseline[key] for key in METRICS}
        assert delta['menF1'] == delta['docF1'] == 0
        _, old, _ = rows(refs['baseline']['path'])
        # Prediction duplicate-vote checks remain exactly the original frozen helpers.
        prior, current = evaluation_keys(old), evaluation_keys(prediction)
        truth = truth_keys(evaluator, gold)
        assert current <= prior
        removed = prior - current
        results[fold] = dict(metrics=metrics['metrics'], metrics_reference=metric_ref,
            validation_reference=validation_ref, triple_union_baseline=baseline, rules_reference=rules,
            delta_vs_triple_union=delta, delta_vs_rules={k: metrics['metrics'][k] - rules[k] for k in METRICS},
            removed_tp=len(removed & truth), removed_fp=len(removed - truth), added_relations=0,
            fold_keep=delta['Score'] > 0, macro_decreased=delta['macF1'] < 0,
            old_metrics_reused=True, F1_evaluation_reused=(fold == 'F1'))
    keep = all(results[fold]['fold_keep'] for fold in plan['fold_order'])
    assert keep is False and results['F1']['fold_keep'] is False
    # Recheck immutable first-attempt evidence after the only remaining call.
    for value in recovery['original_evaluation_files'].values():
        checked(value)
    save(output / 'result.json', dict(at_utc=now(), folds=results, keep=keep, keep_rule=plan['keep_rule'],
        status='negative_closed_no_scan', common_freeze=recovery['common_freeze'],
        postaccept_receipt=accepted, original_plan=recovery['original_plan'], recovery_plan=ref(args.plan),
        original_sources=plan['sources'], recovery_source=ref(__file__),
        new_evaluator_calls=1, cumulative_control_evaluator_calls=2, new_evaluator_calls_by_fold={'F1': 0, 'F0': 1},
        original_F1_metric_unchanged=True, old_baseline_evaluator_calls=0, original_dev_evaluator_calls=0,
        prior_actual_seconds=recovery['prior_actual_seconds'], recovery_worker_seconds=time.monotonic() - started,
        prior_failure_preserved=recovery['original_evaluation_files']['failure.json'],
        gold_access_disclosure='Original attempt read F1 gold and evaluated F1 once; recovery reads F1 only for fixed diagnostics and F0 for its first evaluate_report call.',
        diagnostic_truth_semantics='Exact pinned evaluator.associations per document; duplicate gold phenotypes deduplicate as sets.',
        NN_or_fit=False, scope=plan['scope'], patient_F0_disclosure=plan['patient_F0_disclosure']))
    print(json.dumps(ref(output / 'result.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--source-push-receipt', type=Path, required=True)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        try:
            worker(args)
        except BaseException:
            recovery = read(args.plan)
            output = Path(recovery['output_directory'])
            if output.exists():
                save(output / 'failure.json', dict(at_utc=now(), traceback=traceback.format_exc()))
            raise
        return
    recovery, plan, guard, accepted, frozen = preflight(args)
    assert not guard['enabled']
    output = Path(recovery['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'launch.json', dict(at_utc=now(), recovery_plan=ref(args.plan), source=ref(__file__),
         source_push_receipt=ref(args.source_push_receipt), original_plan=recovery['original_plan'],
         original_evaluation_files=recovery['original_evaluation_files'], postaccept_receipt=accepted,
         common_freeze=recovery['common_freeze'], budget_seconds=29, term_grace_seconds=1,
         prior_actual_seconds=recovery['prior_actual_seconds'],
         new_separate_bounded_invocation=True, no_claim_of_original_wall_deadline_continuation=True,
         new_evaluator_calls_by_fold={'F1': 0, 'F0': 1}, gold_read_in_driver=False, automatic_retry=False))
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--plan', str(args.plan.resolve()),
               '--source-push-receipt', str(args.source_push_receipt.resolve())]
    started = time.monotonic()
    timed_out = False
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            rc = process.wait(timeout=29)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                rc = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait()
    elapsed = time.monotonic() - started
    save(output / 'terminal.json', dict(at_utc=now(), returncode=rc, timed_out=timed_out,
        recovery_seconds=elapsed, prior_actual_seconds=recovery['prior_actual_seconds'],
        accumulated_actual_invocation_seconds=recovery['prior_actual_seconds'] + elapsed,
        child_pid=process.pid, budget_seconds=29, term_grace_seconds=1,
        separately_bounded_recovery_not_original_wall_deadline=True,
        result=ref(output / 'result.json') if (output / 'result.json').exists() else None,
        postaccept_receipt=accepted, common_freeze=recovery['common_freeze']))
    print(json.dumps(ref(output / 'terminal.json')))
    if rc or timed_out:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
