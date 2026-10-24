"""Exactly two frozen F1 endpoints, unique v2 calls, cached strong-N metrics, 30+TERM1."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from support import (ARMS, BASE_SCORE, KEEP, METRICS, checked, endpoint_choice, load_module,
                     load_plan, now, read, ref, rows, save, validate_freeze)


def evaluator_keys(evaluator, docs):
    # Gold may legally repeat a phenotype. Use v2's set semantics, never prediction duplicate guards.
    return {(str(d['pmc_id']), pid, value) for d in docs
            for pid, values in evaluator.associations(d).items() for value in values}


def key_diagnostics(keys, truth):
    values = {(did, pid, value) for did, pid, kind, value in keys}
    return dict(relations=len(values), tp=len(values & truth), fp=len(values - truth))


def worker(plan_path):
    started = time.monotonic(); plan, guard = load_plan(plan_path)
    assert plan['quality_execution_authorized'] is True
    assert plan['sources']['score.py'] == ref(__file__)
    freeze_path, frozen = validate_freeze(plan_path, plan)
    output = Path(plan['output_directory']) / 'evaluation'
    launch = read(output / 'launch.json')
    assert launch['plan'] == ref(plan_path) and launch['freeze'] == ref(freeze_path) and launch['sources'] == plan['sources']
    save(output / 'before-gold.json', dict(at_utc=now(), common_freeze=ref(freeze_path),
        both_complete_predictions_checked=True, gold_not_yet_read=True, endpoints=ARMS, old_evaluator_calls=0))
    references = plan['references']
    validator = load_module('validate_predictions', references['validator'])
    evaluator = load_module('_grounded_frozen_v2', references['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    hpo = validator.read_hpo(references['hpo']['path'])
    _, inputs, input_ids = rows(references['F1_input']['path'])
    _, baseline, _ = rows(references['baseline']['path'])
    baseline_metrics = read(references['baseline_metrics']['path'])['metrics']
    contributions = read(frozen['contributions']['path'])
    guard['enabled'] = True
    checked(plan['gold']); gold = validator.load_jsonl(plan['gold']['path'])
    assert {str(d['pmc_id']) for d in gold} == set(input_ids)
    truth = evaluator_keys(evaluator, gold); prior = evaluator_keys(evaluator, baseline)
    results = {}; calls = 0
    for arm in ARMS:
        prediction_ref = frozen['predictions'][arm]
        _, prediction, ids = rows(prediction_ref['path']); assert ids == input_ids
        metric_path = output / (arm + '.metrics.json'); attempt_path = output / (arm + '.call-start.json')
        if metric_path.exists():
            metrics = read(metric_path); attempt = read(attempt_path)
            assert attempt['prediction'] == prediction_ref and attempt['freeze'] == ref(freeze_path)
            assert metrics['prediction_sha256'] == prediction_ref['sha256'] and metrics['gold_sha256'] == plan['gold']['sha256']
            assert metrics['evaluator_version'] == 'patientphex-public-approx-v2'
            assert metrics['configuration']['body_scope'] == 'all' and metrics['configuration']['unlinked_document_mode'] == 'text'
            reused = True
        else:
            assert not attempt_path.exists(), 'An attempted endpoint without a metric must not be evaluated again'
            validation = validator.validate_predictions(inputs, prediction, hpo)
            save(output / (arm + '.validation.json'), validation); assert validation['valid']
            save(attempt_path, dict(at_utc=now(), arm=arm, prediction=prediction_ref, gold=plan['gold'],
                evaluator=references['evaluator'], freeze=ref(freeze_path), plan=ref(plan_path), unique_new_v2_call=True))
            metrics = evaluator.evaluate_report(gold, prediction, 'all', 'text'); calls += 1
            metrics.update(gold_sha256=plan['gold']['sha256'], prediction_sha256=prediction_ref['sha256'])
            save(metric_path, metrics)  # Preserve the actual metric before any diagnostic assertion.
            reused = False
        delta = {key: metrics['metrics'][key] - baseline_metrics[key] for key in METRICS}
        current = evaluator_keys(evaluator, prediction); added = current - prior; removed = prior - current
        diagnostic = dict(metrics=metrics['metrics'], metric_ref=ref(metric_path), existing_metric_reused=reused,
            strong_N_metrics=baseline_metrics, delta_vs_strong_N=delta,
            added_tp=len(added & truth), added_fp=len(added - truth), removed_tp=len(removed & truth), removed_fp=len(removed - truth),
            macro_decreased_vs_strong_N=delta['macF1'] < 0, Task1_unchanged=(delta['menF1'] == delta['docF1'] == 0),
            baseline_metrics_reused_not_rescored=True)
        save(output / (arm + '.diagnostics.json'), diagnostic)
        assert diagnostic['Task1_unchanged']; results[arm] = diagnostic
    winner, keep = endpoint_choice({arm: results[arm]['metrics'] for arm in ARMS})
    primary_pass = results['primary']['metrics']['Score'] > BASE_SCORE
    save(output / 'result.json', dict(at_utc=now(), endpoints=results, keep=keep, keep_rule=KEEP,
        selected_endpoint=winner if keep else None, highest_endpoint=winner, tie_priority='primary',
        primary_union_hypothesis_passed=primary_pass, diagnostic_can_win_by_prespecified_rule=True,
        macro_has_no_additional_veto=True, candidate_ceiling_contribution={
            name: key_diagnostics(keys, truth) for name, keys in contributions['keys'].items()},
        contribution_source=frozen['contributions'], candidate_escape_and_correctness_reported_separately=True,
        status='fixed_endpoint_improvement_pending_review' if keep else 'both_fixed_endpoints_negative_closed_no_scan',
        common_freeze=ref(freeze_path), plan=ref(plan_path), sources=plan['sources'],
        new_evaluator_calls_this_worker=calls, completed_endpoint_metrics=2, old_evaluator_calls=0,
        NN_or_fit=False, seconds=time.monotonic() - started,
        scope='Known F1 development comparison only; no automatic F0/originaldev migration, new arm, or official action'))
    print(json.dumps(ref(output / 'result.json')))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); args.plan = args.plan.resolve()
    if args.worker:
        try: worker(args.plan)
        except BaseException:
            output = Path(read(args.plan)['output_directory']) / 'evaluation'
            if output.exists(): save(output / 'failure.json', dict(at_utc=now(), traceback=traceback.format_exc(), automatic_retry=False))
            raise
        return
    started = time.monotonic(); plan, guard = load_plan(args.plan)
    assert plan['quality_execution_authorized'] is True, 'Source-only template does not authorize scoring'
    assert plan['sources']['score.py'] == ref(__file__)
    freeze_path, frozen = validate_freeze(args.plan, plan)
    assert not guard['enabled']; output = Path(plan['output_directory']) / 'evaluation'
    if output.exists():
        assert (output / 'result.json').is_file() and (output / 'terminal.json').is_file(), 'Preserve incomplete attempt; no automatic retry'
        result = read(output / 'result.json'); terminal = read(output / 'terminal.json')
        assert result['plan'] == ref(args.plan) and result['common_freeze'] == ref(freeze_path)
        assert terminal['returncode'] == 0 and terminal['timed_out'] is False
        for arm in ARMS: checked(result['endpoints'][arm]['metric_ref'])
        print(json.dumps(dict(existing_result=ref(output / 'result.json'), new_evaluator_calls=0))); return
    output.mkdir()
    save(output / 'launch.json', dict(at_utc=now(), plan=ref(args.plan), sources=plan['sources'], freeze=ref(freeze_path),
        budget_seconds=30, term_grace_seconds=1, gold_read_in_supervisor=False, automatic_retry=False))
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--plan', str(args.plan)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    timed_out = False
    with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
        save(output / 'child-started.json', dict(pid=child.pid, at_utc=now()))
        try: code = child.wait(timeout=max(.001, started + 30 - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(child.pid, signal.SIGTERM)
            try: code = child.wait(timeout=max(.001, started + 31 - time.monotonic()))
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        for stream in (stdout, stderr): stream.flush(); os.fsync(stream.fileno())
    save(output / 'terminal.json', dict(at_utc=now(), returncode=code, timed_out=timed_out,
        seconds=time.monotonic() - started, child_pid=child.pid, budget_seconds=30, term_grace_seconds=1,
        common_freeze=ref(freeze_path), result=ref(output / 'result.json') if (output / 'result.json').exists() else None))
    print(json.dumps(ref(output / 'terminal.json')))
    if code or timed_out: raise SystemExit(1)


if __name__ == '__main__': main()
