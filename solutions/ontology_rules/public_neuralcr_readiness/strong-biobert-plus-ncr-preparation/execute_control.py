"""One shared 900-second strongB plus NCR fourfold fit/decode/freeze/evaluation window.

No task work runs on import. Source preparation is not execution authorization.
Only children started by this controller in fresh process groups are terminated.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from prepare_pools import (
    FEATURES, OLD_FEATURES, BIOBERT_FEATURES, BASE_MANIFEST_SHA256,
    checked, read, ref, require, rows, same_json, sha, sources)


FORMAL_SECONDS = 900
TERM_SECONDS = 30
HPO_SHA256 = 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
STRONG_FREEZE_SHA256 = 'cf8c1a2820ca94099d6129e9c121eb11c1fd049e4ec35d89de725ef169408f82'
METRICS = ('menF1', 'docF1', 'micF1', 'macF1', 'Score')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    """Exclusive immutable JSON artifact, independent of helper writer changes."""
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def save_state(root, state):
    path = root / 'execution-state.json'
    temporary = root / 'execution-state.json.tmp'
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def nonentities(doc):
    return {key: value for key, value in doc.items() if key != 'entities'}


def objective(report):
    return (report['metrics']['menF1'] + report['metrics']['docF1']) / 2


def deltas(report, baseline):
    return {name: report['metrics'][name] - baseline['metrics'][name] for name in METRICS}


def require_time(deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'The single formal 900-second deadline has expired')
    return remaining


def checked_sources(plan):
    sources(plan)
    require(str(Path(__file__).resolve()) in plan['sources'], 'Controller source is not bound to plan')
    require(str(Path(__file__).with_name('fit_control.py').resolve()) in plan['sources'],
            'Fresh NCR fit source is not bound to plan')


def validate_plan(plan):
    require(plan['execution_ready'] is True,
            'Source-only preparation: root must authorize actual fitting before execution_ready=true')
    require(plan['formal_CPU_cap_seconds'] == FORMAL_SECONDS and
            plan['TERM_grace_seconds'] == TERM_SECONDS, 'Expected one 900-second window plus TERM30')
    require(plan['cut'] == .3 and [fold['fold'] for fold in plan['folds']] == [0, 1, 2, 3],
            'Fixed cut/original fourfold policy changed')
    require(plan['hpo']['sha256'] == HPO_SHA256, 'Current official HPO changed')
    require(plan['base_prepared_manifest']['sha256'] == BASE_MANIFEST_SHA256,
            'Primary input pools must be the frozen strong BioBERT base')
    checked_sources(plan)


def verify_baselines(plan):
    """Metadata/file bindings only; no new evaluation or quality-label read."""
    require(plan['baseline_freeze']['sha256'] == STRONG_FREEZE_SHA256,
            'Expected the already completed strong BioBERT fourfold freeze')
    frozen = read(checked(plan['baseline_freeze']))
    require(frozen['all_four_predictions_complete'] is True and
            [fold['fold'] for fold in frozen['folds']] == [0, 1, 2, 3],
            'Strong BioBERT baseline freeze is incomplete')
    for fold, old in zip(plan['folds'], frozen['folds']):
        require(fold['baseline_prediction'] == old['new_prediction'],
                'Primary baseline must be the completed BioBERT new_prediction, not pre-BioBERT')
        strong_directory = Path(old['new_prediction']['path']).resolve().parent
        weak_directory = Path(old['baseline_prediction']['path']).resolve().parent
        require(Path(fold['baseline_metrics']['path']).resolve() == strong_directory / 'biobert.metrics.json' and
                Path(fold['baseline_validation']['path']).resolve() == strong_directory / 'biobert.validation.json' and
                Path(fold['preB_metrics']['path']).resolve() == weak_directory / 'nested.metrics.json',
                'Strong/weak metric roles do not match the actual frozen prediction sources')
        checked(fold['baseline_prediction'])
        checked(old['baseline_prediction'])
        for name in ('baseline_metrics', 'baseline_validation', 'preB_metrics'):
            checked(fold[name])
    checked(plan['baseline_pooled_metrics'])
    checked(plan['preB_pooled_metrics'])
    require(Path(plan['baseline_pooled_metrics']['path']).resolve() ==
            Path(plan['baseline_freeze']['path']).resolve().parent / 'biobert.pooled.metrics.json' and
            Path(plan['preB_pooled_metrics']['path']).resolve() ==
            Path(frozen['folds'][0]['baseline_prediction']['path']).resolve().parent.parent / 'nested.pooled.metrics.json',
            'Strong/weak pooled metric role mismatch')
    return frozen


def group_exists(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_owned(process, job):
    """Terminate only the new session/group recorded for this exact child."""
    group = job['process_group']
    require(group == process.pid and job['started_new_session'] is True,
            'Refusing to signal an unowned process group')
    began = time.monotonic()
    termination = {'process_group': group, 'requested_at_utc': now(), 'TERM_grace_seconds': TERM_SECONDS,
                   'signals': []}
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    for sig in saved_handlers:
        signal.signal(sig, signal.SIG_IGN)
    try:
        if group_exists(group):
            try:
                os.killpg(group, signal.SIGTERM)
                termination['signals'].append('SIGTERM')
            except ProcessLookupError:
                pass
            grace_deadline = began + TERM_SECONDS
            # Reap the direct child as it exits, but also account for its own group descendants.
            while group_exists(group) and time.monotonic() < grace_deadline:
                process.poll()
                time.sleep(min(.1, max(0., grace_deadline - time.monotonic())))
            if group_exists(group):
                try:
                    os.killpg(group, signal.SIGKILL)
                    termination['signals'].append('SIGKILL')
                except ProcessLookupError:
                    pass
        try:
            termination['returncode'] = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            termination['direct_child_reap_timed_out_after_KILL'] = True
    finally:
        for sig, previous in saved_handlers.items():
            signal.signal(sig, previous)
    termination.update(seconds=time.monotonic() - began, finished_at_utc=now(),
                       group_still_observed=group_exists(group))
    job['termination'] = termination


def invoke(command, deadline, root, name, state):
    remaining = require_time(deadline)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    job = {'name': name, 'command': command, 'started_utc': now(),
           'shared_deadline_monotonic': deadline, 'remaining_window_seconds_at_launch': remaining,
           'status': 'launching', 'started_new_session': True}
    state['jobs'].append(job)
    save_state(root, state)
    process = None
    began = time.monotonic()
    try:
        with (root / (name + '.stdout')).open('x') as out, (root / (name + '.stderr')).open('x') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                       cwd=str(Path(__file__).resolve().parent), start_new_session=True)
            job.update(pid=process.pid, process_group=process.pid, status='running')
            require(os.getpgid(process.pid) == process.pid, 'Child did not receive its own process group')
            save_state(root, state)
            try:
                returncode = process.wait(timeout=require_time(deadline))
            except subprocess.TimeoutExpired:
                job['timed_out_at_shared_deadline'] = True
                terminate_owned(process, job)
                raise TimeoutError('Formal shared deadline expired during ' + name)
            require(returncode == 0, 'Child failed: %s returncode=%s' % (name, returncode))
            require(not group_exists(process.pid), 'Child exited but its own process-group descendants remain')
            require_time(deadline)
            job.update(status='complete', returncode=returncode)
    except BaseException as error:
        job.update(status='failed_preserved', error=repr(error))
        if time.monotonic() >= deadline:
            job['timed_out_at_shared_deadline'] = True
        if process is not None:
            job['returncode'] = process.poll()
            if 'termination' not in job:
                terminate_owned(process, job)
        raise
    finally:
        job.update(finished_utc=now(), seconds=time.monotonic() - began)
        save_state(root, state)


def finalize(plan_path):
    started = time.monotonic()
    plan = read(plan_path)
    validate_plan(plan)
    plan_sha = sha(plan_path)
    root = Path(plan['output_dir'])
    window = read(root / 'window.json')
    require(window['plan_sha256'] == plan_sha and window['cap_seconds'] == FORMAL_SECONDS and
            window['TERM_grace_seconds'] == TERM_SECONDS, 'Formal window binding mismatch')
    deadline = window['deadline_monotonic']
    require_time(deadline)
    fitstage = read(root / 'fit-stage.json')
    require(fitstage['plan_sha256'] == plan_sha and fitstage['all_four_fits_complete'] is True and
            fitstage['window'] == ref(root / 'window.json'), 'Missing complete fourfold fit stage')
    for item in fitstage['manifests']:
        checked(item)
    require(not (root / 'complete-freeze.json').exists() and not (root / 'evaluation-start.json').exists(),
            'Refusing to rerun an existing freeze/evaluation endpoint')
    prepared = read(checked(plan['prepared_manifest']))
    require(list(prepared['new_features']) == list(FEATURES), 'NCR pool feature list changed')
    strong_freeze = verify_baselines(plan)
    require('evaluate' not in sys.modules and 'validate_predictions' not in sys.modules,
            'Evaluator was imported before new predictions were frozen')
    sys.path.insert(0, plan['nested_source'])
    decoder = importlib.import_module('decode_nested_selector')
    decoder_path = str(Path(decoder.__file__).resolve())
    require(sha(decoder_path) == plan['sources'][decoder_path] == strong_freeze['proper_nesting_source_sha256'],
            'Original fixed nesting decoder source changed')
    _, allowed, _, _ = decoder.prior.base.parse_hpo(checked(plan['hpo']))
    allowed.discard('HP:0000118')
    freeze = []
    predictions = {}
    input_by_id = {}
    seen = set()
    diagnostics = []
    for fold in plan['folds']:
        require_time(deadline)
        index = fold['fold']
        pool = prepared['folds'][index]
        target = root / ('fold' + str(index))
        fit = target / 'fitted'
        manifest = read(fit / 'manifest.json')
        require(manifest['status'] == 'fit_and_input_scores_complete_no_decode_or_quality' and
                manifest['plan_sha256'] == plan_sha and manifest['fold'] == index and
                manifest['actual_iterations'] == 160 and manifest['fresh_vectorizer_and_estimator'] is True,
                'Unexpected new fit terminal manifest')
        require(manifest['source_sha256'] == plan['sources'][str(Path(__file__).with_name('fit_control.py').resolve())]
                and list(manifest['new_features']) == list(FEATURES) and
                manifest['actual_F'] == manifest['actual_feature_count'] == len(manifest['features']) and
                set(BIOBERT_FEATURES) <= set(manifest['features']) and
                manifest['base_BioBERT_features_and_candidates_preserved'] is True,
                'Actual fresh fit source/features changed')
        require(manifest['train_probabilities_generated'] is False and
                manifest['old_model_pickle_or_posterior_loaded'] is False, 'Unexpected old-model/train-posterior use')
        require(manifest['prepared_manifest'] == plan['prepared_manifest'] and
                manifest['base_prepared_manifest'] == plan['base_prepared_manifest'] and
                manifest['base_fit_manifest'] == fold['base_fit_manifest'] == pool['base_fit_manifest'] and
                list(manifest['preserved_source_features']) == list(OLD_FEATURES),
                'Strong base/new-pool fit lineage changed')
        strong = rows(checked(fold['baseline_prediction']))
        weak_ref = strong_freeze['folds'][index]['baseline_prediction']
        weak = rows(checked(weak_ref))
        scores = rows(checked(manifest['outputs']['selector-scores.jsonl']))
        checked(manifest['outputs']['model-vectorizer.pkl'])
        tree_state = read(checked(manifest['tree_state']))
        require(manifest['tree_state'] == manifest['outputs']['tree-state.json'] and
                tree_state['actual_F'] == manifest['actual_F'] and tree_state['totals'] == manifest['tree_state_totals'],
                'Actual tree-state inventory binding changed')
        inputs = rows(checked(pool['input']))
        ids = [doc['pmc_id'] for doc in strong]
        require(len(ids) == len(set(ids)) == 16 and ids == [doc['pmc_id'] for doc in weak] ==
                [doc['pmc_id'] for doc in scores] == [doc['pmc_id'] for doc in inputs] == pool['input_ids'],
                'Strong/weak/new document identity or order mismatch')
        require(not seen & set(ids), 'Held documents overlap across folds')
        seen.update(ids)
        new = []
        for old, pre_b, scored, row in zip(strong, weak, scores, inputs):
            require(set(row['input']) == {'pmc_id', 'pmid', 'patient', 'full_text'}, 'Held pool is not input-only')
            require(same_json(nonentities(old), nonentities(pre_b)), 'Strong and preB nonentity objects differ')
            stripped = [{key: value for key, value in candidate.items() if key != 'selector_probability'}
                        for candidate in scored['candidates']]
            require(same_json(stripped, row['candidates']), 'New held posterior does not bind to the new strongB plus NCR pool')
            # Neither strong nor weak old entities enter either decoder call.
            greedy = decoder.greedy(scored['candidates'], .3)
            selected, diag = decoder.decode(scored['candidates'], greedy, row['input'], allowed, cut=.3)
            updated = copy.deepcopy(old)
            updated['entities'] = selected
            require(same_json(nonentities(updated), nonentities(old)), 'New prediction changed nonentity objects')
            new.append(updated)
            input_by_id[old['pmc_id']] = row['input']
            diagnostics.append(dict(fold=index, pmc_id=old['pmc_id'], **diag))
        path = target / 'ncr.nested.jsonl'
        with path.open('x', encoding='utf-8') as handle:
            for doc in new:
                handle.write(json.dumps(doc, ensure_ascii=False, allow_nan=False) + '\n')
        freeze.append({'fold': index, 'baseline_prediction': fold['baseline_prediction'],
                       'preB_prediction': weak_ref, 'new_prediction': ref(path),
                       'new_manifest': ref(fit / 'manifest.json'),
                       'scores': manifest['outputs']['selector-scores.jsonl'],
                       'model': manifest['outputs']['model-vectorizer.pkl'],
                       'tree_state': manifest['tree_state'], 'actual_F': manifest['actual_F'],
                       'runtime': manifest['runtime']})
        predictions[index] = (strong, new)
    require(len(seen) == 64, 'Incomplete held64 output coverage')
    checked_sources(plan)
    require(sha(plan_path) == plan_sha, 'Plan changed during input-only decode')
    write(root / 'input-only-diagnostics.json', diagnostics)
    require_time(deadline)
    require('evaluate' not in sys.modules and 'validate_predictions' not in sys.modules,
            'Decoder imported evaluator before complete freeze')
    frozen = {'frozen_at_utc': now(), 'plan_sha256': plan_sha, 'source_sha256': sha(__file__),
              'window': ref(root / 'window.json'), 'folds': freeze,
              'all_four_predictions_complete': True, 'new_quality_evaluation_started': False,
              'all_nonentity_objects_unchanged': True, 'strong_and_preB_nonentities_equal': True,
              'all_entities_from_new_strongB_plus_NCR_held_scores': True, 'old_entity_predictions_used_for_selection': False,
              'proper_nesting_source_sha256': sha(decoder.__file__),
              'seconds_before_quality_labels': time.monotonic() - started}
    write(root / 'complete-freeze.json', frozen)
    # All four full output byte hashes are checked before importing v2 or reading train64 evaluation gold.
    for item in freeze:
        checked(item['new_prediction'])
        checked(item['baseline_prediction'])
        checked(item['preB_prediction'])
    require_time(deadline)
    sys.path.insert(0, plan['evaluation_source'])
    evaluate = importlib.import_module('evaluate')
    validator = importlib.import_module('validate_predictions')
    require(evaluate.VERSION == 'patientphex-public-approx-v2', 'Evaluator version changed')
    for module in (evaluate, validator):
        module_path = str(Path(module.__file__).resolve())
        require(sha(module_path) == plan['sources'][module_path], 'Evaluator/validator source changed')
    write(root / 'evaluation-start.json', {'started_utc': now(), 'complete_freeze': ref(root / 'complete-freeze.json'),
          'window': ref(root / 'window.json'), 'plan_sha256': plan_sha, 'source_sha256': sha(__file__),
          'policy': 'One fixed endpoint: each fold and pooled64, all/text v2; no scan or baseline reevaluation.'})
    gold = rows(checked(plan['train64_gold']))
    truth = {doc['pmc_id']: doc for doc in gold}
    require(len(gold) == len(truth) == 64 and set(truth) == seen, 'Evaluation gold64 identity mismatch')
    for did, doc in truth.items():
        require({key: value for key, value in doc.items() if key not in ('entities', 'association')} == input_by_id[did],
                'Frozen prediction input differs from evaluation document')
    hpo = validator.read_hpo(checked(plan['hpo']))
    require(hpo['version'] == 'hp/releases/2026-06-23', 'Validator HPO version changed')
    results = []
    by_id = {}
    for fold in plan['folds']:
        require_time(deadline)
        index = fold['fold']
        _, after = predictions[index]
        gold_fold = [truth[doc['pmc_id']] for doc in after]
        strong_metrics = read(checked(fold['baseline_metrics']))
        strong_valid = read(checked(fold['baseline_validation']))
        weak_metrics = read(checked(fold['preB_metrics']))
        report = evaluate.evaluate_report(gold_fold, after, 'all', 'text')
        valid = validator.validate_predictions(gold_fold, after, hpo)
        require(valid['valid'] and valid['error_count'] == valid['warning_count'] == 0,
                'New fold output failed frozen validator')
        require(strong_valid['valid'] and strong_valid['error_count'] == 0, 'Invalid frozen strong baseline')
        for baseline in (strong_metrics, weak_metrics):
            require(report['counts']['relation'] == baseline['counts']['relation'] and
                    report['patients'] == baseline['patients'] and
                    all(report['metrics'][name] == baseline['metrics'][name] for name in ('micF1', 'macF1')),
                    'Unchanged relations do not match frozen strong/weak metrics')
        target = root / ('fold' + str(index))
        write(target / 'ncr.metrics.json', report)
        write(target / 'ncr.validation.json', valid)
        results.append({'fold': index, 'baseline_metrics': fold['baseline_metrics'], 'preB_metrics': fold['preB_metrics'],
                        'baseline_biobert': strong_metrics['metrics'], 'preB': weak_metrics['metrics'],
                        'ncr': report['metrics'], 'deltas_vs_biobert': deltas(report, strong_metrics),
                        'deltas_vs_preB': deltas(report, weak_metrics),
                        'baseline_objective': objective(strong_metrics), 'preB_objective': objective(weak_metrics),
                        'ncr_objective': objective(report),
                        'objective_delta_vs_biobert': objective(report) - objective(strong_metrics),
                        'objective_delta_vs_preB': objective(report) - objective(weak_metrics),
                        'metrics': ref(target / 'ncr.metrics.json'), 'validation': ref(target / 'ncr.validation.json')})
        by_id.update({doc['pmc_id']: doc for doc in after})
    require_time(deadline)
    ordered = [by_id[doc['pmc_id']] for doc in gold]
    pooled = evaluate.evaluate_report(gold, ordered, 'all', 'text')
    validation = validator.validate_predictions(gold, ordered, hpo)
    strong_pooled = read(checked(plan['baseline_pooled_metrics']))
    weak_pooled = read(checked(plan['preB_pooled_metrics']))
    require(validation['valid'] and validation['error_count'] == validation['warning_count'] == 0,
            'Pooled64 output failed frozen validator')
    for baseline in (strong_pooled, weak_pooled):
        require(pooled['counts']['relation'] == baseline['counts']['relation'] and
                pooled['patients'] == baseline['patients'] and
                all(pooled['metrics'][name] == baseline['metrics'][name] for name in ('micF1', 'macF1')),
                'Pooled unchanged relation metrics differ from frozen baseline')
    write(root / 'ncr.pooled.metrics.json', pooled)
    write(root / 'ncr.pooled.validation.json', validation)
    delta = objective(pooled) - objective(strong_pooled)
    nondecreasing = sum(item['objective_delta_vs_biobert'] >= 0 for item in results)
    checked_sources(plan)
    require(sha(plan_path) == plan_sha, 'Plan changed during unique evaluation')
    for item in freeze:
        checked(item['new_prediction'])
    require_time(deadline)
    result = {
        'status': 'completed_unique_fixed_endpoint', 'finished_utc': now(),
        'plan_sha256': plan_sha, 'source_sha256': sha(__file__), 'complete_freeze': ref(root / 'complete-freeze.json'),
        'window': ref(root / 'window.json'), 'folds': results,
        'primary_comparator': 'Completed BioBERT fourfold fixed .3/nested control',
        'secondary_comparator': 'Pre-BioBERT two-public-model fixed .3/nested; attribution only',
        'pooled_baseline_biobert': strong_pooled['metrics'], 'pooled_preB': weak_pooled['metrics'],
        'pooled_ncr': pooled['metrics'], 'pooled_deltas_vs_biobert': deltas(pooled, strong_pooled),
        'pooled_deltas_vs_preB': deltas(pooled, weak_pooled),
        'pooled_objective_delta_vs_biobert': delta,
        'pooled_objective_delta_vs_preB': objective(pooled) - objective(weak_pooled),
        'nondecreasing_folds_vs_biobert': nondecreasing,
        'prespecified_success': delta > 0 and nondecreasing >= 3,
        'prespecified_success_rule': 'Pooled (menF1+docF1)/2 strictly improves over strong BioBERT and at least3/4fold entity means do not decrease.',
        'relations_unchanged': True, 'strong_and_preB_nonentities_equal': True,
        'baseline_metrics_reused_without_new_evaluation': True,
        'original_dev_read': False, 'official_actions': False, 'automatic_original_dev_transfer': False,
        'next_action': 'Root reviews the frozen endpoint; no automatic dev/cache/fit expansion.',
        'outputs': {name: ref(root / name) for name in ('ncr.pooled.metrics.json', 'ncr.pooled.validation.json')},
        'finalize_seconds': time.monotonic() - started,
        'formal_elapsed_seconds_before_results_write': time.monotonic() - window['started_monotonic']
    }
    write(root / 'results.json', result)
    print(json.dumps({key: result[key] for key in
                      ('status', 'pooled_objective_delta_vs_biobert', 'nondecreasing_folds_vs_biobert', 'prespecified_success')}))


def interrupted(signum, _frame):
    if signum == signal.SIGALRM:
        raise TimeoutError('Controller reached the single formal 900-second deadline')
    raise InterruptedError('Controller received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--finalize', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    if args.finalize:
        return finalize(plan_path)
    preflight_started = time.monotonic()
    plan = read(plan_path)
    validate_plan(plan)
    plan_sha = sha(plan_path)
    prepared = read(checked(plan['prepared_manifest']))
    require(prepared['status'] == 'prepared_input_only_no_fit_or_quality' and
            list(prepared['new_features']) == list(FEATURES), 'NCR pools must be complete before the formal window')
    require(prepared['base_BioBERT_prefix_preserved'] is True and
            prepared['no_old_selector_posteriors'] is True, 'Missing strong BioBERT prefix or old posterior found')
    require(prepared['base_prepared_manifest'] == plan['base_prepared_manifest'] and
            list(prepared['preserved_source_features']) == list(OLD_FEATURES) and
            [fold['fold'] for fold in prepared['folds']] == [0, 1, 2, 3],
            'Prepared strong-base lineage or retained feature schema changed')
    for fold in prepared['folds']:
        for role in ('train', 'input'):
            checked(fold[role])
    checked(prepared['cache_manifest'])
    checked(prepared['canonical'])
    checked(plan['hpo'])
    verify_baselines(plan)
    # Do not verify/read train64 evaluation gold here; that belongs after the all-four freeze.
    require(sha(plan_path) == plan_sha, 'Plan changed during preflight')
    root = Path(plan['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    deadline = start + FORMAL_SECONDS
    window = {'started_utc': started_utc.isoformat(),
              'deadline_utc': (started_utc + timedelta(seconds=FORMAL_SECONDS)).isoformat(),
              'started_monotonic': start, 'deadline_monotonic': deadline,
              'cap_seconds': FORMAL_SECONDS, 'TERM_grace_seconds': TERM_SECONDS,
              'plan_sha256': plan_sha, 'source_sha256': sha(__file__), 'controller_pid': os.getpid(),
              'scope': 'All4fresh fit/infer + fixed greedy/nested decode + freeze + sole all/text v2 endpoint; no per-fold cap reset.',
              'preflight_seconds_outside_formal_window': start - preflight_started}
    write(root / 'window.json', window)
    state = {'status': 'running', 'started_utc': now(), 'plan_sha256': plan_sha,
             'source_sha256': sha(__file__), 'window': ref(root / 'window.json'), 'jobs': []}
    save_state(root, state)
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    for signum in handlers:
        signal.signal(signum, interrupted)
    try:
        signal.setitimer(signal.ITIMER_REAL, require_time(deadline))
        for fold in range(4):
            invoke([plan['python'], str(Path(__file__).with_name('fit_control.py').resolve()),
                    '--plan', str(plan_path), '--fold', str(fold)],
                   deadline, root, 'fit' + str(fold), state)
        require_time(deadline)
        manifests = [ref(root / ('fold' + str(fold)) / 'fitted/manifest.json') for fold in range(4)]
        for item in manifests:
            fitted = read(checked(item))
            require(fitted['status'] == 'fit_and_input_scores_complete_no_decode_or_quality' and
                    fitted['plan_sha256'] == plan_sha, 'A formal fold did not complete')
        write(root / 'fit-stage.json', {'plan_sha256': plan_sha, 'all_four_fits_complete': True,
              'seconds_from_formal_start': time.monotonic() - start,
              'window': ref(root / 'window.json'), 'manifests': manifests})
        invoke([plan['python'], str(Path(__file__).resolve()), '--plan', str(plan_path), '--finalize'],
               deadline, root, 'decode-evaluate', state)
        result = read(root / 'results.json')
        require(result['status'] == 'completed_unique_fixed_endpoint' and result['plan_sha256'] == plan_sha,
                'Missing complete unique endpoint')
        require_time(deadline)
        state.update(status='complete', finished_utc=now(), results=ref(root / 'results.json'),
                     formal_elapsed_seconds=time.monotonic() - start)
    except BaseException as error:
        state.update(status='failed_preserved_no_rerun', failure=repr(error), finished_utc=now(),
                     formal_elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        save_state(root, state)
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


if __name__ == '__main__':
    main()
