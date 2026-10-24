"""One posthoc fixed full-support B union all grounded relations; no model/selection."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HELPER = Path(__file__).resolve().parents[1] / 'quality/metadata_recovery/support.py'
HELPER_SHA = 'a70eca6b5c0fbf6a2c5474ebd4ed3976b020109e1c4b43cf9bd81a716f3072b4'
import hashlib
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location('_grounded_support_union_helpers', HELPER)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
BASE = .7115075800266846


def load(path):
    p = H.read(path); gate = H.gold_guard(p)
    assert p['schema'] == 'grounded-uniform-strong-support-union-v1' and p['authorized']
    assert p['source'] == H.ref(__file__) and p['helper'] == H.ref(HELPER)
    assert p['baseline_score'] == BASE and p['new_evaluator_calls'] == 1 and p['old_evaluator_calls'] == 0
    assert p['score_budget_seconds'] == 30 and p['grace_seconds'] == 1
    assert p['policy'] == 'Literal B union every grounded relation; preserve B prefix and all nonrelation bytes; no filtering'
    for r in p['references'].values(): H.checked(r)
    r = p['references']; pointer = H.read(r['strong_result_pointer']['path'])
    fold = next(v for v in pointer['folds'] if v['fold'] == 0)
    assert fold['prediction'] == r['baseline'] and fold['metrics'] == r['baseline_metrics']
    assert fold['new_metrics'] == H.read(r['baseline_metrics']['path'])['metrics']
    assert fold['new_metrics']['Score'] == BASE
    frozen = H.read(r['grounded_freeze']['path'])
    assert frozen['predictions']['new-only'] == r['grounded_prediction']
    return p, gate

def frozen_grounded_count(r):
    return H.read(r['grounded_freeze']['path'])['relation_count']


def assemble(path):
    p, gate = load(path); r = p['references']; out = Path(p['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    try:
        lines, base, bids = H.rows(r['baseline']['path'])
        _, new, nids = H.rows(r['grounded_prediction']['path'])
        _, inputs, iids = H.rows(r['input']['path'])
        assert bids == nids == iids and len(base) == 16
        assert sum(len(d['entities']) for d in base) == 947
        spans, mask = H.byte_helpers(r['byte_helper']); result = []; added = 0; prior = 0; supplied = 0
        for line, b, n, inp in zip(lines, base, new, inputs):
            pids = [v['patient_id'] for v in inp['patient']]
            assert [a['patient_id'] for a in b['association']] == pids
            assert [a['patient_id'] for a in n['association']] == pids
            values = {}
            for ba, na in zip(b['association'], n['association']):
                old = ba['phenotype']; extra = na['phenotype']
                assert len(old) == len(set(old)) and len(extra) == len(set(extra))
                union = old + [v for v in extra if v not in set(old)]
                assert union[:len(old)] == old and set(extra) <= set(union)
                values[ba['patient_id']] = union
                prior += len(old); supplied += len(extra); added += len(union) - len(old)
            updated = H.replace_phenotypes(line, values, spans, mask)
            assert mask(updated) == mask(line); result.append(updated)
        assert supplied == frozen_grounded_count(r) and sum(len(d['patient']) for d in inputs) == 47
        H.publish(out / 'prediction.jsonl', ''.join(result).encode())
        validator = H.load_module('validate_predictions', r['validator'])
        pred = [json.loads(v) for v in result]
        validation = validator.validate_predictions(inputs, pred, validator.read_hpo(r['hpo']['path']))
        H.save(out / 'validation.json', validation); assert validation['valid']
        assert not gate['enabled']
        H.save(out / 'freeze.json', dict(at_utc=H.now(), plan=H.ref(path), source=H.ref(__file__),
            prediction=H.ref(out / 'prediction.jsonl'), validation=H.ref(out / 'validation.json'),
            baseline=r['baseline'], supplied_grounded=r['grounded_prediction'],
            baseline_relations=prior, complete_grounded_relations=supplied, appended=added,
            final_relations=prior+added, documents=16, patients=47, entities=947,
            all_baseline_prefix_and_nonrelation_bytes_exact=True, all_grounded_relations_used=True,
            gold_read_or_hashed=False, new_evaluator_calls=0, NN_or_normalization_calls=0))
        print(json.dumps(H.ref(out / 'freeze.json')))
    except BaseException:
        H.save(out / 'assembly-failure.json', dict(traceback=traceback.format_exc(), no_retry=True)); raise


def frozen_plan(path):
    p, gate = load(path); out = Path(p['output_directory']); freeze = H.read(out / 'freeze.json')
    assert freeze['plan'] == H.ref(path) and freeze['source'] == H.ref(__file__)
    H.checked(freeze['prediction']); H.checked(freeze['validation'])
    assert freeze['all_baseline_prefix_and_nonrelation_bytes_exact'] and freeze['all_grounded_relations_used']
    return p, gate, out, freeze


def worker(path):
    began = time.monotonic(); p, gate, out, freeze = frozen_plan(path); r = p['references']; ev = out / 'evaluation'
    H.save(ev / 'before-gold.json', dict(at_utc=H.now(), freeze=H.ref(out / 'freeze.json'), gold_not_yet_opened=True))
    validator = H.load_module('validate_predictions', r['validator'])
    evaluator = H.load_module('_grounded_strong_union_v2', r['evaluator'])
    assert evaluator.validate_predictions is validator.validate_predictions
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    gate['enabled'] = True; H.checked(p['gold'])
    gold = validator.load_jsonl(p['gold']['path']); _, pred, ids = H.rows(freeze['prediction']['path'])
    assert {str(d['pmc_id']) for d in gold} == set(ids)
    H.save(ev / 'call-start.json', dict(at_utc=H.now(), prediction=freeze['prediction'], gold=p['gold'],
        freeze=H.ref(out / 'freeze.json'), evaluator=r['evaluator'], unique_v2=True))
    metrics = evaluator.evaluate_report(gold, pred, 'all', 'text')
    metrics.update(prediction_sha256=freeze['prediction']['sha256'], gold_sha256=p['gold']['sha256'])
    H.save(ev / 'metrics.json', metrics)
    old = H.read(r['baseline_metrics']['path'])['metrics']
    delta = {k: metrics['metrics'][k]-old[k] for k in H.METRICS}
    assert delta['menF1'] == delta['docF1'] == 0
    def keys(docs):
        return {(str(d['pmc_id']), pid, value) for d in docs for pid, vals in evaluator.associations(d).items() for value in vals}
    _, baseline, _ = H.rows(r['baseline']['path']); truth = keys(gold); new = keys(pred)-keys(baseline)
    assert keys(baseline) <= keys(pred)
    keep = metrics['metrics']['Score'] > BASE
    H.save(ev / 'result.json', dict(at_utc=H.now(), plan=H.ref(path), freeze=H.ref(out / 'freeze.json'),
        metrics=metrics['metrics'], metric_ref=H.ref(ev / 'metrics.json'), old_metrics=old, delta=delta,
        keep=keep, keep_rule='Score strictly above fixed strong support B; macro disclosed without veto',
        macro_decreased=delta['macF1'] < 0, added_tp=len(new & truth), added_fp=len(new-truth),
        new_evaluator_calls=1, old_evaluator_calls=0, NN_or_normalization_calls=0,
        known_F0_development=True, seconds=time.monotonic()-began,
        next='Retain fixed independent cross-fold candidate; root controls multi-source fusion, no automatic originaldev' if keep else 'Close this fixed uniform strong fusion; no threshold or relation filtering'))


def score(path):
    began = time.monotonic(); p, gate, out, freeze = frozen_plan(path); ev = out / 'evaluation'
    ev.mkdir(exist_ok=False)
    H.save(ev / 'launch.json', dict(at_utc=H.now(), plan=H.ref(path), freeze=H.ref(out / 'freeze.json'),
        budget_seconds=30, grace_seconds=1, no_retry=True, gold_opened=False))
    timeout = False
    with (ev/'stdout.log').open('xb') as stdout, (ev/'stderr.log').open('xb') as stderr:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        child = subprocess.Popen([sys.executable, __file__, 'worker', '--plan', str(path)], env=env,
            stdout=stdout, stderr=stderr, start_new_session=True)
        try: code = child.wait(timeout=max(.001, began+30-time.monotonic()))
        except subprocess.TimeoutExpired:
            timeout = True; os.killpg(child.pid, signal.SIGTERM)
            try: code = child.wait(timeout=max(.001, began+31-time.monotonic()))
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        for stream in [stdout, stderr]: stream.flush(); os.fsync(stream.fileno())
    H.save(ev / 'terminal.json', dict(at_utc=H.now(), returncode=code, timed_out=timeout,
        seconds=time.monotonic()-began, budget_seconds=30, grace_seconds=1, child_pid=child.pid,
        result=H.ref(ev/'result.json') if (ev/'result.json').exists() else None))
    print(json.dumps(H.ref(ev / 'terminal.json')))
    if code or timeout: raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['assemble','score','worker']); ap.add_argument('--plan', type=Path, required=True)
    a = ap.parse_args(); globals()[a.stage](a.plan.resolve())
