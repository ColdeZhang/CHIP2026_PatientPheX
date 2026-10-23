#!/usr/bin/env python3
"""One fixed .5 veto of scored existing entities; untouched others/nonentities."""
from pathlib import Path
import argparse
import copy
import datetime
import hashlib
import json
import math
import signal
import sys
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(line) for line in Path(p).read_bytes().splitlines()]
def save(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def key(pmc, e): return str(pmc), e['offset'], e['length'], e['identifier']


def veto(doc, scores):
    out = copy.deepcopy(doc)
    out['entities'] = []
    removed = []
    for e in doc['entities']:
        k = key(doc['pmc_id'], e)
        if k not in scores:
            out['entities'].append(copy.deepcopy(e))
            continue
        p = scores[k]
        assert type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
        if p < .5:
            removed.append(dict(key=list(k), probability=p))
        else:
            out['entities'].append(copy.deepcopy(e))
    assert {k: v for k, v in out.items() if k != 'entities'} == {
        k: v for k, v in doc.items() if k != 'entities'}
    return out, removed


def contracts():
    es = [dict(offset=0, length=5, identifier=i, text='alpha', note={'keep': i})
          for i in ['A', 'B', '-1', 'C;D']]
    doc = dict(pmc_id='x', entities=es, association=[dict(patient_id='p', phenotype=['A'])], other=None)
    original = copy.deepcopy(doc)
    scores = {key('x', es[0]): .5, key('x', es[1]): .499999, key('x', es[3]): .9}
    result, removed = veto(doc, scores)
    assert result['entities'] == [es[0], es[2], es[3]] and len(removed) == 1
    assert doc == original
    assert veto(doc, {}) == (doc, [])
    assert veto(dict(doc, entities=[]), scores) == (dict(doc, entities=[]), [])
    for invalid in [float('nan'), float('inf'), -.01, 1.01, True, '0.4']:
        try:
            veto(doc, {key('x', es[0]): invalid})
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid score accepted')
    return dict(point5_inclusive=True, unscored_and_NIL_unchanged=True,
                entire_compound_key_only=True, no_overlap_redecode=True,
                nonentities_and_input_unchanged=True, empty_pool=True,
                invalid_probability_cases_rejected=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['contracts', 'decode', 'evaluate'])
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--gold', type=Path)
    ap.add_argument('--scripts', type=Path)
    ap.add_argument('--hpo', type=Path)
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU stage >30s')))
    signal.alarm(30)
    started = time.monotonic()
    b = a.directory
    if a.mode == 'contracts':
        save(b / 'contracts.json', dict(source=ref(__file__), completed_at_utc=now(), checks=contracts()))
        return
    plan = load(b / 'plan.json')
    assert plan['source_sha256'] == sha(__file__) and plan['cut'] == .5
    for item in plan['references'].values():
        assert sha(item['path']) == item['sha256']
    refs = plan['references']
    if a.mode == 'decode':
        assert a.gold is a.scripts is a.hpo is None
        inputs = rows(refs['input']['path'])
        assert all(not {'entities', 'association'} & d.keys() for d in inputs)
        baseline = rows(refs['baseline']['path'])
        assert [d['pmc_id'] for d in inputs] == [d['pmc_id'] for d in baseline]
        scores = {}
        for row in rows(refs['binary_scores']['path']):
            assert len(row['keys']) == len(row['probabilities'])
            for k, p in zip(row['keys'], row['probabilities']):
                k = (str(row['pmc_id']), *k)
                assert k not in scores and type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                scores[k] = p
        assert len(scores) == 1589
        prediction, removal_log = [], []
        all_keys = [key(d['pmc_id'], e) for d in baseline for e in d['entities']]
        assert len(all_keys) == len(set(all_keys)) == 1295
        covered = sum(k in scores for k in all_keys)
        assert covered == 1127
        for doc in baseline:
            out, removed = veto(doc, scores)
            prediction.append(out)
            removal_log.extend(removed)
        assert len(removal_log) == 152
        assert sum(len(d['entities']) for d in prediction) == 1143
        with (b / 'pred.jsonl').open('x') as f:
            for doc in prediction:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        save(b / 'removed.json', removal_log)
        save(b / 'freeze.json', dict(frozen_at_utc=now(), plan=ref(b / 'plan.json'),
            source=ref(__file__), baseline=refs['baseline'], prediction=ref(b / 'pred.jsonl'),
            removal_log=ref(b / 'removed.json'), original_entities=1295, covered=covered,
            uncovered_retained=168, covered_positive_retained=975, removed=152, final_entities=1143,
            all_nonentities_exact=True, gold_read=False, fit=False, GPU=False,
            seconds=time.monotonic()-started))
        print((b / 'freeze.json').read_text())
    else:
        freeze = load(b / 'freeze.json')
        assert freeze['gold_read'] is False and freeze['plan']['sha256'] == sha(b / 'plan.json')
        assert sha(freeze['prediction']['path']) == freeze['prediction']['sha256']
        out = b / 'evaluation'
        out.mkdir(exist_ok=False)
        save(out / 'before-gold.json', dict(time_utc=now(), freeze=ref(b / 'freeze.json'),
            evaluator=ref(a.scripts / 'evaluate.py'), validator=ref(a.scripts / 'validate_predictions.py'),
            source=ref(__file__), gold_read=False))
        sys.path.insert(0, str(a.scripts))
        import evaluate as evaluator
        import validate_predictions as validator
        assert evaluator.VERSION == 'patientphex-public-approx-v2'
        inputs = validator.load_jsonl(refs['input']['path'])
        gold = validator.load_jsonl(a.gold)
        assert {d['pmc_id'] for d in inputs} == {d['pmc_id'] for d in gold}
        pred = validator.load_jsonl(freeze['prediction']['path'])
        hpo = validator.read_hpo(a.hpo)
        assert hpo['version'] == 'hp/releases/2026-06-23'
        validation = validator.validate_predictions(inputs, pred, hpo)
        save(out / 'validation.json', validation)
        assert validation['valid']
        report = evaluator.evaluate_report(gold, pred, 'all', 'text')
        report.update(gold_sha256=sha(a.gold), prediction_sha256=freeze['prediction']['sha256'])
        save(out / 'metrics.json', report)
        baseline = load(refs['baseline_metrics']['path'])
        delta = {k: report['metrics'][k] - baseline['metrics'][k] for k in baseline['metrics']}
        objective_delta = delta['menF1'] + delta['docF1']
        count_delta = {kind: {k: report['counts'][kind][k] - baseline['counts'][kind][k]
                      for k in ['tp', 'fp', 'fn']} for kind in ['mention', 'document', 'relation']}
        save(out / 'results.json', dict(completed_at_utc=now(), baseline_metrics=baseline['metrics'],
            new_metrics=report['metrics'], delta=delta, count_delta=count_delta,
            entity_objective_sum_delta=objective_delta, keep=objective_delta > 0,
            baseline_reused_not_reevaluated=True, relations_unchanged=delta['micF1'] == delta['macF1'] == 0,
            development_control=True, same_training_resources=False,
            seconds=time.monotonic()-started))
        print((out / 'results.json').read_text())


if __name__ == '__main__':
    main()
