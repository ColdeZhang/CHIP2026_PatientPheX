"""One fixed twofold CPU control using already selected GLiNER additions."""
import argparse
import collections
import copy
import hashlib
import json
import pathlib
import signal
import sys
import time
from datetime import datetime, timezone


def ref(p):
    p = pathlib.Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def read(p):
    return json.loads(pathlib.Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in pathlib.Path(p).read_text().splitlines() if s]


def save(p, x):
    with pathlib.Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_rows(p, data):
    with pathlib.Path(p).open('x') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False)+'\n')


def key(e):
    return e['offset'], e['length'], e['identifier']


def overlap(a, b):
    return a['offset'] < b['offset'] + b['length'] and b['offset'] < a['offset'] + a['length']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['assemble', 'evaluate'])
    ap.add_argument('--directory', type=pathlib.Path, required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU cap 30s')))
    signal.alarm(30)
    start = time.monotonic()
    b = a.directory
    plan = read(b/'plan.json')
    assert ref(__file__) == plan['source']
    supply = read(plan['supply_report']['path'])
    assert ref(plan['supply_report']['path']) == plan['supply_report']
    for item in plan['folds']:
        for name in ['baseline', 'old_gliner_base', 'old_gliner_augmented', 'input', 'old_metrics']:
            assert ref(item[name]['path']) == item[name]
    if a.mode == 'assemble':
        out = b/'assembled'
        out.mkdir(exist_ok=False)
        refs = []
        logs = []
        for item, audit in zip(plan['folds'], supply['folds']):
            assert item['fold'] == audit['fold']
            baseline = rows(item['baseline']['path'])
            old = {d['pmc_id']: d for d in rows(item['old_gliner_base']['path'])}
            augmented = {d['pmc_id']: d for d in rows(item['old_gliner_augmented']['path'])}
            assert {d['pmc_id'] for d in baseline} == set(old) == set(augmented)
            result = copy.deepcopy(baseline)
            added = []
            for before, d in zip(baseline, result):
                pmc = d['pmc_id']
                old_keys = {key(e) for e in old[pmc]['entities']}
                existing = {key(e) for e in d['entities']}
                candidates = [e for e in augmented[pmc]['entities'] if key(e) not in old_keys]
                for e in candidates:
                    if key(e) in existing or any(overlap(e, v) for v in d['entities']):
                        continue
                    d['entities'].append(copy.deepcopy(e))
                    existing.add(key(e))
                    added.append(dict(pmc_id=pmc, entity=e))
                assert d['entities'][:len(before['entities'])] == before['entities']
                assert {k: v for k, v in d.items() if k != 'entities'} == {k: v for k, v in before.items() if k != 'entities'}
            expected = [dict(pmc_id=x['pmc_id'], entity=x['entity']) for x in audit['details'] if x['status'] == 'nonoverlap_new_key']
            normalize = lambda z: sorted(json.dumps(x, sort_keys=True) for x in z)
            assert normalize(added) == normalize(expected)
            assert len(added) == item['expected_additions']
            p = out/f"fold{item['fold']}.jsonl"
            write_rows(p, result)
            refs.append(dict(fold=item['fold'], baseline=item['baseline'], new_prediction=ref(p)))
            logs.append(dict(fold=item['fold'], added=added))
        save(out/'added.json', logs)
        save(out/'freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            plan=ref(b/'plan.json'), predictions=refs, additions=ref(out/'added.json'),
            old_entities_and_nonentities_exact=True, gold_read=False, seconds=time.monotonic()-start))
        print(json.dumps(ref(out/'freeze.json')))
        return
    freeze = read(b/'assembled/freeze.json')
    assert freeze['gold_read'] is False and freeze['plan'] == ref(b/'plan.json')
    for x in freeze['predictions']:
        assert x['new_prediction'] == ref(x['new_prediction']['path'])
    out = b/'evaluation'
    out.mkdir(exist_ok=False)
    save(out/'before-gold.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        freeze=ref(b/'assembled/freeze.json'), gold_read=False))
    sys.path.insert(0, plan['scripts'])
    import evaluate as ev
    import validate_predictions as va
    assert ref(pathlib.Path(plan['scripts'])/'evaluate.py') == plan['evaluator']
    assert ref(pathlib.Path(plan['scripts'])/'validate_predictions.py') == plan['validator']
    assert ev.VERSION == 'patientphex-public-approx-v2'
    hpo = va.read_hpo(plan['hpo']['path'])
    assert ref(plan['hpo']['path']) == plan['hpo']
    pooled_old, pooled_new, pooled_gold = [], [], []
    summaries = []
    for item, fr in zip(plan['folds'], freeze['predictions']):
        new, old = rows(fr['new_prediction']['path']), rows(item['baseline']['path'])
        gold, inputs = rows(item['gold_path']), rows(item['input']['path'])
        assert [d['pmc_id'] for d in new] == [d['pmc_id'] for d in old] == [d['pmc_id'] for d in inputs]
        validation = va.validate_predictions(inputs, new, hpo)
        assert validation['valid']
        score = ev.evaluate_report(gold, new, 'all', 'text')
        previous = read(item['old_metrics']['path'])
        delta = {k: score['metrics'][k]-v for k,v in previous['metrics'].items()}
        assert delta['micF1'] == delta['macF1'] == 0
        save(out/f"fold{item['fold']}.metrics.json", score)
        save(out/f"fold{item['fold']}.validation.json", validation)
        summaries.append(dict(fold=item['fold'], metrics=score['metrics'], delta=delta,
            entity_objective_delta=(delta['menF1']+delta['docF1'])/2,
            old_metrics_reused=True))
        pooled_old.extend(old); pooled_new.extend(new); pooled_gold.extend(gold)
    assert len(pooled_gold) == len({d['pmc_id'] for d in pooled_gold}) == 32
    old_score = ev.evaluate_report(pooled_gold, pooled_old, 'all', 'text')
    new_score = ev.evaluate_report(pooled_gold, pooled_new, 'all', 'text')
    save(out/'pooled.baseline.metrics.json', old_score)
    save(out/'pooled.new.metrics.json', new_score)
    delta = {k: new_score['metrics'][k]-v for k,v in old_score['metrics'].items()}
    keep = (delta['menF1']+delta['docF1'])/2 > 0
    save(out/'results.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        plan=ref(b/'plan.json'), freeze=ref(b/'assembled/freeze.json'), folds=summaries,
        pooled_baseline=old_score['metrics'], pooled_new=new_score['metrics'], delta=delta,
        entity_objective_delta=(delta['menF1']+delta['docF1'])/2, keep=keep,
        fold_directions_agree=all(x['entity_objective_delta']>0 for x in summaries) or all(x['entity_objective_delta']<=0 for x in summaries),
        scope='Two development folds, GLiNER48 for F0 and GLiNER32 for F1; no fourfold robustness claim or originaldev.',
        seconds=time.monotonic()-start))
    print((out/'results.json').read_text())


if __name__ == '__main__':
    main()
