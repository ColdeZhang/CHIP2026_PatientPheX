"""Prospectively fixed original-dev full64 fusion and grounded-union control."""
import argparse
from collections import Counter
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


def ref(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def run(planpath, mode):
    started = time.monotonic()
    p = json.loads(Path(planpath).read_text())
    r, out = p['refs'], Path(p['output'])
    assert p['scope'] == 'original_dev16_development_full64'
    assert r['baseline']['sha256'] == '9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67'
    assert r['source'] == ref(__file__)
    ep = checked(r['evaluator'])
    checked(r['validator'])
    sys.path.insert(0, str(ep.parent))
    ev, va = importlib.import_module('evaluate'), importlib.import_module('validate_predictions')
    def pairs(rows):
        return {(d['pmc_id'], pid, h) for d in rows
                for pid, values in ev.associations(d).items() for h in values}
    if mode == 'assemble':
        forbidden = os.path.realpath(r['gold']['path'])
        def audit(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                if os.path.realpath(os.fsdecode(args[0])) == forbidden:
                    raise PermissionError('Held gold forbidden before joint freeze')
        sys.addaudithook(audit)
        out.mkdir(parents=True, exist_ok=False)
        put(out/'started.json', dict(at_utc=now(), plan=ref(planpath)))
        inputs = va.load_jsonl(checked(r['input']))
        base = va.load_jsonl(checked(r['baseline']))
        old, by = pairs(base), {d['pmc_id']: d for d in base}
        assert len(old) == 416 and len(by) == len(inputs) == 16
        assert by.keys() == {d['pmc_id'] for d in inputs}
        assert sum(len(d['entities']) for d in base) == 1061
        assert sum(len(d['association']) for d in base) == 36
        votes = Counter()
        counts = {}
        source_new = {}
        source_totals = {}
        for key in ['grounded', 'sampled', 'qwen3']:
            rows = va.load_jsonl(checked(r[key]))
            assert {d['pmc_id'] for d in rows} == by.keys()
            for d in rows:
                assert {k:v for k,v in d.items() if k not in ('association', 'entities')} == {
                    k:v for k,v in by[d['pmc_id']].items() if k not in ('association', 'entities')}
                assert ev.associations(d).keys() == ev.associations(by[d['pmc_id']]).keys()
            current = pairs(rows)
            additions = current-old
            counts[key] = len(additions)
            source_new[key] = additions
            source_totals[key] = len(current)
            votes.update(additions)
        assert counts == p['expected_source_additions']
        predictions = {}
        for name in ['two_of_three', 'grounded_union']:
            additions = ({k for k,v in votes.items() if v >= 2}
                         if name == 'two_of_three' else source_new['grounded'])
            pred = copy.deepcopy(base)
            for d in pred:
                for a in d['association']:
                    a['phenotype'].extend(sorted(h for doc,pid,h in additions
                        if doc == d['pmc_id'] and pid == a['patient_id']))
            assert pairs(pred) == old|additions
            valid = va.validate_predictions(inputs, pred, va.read_hpo(checked(r['hpo'])))
            put(out/f'{name}.format.json', valid)
            assert valid['valid']
            path = out/f'{name}.prediction.jsonl'
            with path.open('x') as f:
                for d in pred:
                    f.write(json.dumps(d, ensure_ascii=False, separators=(',', ':'))+'\n')
                f.flush()
                os.fsync(f.fileno())
            predictions[name] = dict(prediction=ref(path), added=len(additions),
                added_pairs=sorted(additions), format=ref(out/f'{name}.format.json'))
        put(out/'freeze.json', dict(at_utc=now(), plan=ref(planpath), predictions=predictions,
            base_pairs=416, source_additions=counts, source_totals=source_totals, vote_histogram=dict(Counter(votes.values())),
            nonassociation_objects_exact=True, removed=0, no_gold_read=True,
            NN_calls=0, fit_calls=0, seconds=time.monotonic()-started))
    else:
        frozen = json.loads((out/'freeze.json').read_text())
        assert frozen['plan'] == ref(planpath)
        for row in frozen['predictions'].values():
            checked(row['prediction'])
        put(out/'evaluation-intent.json', dict(at_utc=now(), freeze=ref(out/'freeze.json')))
        gold = va.load_jsonl(checked(r['gold']))
        old = json.loads(checked(r['baseline_metrics']).read_text())['metrics']
        assert old['Score'] == .7543613981990362
        comparison = json.loads(checked(r['comparison_metrics']).read_text())['metrics']
        assert comparison['Score'] == p['comparison_score'] >= old['Score']
        results = {}
        for name, row in frozen['predictions'].items():
            pred = va.load_jsonl(row['prediction']['path'])
            metrics = ev.evaluate_report(gold, pred, body_scope='all', no_id_doc='text')
            put(out/f'{name}.metrics.json', metrics)
            delta = {k:metrics['metrics'][k]-v for k,v in old.items()}
            assert delta['menF1'] == delta['docF1'] == 0
            added = set(map(tuple, row['added_pairs']))
            tp = len(added & pairs(gold))
            results[name] = dict(metrics=metrics['metrics'], delta=delta,
                added=len(added), added_TP=tp, added_FP=len(added)-tp,
                delta_vs_previous_best=metrics['metrics']['Score']-p['comparison_score'])
        best = max(results, key=lambda k:results[k]['metrics']['Score'])
        retain = results[best]['delta_vs_previous_best'] > 0
        put(out/'result.json', dict(status='COMPLETE_TWO_FIXED_ORIGINALDEV_ENDPOINTS', at_utc=now(),
            baseline=old, previous_best_score=p['comparison_score'], results=results,
            retain=retain, selected=best if retain else None, new_v2=2, old_v2=0,
            fit_calls=0, NN_calls=0, not_blind=True, original_dev_read=True,
            freeze=ref(out/'freeze.json'), seconds=time.monotonic()-started))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('mode', choices=['assemble', 'evaluate'])
    a = ap.parse_args()
    run(a.plan, a.mode)
