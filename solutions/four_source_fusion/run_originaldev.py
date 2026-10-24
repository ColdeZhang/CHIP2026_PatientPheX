"""Prelocked full64 four-source agreement; one endpoint, no fitted fusion."""
import argparse
from collections import Counter
import copy
import importlib
import json
import os
from pathlib import Path
import sys
from run import checked, now, put, ref


def agreement(sets):
    votes = Counter()
    for current in sets.values():
        votes.update(current)
    return {key for key, count in votes.items() if count >= 2}


def run(plan_path, mode):
    plan = json.loads(Path(plan_path).read_text())
    refs, out = plan['refs'], Path(plan['output'])
    assert plan['scope'] == 'original_dev16_development_full64'
    assert plan['formula'] == 'B901 UNION at_least_two_of(G,S,Q,C)'
    assert refs['source'] == ref(__file__)
    assert refs['helper'] == ref(Path(__file__).with_name('run.py'))
    assert refs['baseline']['sha256'] == '9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67'
    checked(refs['decision'])
    evaluator = checked(refs['evaluator'])
    checked(refs['validator'])
    sys.path.insert(0, str(evaluator.parent))
    ev = importlib.import_module('evaluate')
    va = importlib.import_module('validate_predictions')
    assert ev.VERSION == 'patientphex-public-approx-v2'

    def pairs(rows):
        return {(d['pmc_id'], pid, h) for d in rows
                for pid, hs in ev.associations(d).items() for h in hs}

    if mode == 'assemble':
        forbidden = os.path.realpath(refs['gold']['path'])
        def guard(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                if os.path.realpath(os.fsdecode(args[0])) == forbidden:
                    raise PermissionError('Held gold blocked before full prediction freeze')
        sys.addaudithook(guard)
        out.mkdir(parents=True, exist_ok=False)
        put(out/'started.json', dict(at_utc=now(), plan=ref(plan_path)))
        base = va.load_jsonl(checked(refs['baseline']))
        inputs = va.load_jsonl(checked(refs['input']))
        by = {d['pmc_id']: d for d in base}
        old = pairs(base)
        assert len(base) == len(by) == len(inputs) == 16
        assert set(by) == {d['pmc_id'] for d in inputs}
        assert len(old) == 416
        assert sum(len(d['entities']) for d in base) == 1061
        assert sum(len(d['association']) for d in base) == 36
        sets = {}
        for name in ('grounded', 'sampled', 'qwen3', 'coded'):
            rows = va.load_jsonl(checked(refs[name]))
            assert len(rows) == 16 and {d['pmc_id'] for d in rows} == set(by)
            for d in rows:
                b = by[d['pmc_id']]
                assert {k: v for k, v in d.items() if k not in ('association', 'entities')} == {
                    k: v for k, v in b.items() if k not in ('association', 'entities')}
                assert ev.associations(d).keys() == ev.associations(b).keys()
            # Sources may be pure or B901-union; all B901 atoms are already protected.
            sets[name] = pairs(rows) - old
        assert {k: len(v) for k, v in sets.items()} == plan['expected_source_additions']
        b3 = va.load_jsonl(checked(refs['three_source_prediction']))
        assert pairs(b3) == old | agreement({k: v for k, v in sets.items() if k != 'coded'})
        additions = agreement(sets)
        pred = copy.deepcopy(base)
        for d in pred:
            for a in d['association']:
                a['phenotype'].extend(sorted(h for doc, pid, h in additions
                    if doc == d['pmc_id'] and pid == a['patient_id']))
        assert pairs(pred) == old | additions
        assert pairs(b3) <= pairs(pred)
        for b, d in zip(base, pred):
            assert {k: v for k, v in b.items() if k != 'association'} == {
                k: v for k, v in d.items() if k != 'association'}
            for before, after in zip(b['association'], d['association']):
                assert after['phenotype'][:len(before['phenotype'])] == before['phenotype']
                assert {k: v for k, v in before.items() if k != 'phenotype'} == {
                    k: v for k, v in after.items() if k != 'phenotype'}
        valid = va.validate_predictions(inputs, pred, va.read_hpo(checked(refs['hpo'])))
        put(out/'format.json', valid)
        assert valid['valid']
        content = ''.join(json.dumps(d, ensure_ascii=False, separators=(',', ':'))+'\n' for d in pred).encode()
        inherited = None
        for name in ('three_source', 'comparison', 'baseline'):
            pkey = 'baseline' if name == 'baseline' else name+'_prediction'
            if content == checked(refs[pkey]).read_bytes():
                inherited = name
                break
        path = out/'prediction.jsonl'
        with path.open('xb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        put(out/'freeze.json', dict(at_utc=now(), plan=ref(plan_path), prediction=ref(path),
            format=ref(out/'format.json'), source_additions={k:len(v) for k,v in sets.items()},
            added_from_901=sorted(additions), added_from_three_source=sorted(pairs(pred)-pairs(b3)),
            final_relations=len(pairs(pred)), inherited_exact_prediction=inherited,
            baseline_relation_prefix_exact=True, nonassociation_objects_exact=True,
            gold_read=False, fit_calls=0, NN_calls=0))
    else:
        freeze = json.loads((out/'freeze.json').read_text())
        assert freeze['plan'] == ref(plan_path)
        checked(freeze['prediction']); checked(freeze['format'])
        prior = {}
        for name in ('baseline', 'comparison', 'three_source'):
            prior[name] = json.loads(checked(refs[name+'_metrics']).read_text())['metrics']
        assert prior['baseline']['Score'] == .7543613981990362
        assert prior['comparison']['Score'] == plan['comparison_score']
        inherited = freeze['inherited_exact_prediction']
        if inherited is not None:
            metrics = prior[inherited]
            new_v2 = 0
            diagnosis = None
        else:
            put(out/'evaluation-intent.json', dict(at_utc=now(), freeze=ref(out/'freeze.json'), new_v2=1, old_v2=0))
            gold = va.load_jsonl(checked(refs['gold']))
            pred = va.load_jsonl(checked(freeze['prediction']))
            report = ev.evaluate_report(gold, pred, body_scope='all', no_id_doc='text')
            report.update(prediction_sha256=freeze['prediction']['sha256'], gold_sha256=refs['gold']['sha256'])
            put(out/'metrics.json', report)
            metrics = report['metrics']; new_v2 = 1
            new = set(map(tuple, freeze['added_from_three_source']))
            tp = len(new & pairs(gold))
            diagnosis = dict(added=len(new), TP=tp, FP=len(new)-tp)
        assert metrics['menF1'] == prior['baseline']['menF1']
        assert metrics['docF1'] == prior['baseline']['docF1']
        put(out/'result.json', dict(at_utc=now(), status='COMPLETE_FIXED_FULL64_FOUR_SOURCE',
            metrics=metrics, comparisons=prior,
            Score_delta={k:metrics['Score']-v['Score'] for k,v in prior.items()},
            retain=metrics['Score']>prior['comparison']['Score'], new_v2=new_v2, old_v2=0,
            inherited_exact_prediction=inherited, incremental_diagnosis=diagnosis,
            freeze=ref(out/'freeze.json'), development_not_blind=True, NN_calls=0, fit_calls=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('mode', choices=['assemble', 'evaluate'])
    a = ap.parse_args()
    run(a.plan, a.mode)
