#!/usr/bin/env python3
"""Freeze all eight train64 OOF outputs, then evaluate once under a 120s cap."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import decode_weighted_intervals as decoder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def artifact(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); a = p.parse_args()
    r = Path(a.directory); started = time.monotonic(); signal.alarm(120)
    plan = json.loads((r/'plan.json').read_text())
    assert plan['cut'] == .3 and plan['total_cpu_cap_seconds'] == 120
    assert not (r/'complete-freeze.json').exists() and not (r/'results.json').exists()
    for path, digest in plan['sources'].items(): assert sha(path) == digest
    assert sha(plan['contracts']['path']) == plan['contracts']['sha256']
    assert sha(plan['baseline_reuse']['freeze']['path']) == plan['baseline_reuse']['freeze']['sha256']
    old_freeze = json.loads(Path(plan['baseline_reuse']['freeze']['path']).read_text())
    freezes = []; diagnostics = []; predictions = {}; seen = set()
    for spec in plan['folds']:
        for role in ['scores', 'baseline_prediction', 'fit_manifest', 'input']:
            assert sha(spec[role]['path']) == spec[role]['sha256']
        assert old_freeze['folds'][spec['fold']]['baseline_prediction']['sha256'] == spec['baseline_prediction']['sha256']
        rows = read(spec['scores']['path']); old = read(spec['baseline_prediction']['path']); inputs = read(spec['input']['path'])
        ids = [d['pmc_id'] for d in old]
        assert len(ids) == len(set(ids)) == 16 and ids == [d['pmc_id'] for d in rows] == [d['pmc_id'] for d in inputs]
        assert not seen.intersection(ids); seen.update(ids)
        assert all('entities' not in d and 'association' not in d for d in inputs)
        out = r/f'fold{spec["fold"]}'; out.mkdir(exist_ok=False)
        shutil.copyfile(spec['baseline_prediction']['path'], out/'baseline.jsonl')
        new = []
        for scored, doc in zip(rows, old):
            assert decoder.greedy(scored['candidates']) == doc['entities']
            selected, diag = decoder.decode(scored['candidates'], doc['entities'])
            updated = dict(doc, entities=selected)
            assert {k:v for k,v in updated.items() if k != 'entities'} == {k:v for k,v in doc.items() if k != 'entities'}
            new.append(updated)
            diagnostics.append(dict(fold=spec['fold'], pmc_id=doc['pmc_id'], old_count=len(doc['entities']),
                                    new_count=len(selected), changed=selected != doc['entities'], **diag))
        (out/'wis.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in new))
        freezes.append(dict(fold=spec['fold'], baseline=artifact(out/'baseline.jsonl'), wis=artifact(out/'wis.jsonl')))
        predictions[spec['fold']] = dict(baseline=old, wis=new)
    assert len(seen) == 64
    write(r/'input-only-diagnostics.json', diagnostics)
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(r/'plan.json'),
                  source_sha256=sha(__file__), all_eight_predictions=freezes, new_evaluation_started=False,
                  all_nonentity_objects_unchanged=True, baseline_greedy_entities_exact=True,
                  seconds_before_labels=time.monotonic()-started, diagnostics=artifact(r/'input-only-diagnostics.json'))
    write(r/'complete-freeze.json', freeze)
    # Import and read labels only after all eight complete predictions freeze.
    import evaluate_exact_gate_control as common
    evaluate, validator = common.evaluate, common.validator
    assert sha(evaluate.__file__) == plan['evaluator_sha256'] and sha(validator.__file__) == plan['validator_sha256']
    assert sha(plan['train64']['path']) == plan['train64']['sha256']
    gold_all = read(plan['train64']['path']); truth = {d['pmc_id']:d for d in gold_all}
    assert seen == set(truth) and len(gold_all) == 64
    hpo = validator.read_hpo(plan['hpo']['path']); assert sha(plan['hpo']['path']) == plan['hpo']['sha256']
    evaluation_started = datetime.now(timezone.utc).isoformat()
    folds = []; byid = {'baseline':{}, 'wis':{}}
    for spec in plan['folds']:
        f = spec['fold']; arms = predictions[f]; gold = [truth[d['pmc_id']] for d in arms['baseline']]
        item = dict(fold=f, arms={}, changes=common.changes(gold, arms['baseline'], arms['wis']))
        for arm, pred in arms.items():
            if arm == 'baseline':
                for role in ['baseline_metrics', 'baseline_validation']:
                    assert sha(spec[role]['path']) == spec[role]['sha256']
                report = json.loads(Path(spec['baseline_metrics']['path']).read_text())
                check = json.loads(Path(spec['baseline_validation']['path']).read_text())
            else:
                report = evaluate.evaluate_report(gold, pred, 'all', 'text')
                check = validator.validate_predictions(gold, pred, hpo)
            assert check['valid'] and check['error_count'] == 0
            write(r/f'fold{f}'/f'{arm}.metrics.json', report); write(r/f'fold{f}'/f'{arm}.validation.json', check)
            item['arms'][arm] = dict(metrics=report['metrics'], counts=report['counts'], errors=check['error_count'], warnings=check['warning_count'])
            byid[arm].update({d['pmc_id']:d for d in pred})
        item['deltas'] = {k:item['arms']['wis']['metrics'][k]-item['arms']['baseline']['metrics'][k] for k in report['metrics']}
        folds.append(item)
    combined = {arm:[byid[arm][d['pmc_id']] for d in gold_all] for arm in byid}
    old_pooled = plan['baseline_reuse']['pooled_metrics']
    assert sha(old_pooled['path']) == old_pooled['sha256']
    pooled = dict(baseline=json.loads(Path(old_pooled['path']).read_text()),
                  wis=evaluate.evaluate_report(gold_all, combined['wis'], 'all', 'text'))
    for arm, report in pooled.items(): write(r/f'{arm}.pooled.metrics.json', report)
    result = dict(evaluation_started_utc=evaluation_started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=sha(r/'plan.json'), freeze_sha256=sha(r/'complete-freeze.json'), source_sha256=sha(__file__),
                  folds=folds, pooled=pooled, pooled_changes=common.changes(gold_all, combined['baseline'], combined['wis']),
                  pooled_deltas={k:pooled['wis']['metrics'][k]-pooled['baseline']['metrics'][k] for k in report['metrics']},
                  unweighted_mean={arm:{k:sum(f['arms'][arm]['metrics'][k] for f in folds)/4 for k in report['metrics']} for arm in pooled},
                  input_only_summary=dict(changed_documents=sum(d['changed'] for d in diagnostics), at_cut=sum(d['at_cut'] for d in diagnostics),
                                          utility_gain=sum(d['utility_gain'] for d in diagnostics)),
                  seconds=time.monotonic()-started, new_fit=False, gpu_used=False, originaldev_read=False,
                  baseline_metrics_reused_without_new_evaluation=True,
                  limitations=plan['limitations'])
    assert result['seconds'] < 120
    write(r/'results.json', result); signal.alarm(0)
    print(json.dumps({k:result[k] for k in ['pooled_deltas','pooled_changes','input_only_summary','seconds']}))


if __name__ == '__main__': main()
