"""One explicitly pre-quality amended union; preserve the original primary endpoint."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import finish

AMENDMENT_SHA = 'f43f6f60ff39ee9883ca83e64f23dda81913d9cdfc454e1a627bd0c7c43c959f'
PRIMARY_SOURCE_SHA = '8fd2809a5e5619887abe5dea14da923799554f83906a9e125f9718853cdae4d5'


def build_union(old, new, prompts):
    pool = {(str(p['pmc_id']), str(pid), hpo) for p in prompts
            for pid in p['patient_ids'] for hpo in p['code_map'].values()}
    assert len(old) == len(new)
    result = copy.deepcopy(old)
    old_pairs, new_pairs, result_pairs = set(), set(), set()
    for before, current, output in zip(old, new, result):
        assert {k:v for k,v in before.items() if k != 'association'} == {
            k:v for k,v in current.items() if k != 'association'}
        assert len(before['association']) == len(current['association'])
        for a, b, c in zip(before['association'], current['association'], output['association']):
            assert {k:v for k,v in a.items() if k != 'phenotype'} == {
                k:v for k,v in b.items() if k != 'phenotype'}
            pmc, pid = str(before['pmc_id']), str(a['patient_id'])
            left, right = set(a['phenotype']), set(b['phenotype'])
            assert len(left) == len(a['phenotype']) and len(right) == len(b['phenotype'])
            c['phenotype'] = sorted(left | right)
            old_pairs.update((pmc,pid,h) for h in left)
            new_pairs.update((pmc,pid,h) for h in right)
            result_pairs.update((pmc,pid,h) for h in c['phenotype'])
    assert old_pairs <= pool and new_pairs <= pool
    assert result_pairs == old_pairs | new_pairs
    return result, dict(pool_pairs=len(pool), old_relations=len(old_pairs),
                       primary_relations=len(new_pairs), union_relations=len(result_pairs),
                       added_over_old=len(result_pairs-old_pairs), removed_from_old=0,
                       all_nonassociation_exact=True, all_patient_metadata_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['assemble-common', 'evaluate-both'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); b = args.directory; started = time.monotonic()
    amendment = b/'secondary-union-amendment/amendment.json'
    assert finish.ref(amendment)['sha256'] == AMENDMENT_SHA
    assert finish.ref(finish.__file__)['sha256'] == PRIMARY_SOURCE_SHA
    policy = finish.read(amendment)
    assert not policy['new_patient_first_quality_read'] and not policy['new_patient_first_responses_read']
    plan = finish.read(b/'endpoint-plan.json'); refs = plan['references']
    assert finish.ref(b/'endpoint-plan.json')['sha256'] == policy['primary_endpoint_plan_sha256']
    for item in refs.values(): finish.check_ref(item)
    primary = finish.read(b/'assembled/freeze.json')
    for key in ['prediction','responses','parsed','inference_manifest']: finish.check_ref(primary[key])
    assert not primary['gold_read'] and primary['all80_strict']
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU union endpoint >30s')))
    out = b/'secondary-union'
    if args.mode == 'assemble-common':
        signal.alarm(30)
        assert not (b/'evaluation').exists() and not out.exists()
        old = finish.rows(refs['baseline_pred']['path'])
        new = finish.rows(primary['prediction']['path'])
        prompts = finish.rows(refs['prompts']['path'])
        prediction, contract = build_union(old,new,prompts)
        assert len(old) == len(new) == 16 and len(prompts) == 80
        assert sum(len(d['entities']) for d in old) == 1189
        assert contract['pool_pairs'] == 1557
        out.mkdir()
        with (out/'union.jsonl').open('x') as handle:
            for doc in prediction: handle.write(json.dumps(doc,ensure_ascii=False)+'\n')
        finish.save(out/'common-freeze.json',dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), amendment=finish.ref(amendment),
            source=finish.ref(__file__), primary_source=finish.ref(finish.__file__),
            primary_freeze=finish.ref(b/'assembled/freeze.json'), primary=primary['prediction'],
            secondary=finish.ref(out/'union.jsonl'), responses=primary['responses'],
            baseline_prediction=refs['baseline_pred'], baseline_metrics=refs['baseline_metrics'],
            contract=contract, entities=1189, gold_read=False,
            new_quality_calls_before_freeze=0, seconds=time.monotonic()-started))
        print(json.dumps(finish.read(out/'common-freeze.json'))); return
    common = finish.read(out/'common-freeze.json')
    assert common['source']['sha256'] == finish.ref(__file__)['sha256'] and common['gold_read'] is False
    for key in ['amendment','source','primary_source','primary_freeze','primary','secondary','responses',
                'baseline_prediction','baseline_metrics']: finish.check_ref(common[key])
    assert not (b/'evaluation').exists() and not (out/'evaluation').exists()
    finish.save(out/'before-both-quality.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
                common_freeze=finish.ref(out/'common-freeze.json'), gold_not_yet_read=True,
                exactly_two_new_arms=True, old_baseline_reused=True))
    # Original source, plan, keep rule and primary evaluator invocation stay intact.
    subprocess.run([sys.executable,str(Path(finish.__file__)),'evaluate','--directory',str(b)],
                   check=True,timeout=35)
    signal.alarm(30)
    evaluation = out/'evaluation'; evaluation.mkdir(exist_ok=False)
    finish.save(evaluation/'before-gold.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
                common_freeze=finish.ref(out/'common-freeze.json'), secondary_gold_not_yet_read=True))
    sys.path.insert(0,str(Path(refs['evaluator']['path']).parent))
    import evaluate as evaluator
    import validate_predictions as validator
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    finish.check_ref(plan['gold'])
    gold = validator.load_jsonl(plan['gold']['path'])
    inputs = validator.load_jsonl(refs['input']['path'])
    prediction = validator.load_jsonl(common['secondary']['path'])
    assert {d['pmc_id'] for d in gold} == {d['pmc_id'] for d in inputs}
    hpo = validator.read_hpo(refs['hpo']['path']); assert hpo['version'] == 'hp/releases/2026-06-23'
    validation = validator.validate_predictions(inputs,prediction,hpo)
    finish.save(evaluation/'validation.json',validation); assert validation['valid']
    metrics = evaluator.evaluate_report(gold,prediction,'all','text')
    finish.save(evaluation/'metrics.json',metrics)
    baseline = finish.read(refs['baseline_metrics']['path'])
    delta = {key:metrics['metrics'][key]-value for key,value in baseline['metrics'].items()}
    assert delta['menF1'] == delta['docF1'] == 0
    oldmap = {d['pmc_id']:finish.relations(d) for d in finish.rows(refs['baseline_pred']['path'])}
    truth = {d['pmc_id']:finish.relations(d) for d in gold}
    changes = dict(added_tp=0,added_fp=0,removed_tp=0,removed_fp=0)
    for doc in prediction:
        current, old = finish.relations(doc), oldmap[doc['pmc_id']]
        assert set(current) == set(old)
        for pid in current:
            assert old[pid] <= current[pid]
            added = current[pid]-old[pid]
            changes['added_tp'] += len(added & truth[doc['pmc_id']][pid])
            changes['added_fp'] += len(added-truth[doc['pmc_id']][pid])
    keep = delta['micF1']+delta['macF1'] > 0 and delta['macF1'] >= 0
    finish.save(evaluation/'results.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        metrics=metrics['metrics'], baseline=baseline['metrics'], baseline_reused_not_rescored=True,
        delta=delta,changes=changes,keep=keep,keep_rule=plan['keep_rule'],
        scope='One pre-quality fixed oldSFT32 union patient-first secondary; no other combinations or scans.',
        common_freeze=finish.ref(out/'common-freeze.json')))
    finish.save(out/'results.json',dict(primary=finish.read(b/'evaluation/results.json'),
                secondary=finish.read(evaluation/'results.json'),amendment=finish.ref(amendment),
                common_freeze=finish.ref(out/'common-freeze.json'),new_evaluator_calls=2,
                baseline_evaluator_calls=0,total_seconds=time.monotonic()-started))
    print(json.dumps(finish.read(out/'results.json')))


if __name__ == '__main__':
    main()
