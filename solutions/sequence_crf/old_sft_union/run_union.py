#!/usr/bin/env python3
"""One predeclared original8B SFT48 union original9B SFT48 control, CPU <=60s."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
EXPECTED = {
 'old8B_pred':'c31574728477fbe0490e7198a826e18920f9316224396f5ff803e444840ea23f',
 'old9B_pred':'63aefe68b07664bd1e64d31e0008d05259020c727e836fbd709a5837e8929edb',
 'old9B_metrics':'e16ff5b0d593ca431ef20ca68afc1ddf8051374da077428553fcd436d35b8218',
 'prompts':'d387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0',
 'input':'ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37',
 'evaluator':'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781',
 'validator':'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f',
 'hpo':'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
}
GOLD_SHA='630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48'
KEEP='delta(micF1+macF1)>0 and delta(macF1)>=0'


def ref(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def read(path):return json.loads(Path(path).read_text())


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fsync_dir(Path(path).parent)


def check_ref(item):
    actual=ref(item['path'])
    assert all(actual[k]==item[k] for k in ['bytes','sha256'])
    return actual


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


def relations(doc):
    result = {}
    for association in doc['association']:
        pid = association['patient_id']; assert pid not in result
        result[pid] = {part for value in association['phenotype']
                       for part in (value.split(';') if COMPOUND.fullmatch(value) else [value])}
    return result


def main():
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed CPU endpoint exceeded60s')))
    signal.alarm(60)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();b=args.plan.resolve().parent;plan=read(args.plan);refs=plan['references']
    assert plan['source']==ref(__file__)
    assert plan['CPU_total_cap_seconds']==60 and plan['keep_rule']==KEEP
    assert plan['new_evaluator_calls']==1 and plan['baseline_evaluator_calls']==0
    assert plan['union_policy']=='literal old9B prediction union old8B prediction; no other relations'
    assert set(refs)==set(EXPECTED) and plan['gold']['sha256']==GOLD_SHA
    assert plan['new_control_quality_read'] is False
    for name,item in refs.items():
        assert item['sha256']==EXPECTED[name];check_ref(item)
    # No gold locator open/hash/stat above this point or during assembly.
    out=b/'terminal';out.mkdir(exist_ok=False)
    old8,old9,prompts=rows(refs['old8B_pred']['path']),rows(refs['old9B_pred']['path']),rows(refs['prompts']['path'])
    assert len(old8)==len(old9)==16 and len(prompts)==58 and all('answer' not in p for p in prompts)
    assert len({str(d['pmc_id']) for d in old9})==16
    assert [d['pmc_id'] for d in old8]==[d['pmc_id'] for d in old9]
    assert {str(p['pmc_id']) for p in prompts}=={str(d['pmc_id']) for d in old9}
    assert sum(len(d['entities']) for d in old9)==947
    assert sum(len(d['association']) for d in old9)==47
    assert sum(len(p['code_map']) for p in prompts)==415
    prediction,contract=build_union(old9,old8,prompts)
    assert contract==dict(pool_pairs=1166,old_relations=276,primary_relations=297,union_relations=330,
        added_over_old=54,removed_from_old=0,all_nonassociation_exact=True,all_patient_metadata_exact=True)
    prediction_path=out/'union.pred.jsonl'
    with prediction_path.open('x',encoding='utf-8') as f:
        for doc in prediction:f.write(json.dumps(doc,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    fsync_dir(out)
    save(out/'prediction-freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source=ref(__file__),plan=ref(args.plan),prediction=ref(prediction_path),references=refs,
        contract=contract,documents=16,patients=47,entities=947,prompt_rows=58,candidates=415,
        gold_read=False,gold_hash_computed=False,new_evaluator_calls_before_freeze=0,
        old8B_one_fixed_LR_fallback_preserved=True,only_this_fixed_union=True))
    frozen=read(out/'prediction-freeze.json')
    assert frozen['source']==ref(__file__) and frozen['plan']==ref(args.plan)
    assert frozen['gold_read'] is False and frozen['new_evaluator_calls_before_freeze']==0
    check_ref(frozen['prediction'])
    for item in frozen['references'].values():check_ref(item)
    save(out/'before-gold.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),
        freeze=ref(out/'prediction-freeze.json'),source=ref(__file__),plan=ref(args.plan),
        gold_not_yet_read=True,gold_not_yet_hashed=True,new_evaluator_calls=0,
        next_action='format validation then first gold SHA/read and exactly one all/text v2'))
    sys.path.insert(0,str(Path(refs['evaluator']['path']).parent))
    import evaluate as evaluator
    import validate_predictions as validator
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    assert ref(evaluator.__file__)['sha256']==EXPECTED['evaluator']
    assert ref(validator.__file__)['sha256']==EXPECTED['validator']
    inputs=validator.load_jsonl(refs['input']['path'])
    hpo=validator.read_hpo(refs['hpo']['path']);assert hpo['version']=='hp/releases/2026-06-23'
    prediction=validator.load_jsonl(prediction_path)
    validation=validator.validate_predictions(inputs,prediction,hpo)
    save(out/'validation.json',validation);assert validation['valid']
    before=read(out/'before-gold.json');assert before['freeze']==ref(out/'prediction-freeze.json')
    check_ref(plan['gold'])
    gold=validator.load_jsonl(plan['gold']['path'])
    assert {d['pmc_id'] for d in gold}=={d['pmc_id'] for d in inputs}
    metrics=evaluator.evaluate_report(gold,prediction,'all','text')
    save(out/'metrics.json',metrics)
    baseline=read(refs['old9B_metrics']['path'])
    assert baseline['prediction_sha256']==EXPECTED['old9B_pred'] and baseline['gold_sha256']==GOLD_SHA
    delta={name:metrics['metrics'][name]-value for name,value in baseline['metrics'].items()}
    assert delta['menF1']==delta['docF1']==0
    oldmap={d['pmc_id']:relations(d) for d in old9};truth={d['pmc_id']:relations(d) for d in gold}
    changes=dict(added_tp=0,added_fp=0,removed_tp=0,removed_fp=0)
    for doc in prediction:
        current=relations(doc);previous=oldmap[doc['pmc_id']];assert set(current)==set(previous)
        for pid in current:
            assert previous[pid]<=current[pid]
            added=current[pid]-previous[pid]
            changes['added_tp']+=len(added&truth[doc['pmc_id']][pid])
            changes['added_fp']+=len(added-truth[doc['pmc_id']][pid])
    assert changes['added_tp']+changes['added_fp']==54
    keep=delta['micF1']+delta['macF1']>0 and delta['macF1']>=0
    result=dict(status='candidate_pending_F1_robustness' if keep else 'negative_closed_no_scan',
        completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        source=ref(__file__),plan=ref(args.plan),prediction=ref(prediction_path),freeze=ref(out/'prediction-freeze.json'),
        before_gold=ref(out/'before-gold.json'),format_validation=ref(out/'validation.json'),metrics_ref=ref(out/'metrics.json'),
        metrics=metrics['metrics'],baseline_metrics=baseline['metrics'],delta=delta,changes=changes,
        keep=keep,keep_rule=KEEP,contract=contract,new_evaluator_calls=1,baseline_evaluator_calls=0,
        baseline_reused_not_rescored=True,old8B_one_fixed_LR_fallback_preserved=True,
        development_F0_not_blind=True,no_patient_first_or_other_rule_union=True,
        no_training_NN_GPU=True,no_originaldev_or_official=True,no_scan=True)
    save(out/'results.json',result)
    signal.alarm(0)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
