#!/usr/bin/env python3
"""One fixed entity-only nesting transfer after fourfold independent review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import time
import decode_nested_selector as decoder
from run_weighted_interval_control import artifact, read, sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    r=Path(a.directory);started=time.monotonic();signal.alarm(30)
    plan=json.loads((r/'plan.json').read_text())
    assert plan['cut']==.3 and plan['total_cpu_cap_seconds']==30
    assert not (r/'complete-freeze.json').exists() and not (r/'results.json').exists()
    for path,digest in plan['sources'].items():assert sha(path)==digest
    for role in ['input','scores','score_manifest','baseline','baseline_metrics','hpo','training_review','training_results']:
        assert sha(plan[role]['path'])==plan[role]['sha256']
    train_result=json.loads(Path(plan['training_results']['path']).read_text())
    assert sum(f['deltas']['Score']>=0 for f in train_result['folds'])>=3
    inputs=read(plan['input']['path']);scores=read(plan['scores']['path']);old=read(plan['baseline']['path'])
    assert len(inputs)==len(scores)==len(old)==16
    assert [d['pmc_id'] for d in inputs]==[d['pmc_id'] for d in scores]==[d['pmc_id'] for d in old]
    assert all('entities' not in d and 'association' not in d for d in inputs)
    _,allowed,_,_=decoder.prior.base.parse_hpo(plan['hpo']['path']);allowed.discard('HP:0000118')
    new=[];diagnostics=[]
    for doc,scored,prior in zip(inputs,scores,old):
        entities,counts=decoder.decode(scored['candidates'],prior['entities'],doc,allowed)
        updated=dict(prior,entities=entities)
        assert {k:v for k,v in updated.items() if k!='entities'}=={k:v for k,v in prior.items() if k!='entities'}
        new.append(updated);diagnostics.append(dict(pmc_id=doc['pmc_id'],old_entities=len(prior['entities']),new_entities=len(entities),**counts))
    shutil.copyfile(plan['baseline']['path'],r/'baseline.jsonl')
    (r/'nested.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in new))
    write(r/'input-only-diagnostics.json',diagnostics)
    freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(r/'plan.json'),source_sha256=sha(__file__),
                predictions={arm:artifact(r/f'{arm}.jsonl') for arm in ['baseline','nested']},
                all_prior_entities_and_nonentity_objects_preserved=True,new_evaluation_started=False,
                diagnostics=artifact(r/'input-only-diagnostics.json'),seconds_before_labels=time.monotonic()-started)
    write(r/'complete-freeze.json',freeze)
    # Only after both complete predictions freeze may the new dev quality be read.
    import evaluate_exact_gate_control as common
    assert sha(common.evaluate.__file__)==plan['evaluator_sha256'] and sha(common.validator.__file__)==plan['validator_sha256']
    assert sha(plan['gold']['path'])==plan['gold']['sha256']
    gold=read(plan['gold']['path']);assert [d['pmc_id'] for d in gold]==[d['pmc_id'] for d in old]
    evaluation_started=datetime.now(timezone.utc).isoformat()
    baseline=json.loads(Path(plan['baseline_metrics']['path']).read_text())
    report=common.evaluate.evaluate_report(gold,new,'all','text')
    check=common.validator.validate_predictions(gold,new,common.validator.read_hpo(plan['hpo']['path']))
    assert check['valid'] and check['error_count']==0
    for metric in ['micF1','macF1']:assert report['metrics'][metric]==baseline['metrics'][metric]
    assert report['counts']['relation']==baseline['counts']['relation']
    write(r/'nested.metrics.json',report);write(r/'nested.validation.json',check)
    summary=dict(status='one fixed dev entity-only transfer complete',plan_sha256=sha(r/'plan.json'),freeze_sha256=sha(r/'complete-freeze.json'),
                 evaluation_started_utc=evaluation_started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 baseline=baseline['metrics'],nested=report['metrics'],deltas={k:report['metrics'][k]-baseline['metrics'][k] for k in baseline['metrics']},
                 changes=common.changes(gold,old,new),counts=report['counts'],errors=check['error_count'],warnings=check['warning_count'],
                 candidate_decisions={key:sum(d.get(key,0) for d in diagnostics) for key in ['all_candidates','already_selected','below_cut','original_overlap_rejection','invalid_hpo_or_nil','negated','not_distinct_proper_nesting','added']},
                 seconds=time.monotonic()-started,new_fit=False,gpu_used=False,old_metrics_reused=True,limitations=plan['limitations'])
    assert summary['seconds']<30
    write(r/'results.json',summary);signal.alarm(0);print(json.dumps(summary))


if __name__=='__main__':main()
