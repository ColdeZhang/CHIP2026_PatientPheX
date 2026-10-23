#!/usr/bin/env python3
"""Evaluate three fixed follow-up folds only after their common freeze."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import evaluate_public_selector as single
base,evaluate,validator,common = single.base,single.evaluate,single.validator,single.common


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    a=p.parse_args();root=Path(a.directory)
    plan=json.loads((root/'plan.json').read_text());freeze=json.loads((root/'complete-freeze.json').read_text())
    assert freeze['plan_sha256']==base.sha(root/'plan.json') and freeze['new_evaluation_started'] is False
    assert len(freeze['predictions'])==3
    for path,digest in plan['sources'].items():assert base.sha(path)==digest
    for fold in freeze['predictions']:
        for item in fold['artifacts'].values():assert base.sha(item['path'])==item['sha256']
    for item in freeze['reused_fold0'].values():assert base.sha(item['path'])==item['sha256']
    assert base.sha(plan['baseline_pooled_metrics']['path'])==plan['baseline_pooled_metrics']['sha256']
    old_pooled=json.loads(Path(plan['baseline_pooled_metrics']['path']).read_text())
    prior0=json.loads(Path(freeze['reused_fold0']['results']['path']).read_text())
    new_predictions=base.read_jsonl(freeze['reused_fold0']['augmented']['path'])
    old_predictions=base.read_jsonl(freeze['reused_fold0']['baseline']['path'])
    reports=[dict(fold=0,baseline=prior0['baseline'],augmented=prior0['augmented'],deltas=prior0['deltas'],reused=True)]
    hpo=validator.read_hpo(plan['hpo'])
    # All three complete predictions and immutable fold0 controls were checked
    # above; only now read the matching outer evaluation labels.
    for spec,ff in zip(plan['folds'],freeze['predictions']):
        assert spec['fold']==ff['fold']
        target=root/('fold'+str(spec['fold']))
        before=base.read_jsonl(ff['artifacts']['baseline']['path']);after=base.read_jsonl(ff['artifacts']['augmented']['path'])
        truth={d['pmc_id']:d for d in base.read_jsonl(spec['gold'])}
        gold=[truth[d['pmc_id']] for d in before]
        assert len(truth)==16 and [d['pmc_id'] for d in before]==[d['pmc_id'] for d in after]
        assert base.sha(spec['baseline_metrics']['path'])==spec['baseline_metrics']['sha256']
        old=json.loads(Path(spec['baseline_metrics']['path']).read_text())
        report=evaluate.evaluate_report(gold,after,'all','text')
        check=validator.validate_predictions(gold,after,hpo);assert check['valid'] and check['error_count']==0
        scores={d['pmc_id']:d['candidates'] for d in base.read_jsonl(ff['artifacts']['scores']['path'])}
        result=dict(fold=spec['fold'],baseline=old['metrics'],augmented=report['metrics'],
            deltas={k:report['metrics'][k]-old['metrics'][k] for k in report['metrics']},
            changes=common.changes(gold,before,after),entity_change_origins=single.breakdown(gold,before,after,scores),
            errors=check['error_count'],warnings=check['warning_count'],reused=False)
        for name,value in [('augmented.metrics.json',report),('augmented.validation.json',check),('results.json',result)]:
            (target/name).write_text(json.dumps(value,indent=2)+'\n')
        reports.append(result);new_predictions.extend(after);old_predictions.extend(before)
    assert len({d['pmc_id'] for d in new_predictions})==len(new_predictions)==64
    gold_all=base.read_jsonl(plan['train64'])
    aftermap={d['pmc_id']:d for d in new_predictions};beforemap={d['pmc_id']:d for d in old_predictions}
    after=[aftermap[d['pmc_id']] for d in gold_all];before=[beforemap[d['pmc_id']] for d in gold_all]
    pooled=evaluate.evaluate_report(gold_all,after,'all','text')
    (root/'augmented.pooled.metrics.json').write_text(json.dumps(pooled,indent=2)+'\n')
    (root/'augmented.pooled.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in after))
    result=dict(scope=plan['scope'],evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.sha(__file__),plan_sha256=base.sha(root/'plan.json'),freeze_sha256=base.sha(root/'complete-freeze.json'),
        evaluator_sha256=base.sha(evaluate.__file__),validator_sha256=base.sha(validator.__file__),folds=reports,
        pooled={'baseline':old_pooled['metrics'],'augmented':pooled['metrics'],
            'deltas':{k:pooled['metrics'][k]-old_pooled['metrics'][k] for k in pooled['metrics']},
            'changes':common.changes(gold_all,before,after)},
        unweighted_mean={arm:{k:sum(f[arm][k] for f in reports)/4 for k in pooled['metrics']}
                         for arm in ['baseline','augmented']},
        limitations=['Outer0 motivated the fixed three-fold follow-up; entire campaign is development validation.',
            'Fold-specific train32 generators and final train48 selector fit preserved; only frozen public64 forward is shared.',
            'No extra feature/threshold/source/ordering selection or originaldev evaluation.',
            'Old fold metrics and fold0 endpoint reused; only three new models/predictions evaluated.'])
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fold_deltas':[f['deltas']['Score'] for f in reports],'pooled':result['pooled'],'mean':result['unweighted_mean']}))


if __name__=='__main__':main()
