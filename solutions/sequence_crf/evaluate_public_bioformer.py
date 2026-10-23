#!/usr/bin/env python3
"""One four-fold assessment after all Bioformer-source predictions freeze."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import evaluate_public_selector as prior
base,evaluate,validator,common=prior.base,prior.evaluate,prior.validator,prior.common


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();r=Path(a.directory)
    plan=json.loads((r/'plan.json').read_text());freeze=json.loads((r/'complete-freeze.json').read_text())
    assert freeze['plan_sha256']==base.sha(r/'plan.json') and freeze['new_evaluation_started'] is False
    assert len(freeze['predictions'])==4
    for path,h in plan['sources'].items():assert base.sha(path)==h
    for ff in freeze['predictions']:
        for info in ff['artifacts'].values():assert base.sha(info['path'])==info['sha256']
    reports=[];aftermap={};beforemap={};hpo=validator.read_hpo(plan['hpo'])
    for spec,ff in zip(plan['folds'],freeze['predictions']):
        assert spec['fold']==ff['fold']
        before=base.read_jsonl(ff['artifacts']['baseline']['path']);after=base.read_jsonl(ff['artifacts']['augmented']['path'])
        assert base.sha(spec['baseline_metrics']['path'])==spec['baseline_metrics']['sha256']
        old=json.loads(Path(spec['baseline_metrics']['path']).read_text())['metrics']
        # This is the first new evaluation-label read, after all four freezes.
        truth={d['pmc_id']:d for d in base.read_jsonl(spec['gold'])};gold=[truth[d['pmc_id']] for d in before]
        assert len(gold)==16 and [d['pmc_id'] for d in before]==[d['pmc_id'] for d in after]
        report=evaluate.evaluate_report(gold,after,'all','text');check=validator.validate_predictions(gold,after,hpo)
        assert check['valid'] and not check['error_count']
        cs={d['pmc_id']:d['candidates'] for d in base.read_jsonl(ff['artifacts']['scores']['path'])}
        # The legacy partition considered only pre-public sources as existing.
        # Mark PubMed-supported keys as existing solely for diagnostic grouping.
        grouped={did:[dict(c,semantic_source=c['semantic_source'] or c['public_present']) for c in rows] for did,rows in cs.items()}
        item=dict(fold=spec['fold'],baseline=old,augmented=report['metrics'],
            deltas={k:report['metrics'][k]-old[k] for k in old},changes=common.changes(gold,before,after),
            entity_change_origins=prior.breakdown(gold,before,after,grouped),
            origin_label_note='added_from_public_only_pool means new Bioformer-only candidate, with neither previous oldpool nor PubMed support.',
            errors=check['error_count'],warnings=check['warning_count'])
        target=r/('fold'+str(spec['fold']))
        for name,value in [('augmented.metrics.json',report),('augmented.validation.json',check),('results.json',item)]:
            (target/name).write_text(json.dumps(value,indent=2)+'\n')
        reports.append(item);aftermap.update({d['pmc_id']:d for d in after});beforemap.update({d['pmc_id']:d for d in before})
    gold=base.read_jsonl(plan['train64']);assert len(aftermap)==len(beforemap)==len(gold)==64
    after=[aftermap[d['pmc_id']] for d in gold];before=[beforemap[d['pmc_id']] for d in gold]
    pooled=evaluate.evaluate_report(gold,after,'all','text')
    assert base.sha(plan['baseline_pooled_metrics']['path'])==plan['baseline_pooled_metrics']['sha256']
    old=json.loads(Path(plan['baseline_pooled_metrics']['path']).read_text())['metrics']
    (r/'augmented.pooled.metrics.json').write_text(json.dumps(pooled,indent=2)+'\n')
    (r/'augmented.pooled.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in after))
    result=dict(scope=plan['scope'],evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.sha(__file__),plan_sha256=base.sha(r/'plan.json'),freeze_sha256=base.sha(r/'complete-freeze.json'),
        evaluator_sha256=base.sha(evaluate.__file__),validator_sha256=base.sha(validator.__file__),folds=reports,
        pooled=dict(baseline=old,augmented=pooled['metrics'],deltas={k:pooled['metrics'][k]-old[k] for k in old},changes=common.changes(gold,before,after)),
        unweighted_mean={arm:{k:sum(f[arm][k] for f in reports)/4 for k in old} for arm in ['baseline','augmented']},
        limitations=['Development folds previously used; no new originaldev labels.',
        'Independent Bioformer source plus six features: method increment, not pure one-column effect.',
        'Public model forwards fold-independent; all task-fit generators remain per-fold and secondary selector train48 only.',
        'No threshold/epoch/source-order sweep. Existing PubMed source fields retain their meaning.',
        'New-candidate source attribution uses key presence, not causal attribution; NIL unit grouping follows existing diagnostic.'])
    (r/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pooled':result['pooled'],'fold_deltas':[f['deltas']['Score'] for f in reports]}))


if __name__=='__main__':main()
