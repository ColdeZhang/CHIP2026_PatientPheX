#!/usr/bin/env python3
"""Evaluate exactly two frozen originaldev arms; reuse the two old metrics."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import evaluate_public_selector as single
base,evaluate,validator,common=single.base,single.evaluate,single.validator,single.common


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
    plan=json.loads((root/'plan.json').read_text());freeze=json.loads((root/'complete-freeze.json').read_text())
    assert freeze['plan_sha256']==base.sha(root/'plan.json') and freeze['evaluation_gold_read'] is False
    assert set(freeze['predictions'])=={'old_selector','current_best','new_rules','preserved_sft'}
    for item in list(freeze['predictions'].values())+list(freeze['support'].values()):assert base.sha(item['path'])==item['sha256']
    for path,digest in plan['sources'].items():assert base.sha(path)==digest
    assert base.sha(plan['baseline_metrics']['path'])==plan['baseline_metrics']['sha256']
    assert base.sha(plan['best_metrics']['path'])==plan['best_metrics']['sha256']
    old_metrics=json.loads(Path(plan['baseline_metrics']['path']).read_text())['metrics']
    best_metrics=json.loads(Path(plan['best_metrics']['path']).read_text())['metrics']
    predictions={k:base.read_jsonl(v['path']) for k,v in freeze['predictions'].items()}
    keyed={k:{d['pmc_id']:d for d in rows} for k,rows in predictions.items()}
    assert len(keyed['new_rules'])==16 and all(set(d)==set(keyed['new_rules']) for d in keyed.values())
    for did in keyed['new_rules']:
        assert keyed['new_rules'][did]['entities']==keyed['preserved_sft'][did]['entities']
        assert keyed['preserved_sft'][did]['association']==keyed['current_best'][did]['association']
    # First originaldev gold read in this fixed migration, after both arms
    # and old controls are fully bound. Historical dev usage is disclosed.
    gold=base.read_jsonl(plan['gold']);assert len(gold)==16
    aligned={k:[rows[d['pmc_id']] for d in gold] for k,rows in keyed.items()}
    hpo=validator.read_hpo(plan['hpo']);arms={}
    for name in ['new_rules','preserved_sft']:
        report=evaluate.evaluate_report(gold,aligned[name],'all','text')
        check=validator.validate_predictions(gold,aligned[name],hpo);assert check['valid'] and check['error_count']==0
        for suffix,item in [('metrics',report),('validation',check)]:
            (root/(name+'.'+suffix+'.json')).write_text(json.dumps(item,indent=2)+'\n')
        arms[name]=dict(metrics=report['metrics'],counts=report['counts'],
            delta_vs_old_selector={k:report['metrics'][k]-old_metrics[k] for k in old_metrics},
            delta_vs_current_best={k:report['metrics'][k]-best_metrics[k] for k in best_metrics},
            changes_vs_old_selector=common.changes(gold,aligned['old_selector'],aligned[name]),
            changes_vs_current_best=common.changes(gold,aligned['current_best'],aligned[name]),
            errors=check['error_count'],warnings=check['warning_count'])
    assert arms['preserved_sft']['metrics']['micF1']==best_metrics['micF1']
    assert arms['preserved_sft']['metrics']['macF1']==best_metrics['macF1']
    result=dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=base.sha(__file__),
        evaluator_sha256=base.sha(evaluate.__file__),validator_sha256=base.sha(validator.__file__),
        plan_sha256=base.sha(root/'plan.json'),freeze_sha256=base.sha(root/'complete-freeze.json'),
        old_selector_metrics=old_metrics,current_best_metrics=best_metrics,arms=arms,
        limitations=['Originaldev used historically for comparisons; this is one fixed development migration, not blind testing.',
            'Four task-OOF train48 pools fit new train64 selector; public model has no task fit.',
            'Full64 original DictVectorizer has85 columns, plus six source fields; expected91 natural columns without dropping a feature.',
            'ArmA uses legacy patient rules; ArmB preserves all aa107790 SFT associations independently of the final entity set.',
            'No new dev-based threshold, feature, epoch or relation selection.'])
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
