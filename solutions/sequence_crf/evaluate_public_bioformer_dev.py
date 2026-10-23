#!/usr/bin/env python3
"""One originaldev evaluation of fixed .4 / preselected .3 entity controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import evaluate_public_selector as shared
base,evaluate,validator,common=shared.base,shared.evaluate,shared.validator,shared.common


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();r=Path(a.directory)
    plan=json.loads((r/'plan.json').read_text());freeze=json.loads((r/'complete-freeze.json').read_text())
    assert freeze['plan_sha256']==base.sha(r/'plan.json') and freeze['originaldev_labels_read'] is False
    for path,h in plan['sources'].items():assert base.sha(path)==h
    for info in freeze['artifacts'].values():assert base.sha(info['path'])==info['sha256']
    bm=plan['baseline_metrics'];assert base.sha(bm['path'])==bm['sha256'];oldreport=json.loads(Path(bm['path']).read_text())
    before=base.read_jsonl(freeze['artifacts']['baseline']['path']);oldassoc={d['pmc_id']:d['association'] for d in before}
    # First new originaldev gold read only after both new entity sets and old
    # complete prediction/model/provenance artifacts have frozen together.
    truth={d['pmc_id']:d for d in base.read_jsonl(plan['gold'])};gold=[truth[d['pmc_id']] for d in before]
    hpo=validator.read_hpo(plan['hpo']);reports={}
    for arm in ['fixed_point4','calibrated_point3']:
        after=base.read_jsonl(freeze['artifacts'][arm]['path'])
        assert [d['pmc_id'] for d in before]==[d['pmc_id'] for d in after]
        assert all(d['association']==oldassoc[d['pmc_id']] for d in after)
        report=evaluate.evaluate_report(gold,after,'all','text');check=validator.validate_predictions(gold,after,hpo)
        assert check['valid'] and check['error_count']==0
        for metric in ['micF1','macF1']:assert report['metrics'][metric]==oldreport['metrics'][metric]
        for name,value in [(arm+'.metrics.json',report),(arm+'.validation.json',check)]:
            (r/name).write_text(json.dumps(value,indent=2)+'\n')
        reports[arm]=dict(metrics=report['metrics'],deltas_from_previous_best={k:report['metrics'][k]-oldreport['metrics'][k] for k in report['metrics']},
            changes=common.changes(gold,before,after),errors=check['error_count'],warnings=check['warning_count'])
    result=dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=base.sha(__file__),plan_sha256=base.sha(r/'plan.json'),
        freeze_sha256=base.sha(r/'complete-freeze.json'),evaluator_sha256=base.sha(evaluate.__file__),validator_sha256=base.sha(validator.__file__),
        previous_best=oldreport['metrics'],arms=reports,
        calibrated_minus_fixed={k:reports['calibrated_point3']['metrics'][k]-reports['fixed_point4']['metrics'][k] for k in oldreport['metrics']},
        limitations=['Originaldev has already been used for comparisons; this is one fixed development transfer, not a new blind test.',
        'Same full64 HGB weights/features, only cut .4 versus .3 selected from fixed training-side OOF grid before this dev evaluation.',
        'Both arms preserve every aa107 SFT association object regardless of final entity set; relation metrics cannot improve here.',
        'No rule-relation arm, new threshold grid, extra model/source-order selection, or official submission.'])
    (r/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
