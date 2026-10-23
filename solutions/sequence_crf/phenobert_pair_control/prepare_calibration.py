"""Bind the prespecified nine-cut script to four frozen new probability pools."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from finalize_plan import ref
import features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal-plan', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    formal = json.loads(Path(a.formal_plan).read_text())
    assert formal['execution_authorized'] is True
    policy = formal['calibration']
    assert policy['cuts'] == [i / 10 for i in range(1, 10)]
    assert policy['objective'] == 'pooled (menF1 + docF1) / 2'
    assert policy['tie'] == 'nearest cut to .4, then higher cut'
    root = Path(formal['output_dir'])
    freeze = json.loads((root / 'complete-freeze.json').read_text())
    assert freeze['plan_sha256'] == features.sha(a.formal_plan)
    assert freeze['all_four_predictions_complete'] is True
    result = json.loads((root / 'results.json').read_text())
    assert result['freeze_sha256'] == features.sha(root / 'complete-freeze.json')
    assert result['quality_evaluation_complete'] is True
    folds = []
    for item in freeze['folds']:
        i = item['fold']
        for role in ['scores', 'new_manifest', 'new_prediction']:
            features.checked(item[role])
        folds.append(dict(fold=i, scores=item['scores'], manifest=item['new_manifest'],
                          baseline_prediction=item['new_prediction'],
                          baseline_metrics=ref(root / f'fold{i}/pair.metrics.json'),
                          old_template=item['baseline_prediction']))
    assert [x['fold'] for x in folds] == [0, 1, 2, 3]
    source = features.checked(policy['source_snapshot'])
    evaluator = Path(formal['evaluation_source']) / 'evaluate.py'
    plan = dict(experiment='Same fixed top3 pools plus public pair means: predeclared nine-cut training calibration',
                frozen_at=datetime.now(timezone.utc).isoformat(),
                formal_plan=ref(a.formal_plan), formal_freeze=ref(root / 'complete-freeze.json'),
                source_policy=policy, folds=folds, cuts=policy['cuts'], CPU_cap_seconds=30,
                output_dir=str(root.parent / 'calibration'),
                sources={str(source): features.sha(source), str(evaluator): features.sha(evaluator)},
                evaluation_source=formal['evaluation_source'], train64=formal['train64_gold'],
                old_calibration_results=policy['old_top3_result'],
                objective=policy['objective'], tie=policy['tie'],
                no_new_fit=True, no_originaldev=True,
                selection_is_training_calibration_not_unbiased_generalization=True)
    with Path(a.output).open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan=ref(a.output), calibration_source=policy['source_snapshot'])))


if __name__ == '__main__':
    main()
