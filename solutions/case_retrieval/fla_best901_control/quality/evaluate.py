"""One complete FLA901 prediction, freeze first, one all/text v2 call."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from common import GOLD_SHA, MARKER, QUALITY_COMPLETE, checked, fsync_directory, module, read, ref, reference_gate, rows, save, sha
from inference_gate import verify
from adapt_assembly import run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    started=time.monotonic();plan_ref=ref(a.plan);plan=read(a.plan)
    records=reference_gate(plan,__file__,True);contract=read(plan['quality_contract']['path'])
    expected=contract['baseline_metrics'];assert expected['Score']==.7543613981990362
    fixed=plan['fixed_references'];baseline=read(fixed['baseline_metrics']['path'])
    assert baseline['metrics']==expected
    baseline_result=read(fixed['baseline_result']['path'])
    assert baseline_result['metrics']==expected and baseline_result['prediction']==fixed['baseline_prediction']
    native_replay=read(plan['native_replay']['path'])
    assert native_replay['status']=='PASS_NATIVE_FOUR_STREAM_901_BYTEEXACT_NO_QUALITY'
    assert native_replay['prediction']['sha256']==fixed['baseline_prediction']['sha256']
    coverage=verify(plan)
    result,AST_proof,patient_observations=run(plan,coverage)
    prediction=result['prediction'];pred=[json.loads(s) for s in prediction.splitlines()]
    inputs=rows(fixed['input']['path']);template=rows(fixed['task1_template']['path'])
    assert len(pred)==16 and sum(len(d['association']) for d in pred)==36
    assert sum(len(d['entities']) for d in pred)==1061
    assert [d['pmc_id'] for d in pred]==[d['pmc_id'] for d in inputs]
    assert [d['entities'] for d in pred]==[d['entities'] for d in template]
    directory=Path(plan['output_dir']);directory.mkdir(parents=True,exist_ok=False);fsync_directory(directory.parent)
    with (directory/'prediction.jsonl').open('xb') as f:f.write(prediction);f.flush();os.fsync(f.fileno())
    prediction_ref=ref(directory/'prediction.jsonl')
    save(directory/'assembly.json',dict(counts={k:len(v) for k,v in result['keys'].items()},
        parser_statistics=result['parser_statistics'],patient_codec_statistics=result['patient_codec_statistics'],
        AST_identity_gate_only_proof=AST_proof,all_nonassociation_literal_bytes_preserved=True,
        method='N sorted then oldGate public append then R literal union; replace oldGate public with newGate public; append P minus reconstructed oldGate a179',
        historical_SFTRaw_or_sixkeys_used=False,intermediate_predictions_saved_or_scored=False))
    save(directory/'patient-codec-decisions.json',patient_observations)
    for record in records+[prediction_ref]:checked(record)
    assert ref(a.plan)==plan_ref
    freeze_path=directory/'complete-prediction-freeze.json'
    save(freeze_path,dict(status='COMPLETE_SINGLE_PIPELINE_PREDICTION_BEFORE_FIRST_GOLD_BYTE',
        at_utc=datetime.now(timezone.utc).isoformat(),plan=plan_ref,prediction=prediction_ref,
        assembly=ref(directory/'assembly.json'),patient_codec=ref(directory/'patient-codec-decisions.json'),
        coverage=coverage,documents=16,patients=36,entities=1061,
        all_nonassociation_literal_bytes_exact=True,gold_read=False,new_v2_calls=0))
    freeze=ref(freeze_path);fsync_directory(directory)
    save(Path(plan['actual']['root'])/MARKER,dict(freeze=freeze,new_v2_calls=1,old_v2_calls=0))
    save(directory/'evaluation-started.json',dict(freeze=freeze,new_v2_calls=1,old_v2_calls=0))
    # H50 registration and every possible gold byte occur only after full freeze.
    if 'validate_predictions' in sys.modules:
        assert sha(sys.modules['validate_predictions'].__file__)==fixed['validator']['sha256']
    validator=module('validate_predictions',checked(fixed['validator']))
    evaluator=module('fixed_original901_FLA_public_v2',checked(fixed['evaluator']))
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    hpo=validator.read_hpo(fixed['hpo']['path'])
    validation=validator.validate_predictions(inputs,pred,hpo)
    save(directory/'validation.json',validation);assert validation['valid']
    assert sha(plan['gold_path'])==GOLD_SHA
    gold=rows(plan['gold_path'])
    assert len(gold)==16 and [str(d['pmc_id']) for d in gold]==[str(d['pmc_id']) for d in inputs]
    checked(prediction_ref)
    metric=evaluator.evaluate_report(gold,pred,'all','text')
    # Save the measured report before checks that could fail after the sole call.
    save(directory/'metrics.json',metric)
    assert metric['documents']==16 and metric['patients']==36
    assert all(metric['metrics'][k]==expected[k] for k in ('menF1','docF1'))
    assert all(math.isfinite(v) for v in metric['metrics'].values())
    delta={k:v-expected[k] for k,v in metric['metrics'].items()}
    keep=metric['metrics']['Score']>expected['Score']
    for record in records+[prediction_ref,freeze]:checked(record)
    assert ref(a.plan)==plan_ref
    summary=dict(status=QUALITY_COMPLETE,prediction=prediction_ref,metrics=metric['metrics'],
        metrics_file=ref(directory/'metrics.json'),validation=ref(directory/'validation.json'),
        baseline_metrics=expected,baseline_prediction=fixed['baseline_prediction'],
        baseline_cached_metrics=fixed['baseline_metrics'],delta=delta,keep=keep,candidate_best=keep,
        retention='Only complete four-FLA pipeline Score strictly above901; macro disclosed, no extra macro gate; otherwise close exact config.',
        new_v2_calls=1,old_v2_calls=0,freeze=freeze,assembly=ref(directory/'assembly.json'),
        no_subset_combination_scan=True,originaldev_repeated_development_not_blind=True,
        no_fit_or_model_calls=True,no_official_actions=True,hardware_disclosure=coverage['hardware_disclosure'],
        meets_numeric_0_8=metric['metrics']['Score']>=.8,accepted_as_best_pending_independent_review=False,
        elapsed_wall_seconds=time.monotonic()-started)
    save(directory/'summary.json',summary)
    print(json.dumps(dict(summary=ref(directory/'summary.json'),keep=keep,Score=metric['metrics']['Score'])))


if __name__=='__main__':main()
