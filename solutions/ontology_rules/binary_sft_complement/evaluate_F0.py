"""One frozen two-feature calibration / expected-patient-F1 F0 endpoint."""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import time

import numpy as np
import scipy
import sklearn
from expected_f1_math import expected_f1_by_size


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def checked(item):
    actual = ref(item['path'])
    assert actual['sha256'] == item['sha256'], item['path']
    if 'bytes' in item: assert actual['bytes'] == item['bytes']
    return Path(item['path'])


def read_rows(item, key):
    data = [json.loads(line) for line in checked(item).read_text().splitlines()]
    assert len({str(row[key]) for row in data}) == len(data)
    return data, {str(row[key]): row for row in data}


def save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def command(argv, path):
    start = time.monotonic(); result = subprocess.run(argv, capture_output=True, text=True)
    save(path, dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    seconds=time.monotonic()-start))
    assert result.returncode == 0, result.stderr


def on_timeout(signum, frame):
    raise TimeoutError('Fixed assembly/evaluation 30-second budget exceeded')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); a = ap.parse_args()
    started = time.monotonic(); plan_path = Path(a.plan).resolve(); plan = json.loads(plan_path.read_text())
    assert plan['CPU_cap_seconds'] == 30 and plan['features'] == ['binary_logit', 'old_sft_positive']
    assert ref(__file__)['sha256'] == plan['source_sha256']
    assert ref(Path(__file__).with_name('expected_f1_math.py')) == plan['expected_math']
    root = Path(plan['output_dir']); assert not root.exists()
    signal.signal(signal.SIGALRM, on_timeout); signal.setitimer(signal.ITIMER_REAL, 30)
    # Explicit whitelist only: gold is not even hashed until the complete prediction freeze.
    refs = plan['before_gold']
    for item in refs.values(): checked(item)
    fit = json.loads(Path(refs['fit_freeze']['path']).read_text())
    assert fit['features'] == plan['features'] and fit['model'] == refs['model']
    assert fit['no_F0_labels'] and fit['no_training_quality_evaluated']
    for name, version in [('sklearn', sklearn.__version__), ('numpy', np.__version__), ('scipy', scipy.__version__)]:
        assert fit['runtime'][name] == version
    assert fit['runtime']['python'] == sys.version.split()[0]
    with Path(refs['model']['path']).open('rb') as stream: saved = pickle.load(stream)
    model = saved['model']
    assert saved['features'] == plan['features'] and saved['plan_sha256'] == fit['plan']['sha256']
    assert np.array_equal(model.coef_, np.array(fit['coefficient']))
    assert np.array_equal(model.intercept_, np.array(fit['intercept']))
    assert model.classes_.tolist() == [0, 1]
    inference_complete = json.loads(Path(refs['inference_complete']['path']).read_text())
    for key, value in plan['inference_complete_contract'].items(): assert inference_complete[key] == value
    assert inference_complete[plan['inference_probability_sha_field']] == refs['probabilities']['sha256']
    assert inference_complete['manifest_sha256'] == refs['inference_manifest']['sha256']
    assert inference_complete['plan_sha256'] == refs['inference_remote_plan']['sha256']
    inference_plan = json.loads(Path(refs['inference_run_plan']['path']).read_text())
    assert inference_plan['references']['checkpoint']['sha256'] == plan['same_binary32_checkpoint_sha256']
    assert inference_plan['references']['training-complete']['sha256'] == plan['same_binary32_training_complete_sha256']
    assert inference_plan['protocol']['only_current_binary32_epoch002'] and inference_plan['protocol']['no_F0_gold']
    inference_freeze = json.loads(Path(refs['inference_scores_freeze']['path']).read_text())
    assert inference_freeze['status'] == 'all58_1166_F0_probabilities_frozen_before_any_new_quality'
    prompts, _ = read_rows(refs['prompts'], 'row_id')
    _, probs = read_rows(refs['probabilities'], 'row_id')
    _, old_raw = read_rows(refs['old_raw'], 'row_id')
    old_rows, old_pred = read_rows(refs['old_prediction'], 'pmc_id')
    inputs, input_by = read_rows(refs['input'], 'pmc_id')
    assert len(prompts) == len(probs) == len(old_raw) == 58
    assert {row['row_id'] for row in prompts} == set(probs) == set(old_raw)
    assert set(input_by) == set(old_pred) == set(fit['excluded_F0_document_ids'])
    assert len(old_rows) == len(inputs) == 16 and not set(old_pred) & set(fit['training_document_ids'])
    assert [str(d['pmc_id']) for d in inputs] == [str(d['pmc_id']) for d in old_rows]
    supplied = {(str(d['pmc_id']), p['patient_id']) for d in inputs for p in d['patient']}
    assert len(supplied) == 47
    old_units = {(str(d['pmc_id']), p['patient_id'], concept) for d in old_rows for p in d['association'] for concept in p['phenotype']}
    compound = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
    assert all(not compound.fullmatch(key[2]) for key in old_units)
    items = []; units = set(); hard_units = set(); CID_count = 0
    for prompt in prompts:
        row_id = prompt['row_id']; doc = str(prompt['pmc_id']); actual = probs[row_id]
        assert str(actual['pmc_id']) == doc
        keys = [[cid, pid] for cid in prompt['code_map'] for pid in prompt['patient_ids']]
        assert keys == actual['keys'] and len(keys) == len(actual['probabilities']) == len(actual['logits'])
        prior = json.loads(old_raw[row_id]['response'])
        assert prior.keys() == prompt['code_map'].keys()
        assert not old_raw[row_id].get('hit_token_limit', False)
        CID_count += len(prompt['code_map'])
        for (cid, pid), probability, logits in zip(keys, actual['probabilities'], actual['logits']):
            concept = prompt['code_map'][cid]
            assert isinstance(concept, str) and not compound.fullmatch(concept)
            unit = (doc, pid, concept); assert unit not in units; units.add(unit)
            assert unit[:2] in supplied and len(logits) == 2 and all(math.isfinite(x) for x in logits)
            difference = logits[1]-logits[0]
            assert 0 <= probability <= 1 and abs(probability-1/(1+math.exp(-difference))) < 2e-7
            assert isinstance(prior[cid], list) and len(prior[cid]) == len(set(prior[cid]))
            assert set(prior[cid]) <= set(prompt['patient_ids'])
            hard = pid in prior[cid]
            assert hard == (unit in old_units)
            if hard: hard_units.add(unit)
            items.append(dict(pmc_id=doc, row_id=row_id, cid=cid, patient_id=pid, concept=concept,
                              binary_logit=difference, old_sft_positive=hard, binary_probability=probability))
    assert len(items) == len(units) == 1166 and CID_count == 415 and old_units == hard_units
    assert len(old_units) == 276
    X = np.array([[v['binary_logit'], float(v['old_sft_positive'])] for v in items], dtype=np.float64)
    calibrated = model.predict_proba(X)[:, 1]
    assert np.isfinite(calibrated).all() and ((calibrated >= 0) & (calibrated <= 1)).all()
    groups = defaultdict(list)
    for item, probability in zip(items, calibrated):
        item['calibrated_probability'] = float(probability)
        groups[item['pmc_id'], item['patient_id']].append(item)
    assert set(groups) == supplied
    selected = {}; decisions = []
    for key, values in groups.items():
        order, expected = expected_f1_by_size([v['calibrated_probability'] for v in values])
        k = int(np.argmax(expected))  # exact ties choose smaller k; argsort inside math is stable
        selected[key] = {values[int(i)]['concept'] for i in order[:k]}
        assert len(selected[key]) == k
        decisions.append(dict(pmc_id=key[0], patient_id=key[1], candidates=len(values), chosen_k=k,
                              expected_F1_by_size=expected.tolist(), selected_indices=[int(i) for i in order[:k]]))
    new_rows = copy.deepcopy(old_rows)
    for before, after in zip(old_rows, new_rows):
        for association in after['association']:
            association['phenotype'] = sorted(selected[str(after['pmc_id']), association['patient_id']])
        assert {k:v for k,v in before.items() if k != 'association'} == {k:v for k,v in after.items() if k != 'association'}
        assert [v['patient_id'] for v in before['association']] == [v['patient_id'] for v in after['association']]
        for old, new in zip(before['association'], after['association']):
            assert {k:v for k,v in old.items() if k != 'phenotype'} == {k:v for k,v in new.items() if k != 'phenotype'}
    root.mkdir()
    for filename, data in [('calibrated-pairs.jsonl', items), ('prediction.jsonl', new_rows)]:
        with (root/filename).open('x') as stream:
            for row in data: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(root/'input-only-decisions.json', dict(patients=decisions, pairs=1166, no_gold=True))
    freeze = dict(frozen_at=datetime.now(timezone.utc).isoformat(), source=ref(__file__), plan=ref(plan_path),
                  prediction=ref(root/'prediction.jsonl'), calibrated_pairs=ref(root/'calibrated-pairs.jsonl'),
                  decisions=ref(root/'input-only-decisions.json'), baseline=refs['old_prediction'], fit_freeze=refs['fit_freeze'],
                  binary_probabilities=refs['probabilities'], no_F0_gold_read=True,
                  all_nonassociation_objects_exact=True, all_patient_and_association_metadata_exact=True,
                  fit_checkpoint_push=plan['fit_checkpoint_push'], seconds_before_gold=time.monotonic()-started)
    save(root/'prediction-freeze.json', freeze)
    save(root/'evaluation-started.json', dict(started_at=datetime.now(timezone.utc).isoformat(), freeze=ref(root/'prediction-freeze.json')))
    checked(plan['gold_after_freeze'])
    command([plan['python'], refs['validator']['path'], '--input', refs['input']['path'], '--pred', str(root/'prediction.jsonl'),
             '--hpo', refs['hpo']['path'], '--output', str(root/'validation.json')], root/'validation-command.json')
    validation = json.loads((root/'validation.json').read_text()); assert validation['valid']
    command([plan['python'], refs['evaluator']['path'], '--gold', plan['gold_after_freeze']['path'], '--pred', str(root/'prediction.jsonl'),
             '--body-scope', 'all', '--no-id-doc', 'text', '--output', str(root/'metrics.json')], root/'evaluation-command.json')
    new_metrics = json.loads((root/'metrics.json').read_text())
    baseline = json.loads(checked(plan['baseline_metrics']).read_text())
    baseline_metrics = baseline['metrics']
    assert baseline['prediction_sha256'] == refs['old_prediction']['sha256']
    assert baseline['gold_sha256'] == plan['gold_after_freeze']['sha256']
    delta = {key:new_metrics['metrics'][key]-baseline_metrics[key] for key in baseline_metrics}
    assert delta['menF1'] == delta['docF1'] == 0
    keep = delta['micF1'] + delta['macF1'] > 0 and delta['macF1'] >= 0
    summary = dict(status='complete_one_F1_calibrated_F0_expected_patient_F1_control',
                   completed_at=datetime.now(timezone.utc).isoformat(), metrics=new_metrics, baseline=baseline,
                   delta=delta, retention_criterion_pass=keep, retention='delta(micF1+macF1)>0 AND delta(macF1)>=0',
                   prediction_freeze=ref(root/'prediction-freeze.json'), fit_freeze=refs['fit_freeze'],
                   total_seconds=time.monotonic()-started, no_threshold_or_decoder_search=True, no_originaldev=True,
                   no_new_large_model_training=True, baseline_metrics_reused=True,
                   limitations=['Only F1 fits LR; F0 is previously developed validation, not blind test.',
                                'Old SFT hard and candidate generator scale from32 on F1 to48 on F0; F0 train48 includes F1 but excludes F0.',
                                'Expected patient F1 assumes independent calibrated candidate probabilities; it is not guaranteed to maximize public micro+macro.',
                                'One relation arm only; complete entities and all nonassociation fields match old SFT48.'])
    assert summary['total_seconds'] <= 30
    save(root/'summary.json', summary)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(delta=delta, retain=keep, seconds=summary['total_seconds'])))


if __name__ == '__main__': main()
