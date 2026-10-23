#!/usr/bin/env python3
"""Assemble both fixed-.5 predictions before permitting one paired evaluation."""
from pathlib import Path
from collections import defaultdict
import argparse
import datetime
import gzip
import hashlib
import json
import math
import subprocess


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(reference):
    require(ref(reference['path']) == reference, 'Frozen file changed: ' + reference['path'])
    return Path(reference['path'])


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def preflight(base):
    plan = read(base / 'evaluation-plan.json')
    require(plan['source_sha256'] == sha(__file__), 'Evaluation source drift')
    require(plan['threshold'] == .5 and plan['arms'] == ['baseline', 'bridge'], 'Changed fixed arms or threshold')
    references = plan['references']
    for reference in references.values():
        checked(reference)
    inputs = rows(references['input']['path'])
    pool = rows(references['entity_pool']['path'])
    lookup = {str(row['pmc_id']): row for row in pool}
    require(len(inputs) == len(pool) == 16 and sum(len(doc['patient']) for doc in inputs) == 47, 'Wrong heldout coverage')
    require({str(doc['pmc_id']) for doc in inputs} == set(lookup), 'Candidate document coverage drift')
    require(sum(len(row['entities']) for row in pool) == 947, 'Wrong entity pool')
    require(all('entities' not in doc and 'association' not in doc for doc in inputs), 'Input contains labels')
    with gzip.open(references['encoded_holdout']['path'], 'rt') as handle:
        encoded = json.load(handle)
    require(encoded['schema'] == 'patient-identity-encoded-holdout-v1' and encoded['label_free'], 'Not label-free encoded data')
    keys = [tuple(row['key']) for row in encoded['rows']]
    require(len(keys) == len(set(keys)) == 1166 and all('label' not in row for row in encoded['rows']), 'Changed pair set')
    complete = read(base / 'formal-complete.json')
    require(complete['status'] == 'complete_two_fresh_final_arms_no_evaluation', 'Two complete arms required')
    require(complete['paired_plan_sha256'] == references['formal_plan']['sha256'], 'Formal protocol differs')
    actual = {}
    for arm in plan['arms']:
        terminal = complete['arms'][arm]
        training = read(base / (arm + '.training-complete.json'))
        inference = read(base / (arm + '.inference-complete.json'))
        training_manifest = read(base / (arm + '.training-manifest.json'))
        require(terminal['training_complete'] == training and terminal['training_manifest'] == training_manifest and
                terminal['inference_complete'] == inference, 'Terminal chain changed')
        require(training['updates'] == 1032 and [e['epoch'] for e in training['epochs']] == [1, 2, 3, 4], 'Incomplete training')
        require(training_manifest['arm'] == arm and training_manifest['fresh_public_base'] and
                not training_manifest['old_gold_CE_or_probe_state_reused'], 'Non-fresh or wrong arm')
        require(inference['final_epoch'] == 4 and inference['training_updates'] == 1032 and inference['arm'] == arm,
                'Wrong inference checkpoint')
        require(inference['model_files'] == terminal['model_files'] and inference['no_gold_access'], 'Wrong inference weights/provenance')
        raw_path = base / (arm + '.pair-scores.jsonl')
        require(sha(raw_path) == inference['output_sha256'] and inference['encoded_sha256'] == references['encoded_holdout']['sha256'], 'Wrong raw/input SHA')
        raw = rows(raw_path)
        require([tuple(row['key']) for row in raw] == keys, 'Pair order or coverage changed')
        require(all(isinstance(row['probability'], (int, float)) and math.isfinite(row['probability']) and
                    0 <= row['probability'] <= 1 for row in raw), 'Invalid probability')
        actual[arm] = dict(raw=raw, raw_ref=ref(raw_path), training_ref=ref(base / (arm + '.training-complete.json')),
                           inference_ref=ref(base / (arm + '.inference-complete.json')),
                           training_manifest_ref=ref(base / (arm + '.training-manifest.json')))
    return plan, inputs, lookup, actual


def freeze(base):
    require(not (base / 'paired-predictions-freeze.json').exists(), 'Already frozen; no overwrite')
    plan, inputs, pool, actual = preflight(base)
    output = {}
    for arm in plan['arms']:
        selected = defaultdict(set)
        for row in actual[arm]['raw']:
            if row['probability'] >= .5:
                pmc, patient, concept = row['key']
                selected[(pmc, patient)].add(concept)
        predictions = []
        for doc in inputs:
            pmc = str(doc['pmc_id'])
            prediction = dict(pool[pmc])
            prediction['association'] = [dict(patient_id=p['patient_id'], phenotype=sorted(selected[(pmc, p['patient_id'])]))
                                         for p in doc['patient']]
            require(prediction['entities'] == pool[pmc]['entities'], 'Entity pool changed')
            predictions.append(prediction)
        path = base / (arm + '.pred.jsonl')
        with path.open('x') as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        output[arm] = dict(prediction=ref(path), **{key: value for key, value in actual[arm].items() if key.endswith('_ref')})
    for reference in plan['references'].values():
        checked(reference)
    value = dict(status='both_predictions_frozen_before_gold', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 evaluation_plan=ref(base / 'evaluation-plan.json'), formal_complete=ref(base / 'formal-complete.json'),
                 arms=output, references=plan['references'], threshold=.5, gold_read=False, evaluation_performed=False)
    save(base / 'paired-predictions-freeze.json', value)
    return dict(status=value['status'], freeze=ref(base / 'paired-predictions-freeze.json'))


def command(argv, base, name):
    result = subprocess.run(argv, capture_output=True, text=True)
    save(base / (name + '-command.json'), dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    require(result.returncode == 0, 'Command failed, preserve outputs without automatic rescore: ' + name)


def evaluate(base):
    require((base / 'paired-predictions-freeze.json').is_file(), 'Both predictions must be frozen first')
    require(not (base / 'evaluation-started.json').exists(), 'Evaluation already attempted')
    frozen = read(base / 'paired-predictions-freeze.json')
    require(frozen['gold_read'] is False and frozen['evaluation_performed'] is False and frozen['threshold'] == .5, 'Invalid freeze')
    checked(frozen['evaluation_plan'])
    checked(frozen['formal_complete'])
    for arm in ['baseline', 'bridge']:
        for reference in frozen['arms'][arm].values():
            checked(reference)
    plan, _, _, _ = preflight(base)
    require(plan['references'] == frozen['references'], 'References changed since freeze')
    save(base / 'evaluation-started.json', dict(freeze=ref(base / 'paired-predictions-freeze.json'),
         at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), new_arms=2, old_controls_re_evaluated=False))
    references = plan['references']
    # First access to actual heldout gold occurs only after both prediction refs.
    require(sha(plan['gold_path']) == plan['historical_gold_sha256'], 'Gold bytes changed')
    arms = {}
    for arm in ['baseline', 'bridge']:
        prediction = frozen['arms'][arm]['prediction']['path']
        validation = base / (arm + '.validation.json')
        metrics = base / (arm + '.metrics.json')
        command([plan['python'], references['validator']['path'], '--input', references['input']['path'], '--pred', prediction,
                 '--hpo', references['hpo']['path'], '--output', str(validation)], base, arm + '-validation')
        require(read(validation)['valid'], 'Invalid prediction format')
        command([plan['python'], references['evaluator']['path'], '--gold', plan['gold_path'], '--pred', prediction,
                 '--body-scope', 'all', '--no-id-doc', 'text', '--output', str(metrics)], base, arm + '-evaluation')
        value = read(metrics)
        require(value['documents'] == 16 and value['patients'] == 47, 'Wrong metric coverage')
        for control in ['rules', 'lr']:
            old = read(references[control + '_metrics']['path'])
            require(all(value['metrics'][name] == old['metrics'][name] for name in ['menF1', 'docF1']), 'Fixed entity metric changed')
        arms[arm] = dict(metrics=value['metrics'], counts=value['counts'], prediction=frozen['arms'][arm]['prediction'],
                         metrics_file=ref(metrics), validation_file=ref(validation))
    old_controls = {name: read(references[name + '_metrics']['path'])['metrics'] for name in ['rules', 'lr']}
    for arm in ['baseline', 'bridge']:
        checked(frozen['arms'][arm]['prediction'])
    for reference in plan['references'].values():
        checked(reference)
    summary = dict(status='complete_two_new_fixed_arms', scope='train48/outerfold0 publicv2 all/text, fixed947 entities/47patients, already used for research not blind',
                   arms=arms, old_controls_reused=old_controls, threshold=.5,
                   delta_bridge_to_baseline={key: arms['bridge']['metrics'][key] - arms['baseline']['metrics'][key] for key in arms['bridge']['metrics']},
                   delta_bridge_to_controls={name: arms['bridge']['metrics']['Score'] - old_controls[name]['Score'] for name in old_controls},
                   freeze=ref(base / 'paired-predictions-freeze.json'), originaldev_read=False, official_submission=False,
                   limitation='Both models fresh same OOF candidate distribution and512 tensors. Added raw anchor clause can include clinical facts; visibility is not proof of identity or score. No threshold/epoch search.')
    save(base / 'paired-summary.json', summary)
    return dict(status=summary['status'], summary=ref(base / 'paired-summary.json'), arms={name: data['metrics'] for name, data in arms.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.directory.resolve()) if args.phase == 'freeze' else evaluate(args.directory.resolve()), indent=2))


if __name__ == '__main__':
    main()
