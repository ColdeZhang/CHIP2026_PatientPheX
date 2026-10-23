#!/usr/bin/env python3
"""Freeze one full64 adapter prediction, then explicitly evaluate only that arm.

Both phases fail closed on missing training/transfer evidence. Import is inert;
`freeze` never opens gold. `evaluate` requires the immutable freeze and uses the
three old metrics as recorded, without re-evaluating any old predictions.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROPOSAL_SHA = 'edf29025dc054b64842d5112e7e23f9fec0d3c2e771b6439dbe160034336e571'
PY = '/workspace/patientphex-runs/day1-20261023/sequence-crf/.venv/bin/python'
GOLD = Path('/workspace/patientphex-data/day1/dev16.gold.jsonl')
SCHEMA = 'full64-dev-fixed-arm-v1'
ADAPTER_NAMES = ('adapter_config.json', 'adapter_model.safetensors')
BASE_PARAMETERS = 8953803264
ADAPTER_PARAMETERS = 43278336


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def file_ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def check_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'}, 'Malformed file reference')
    require(isinstance(ref['path'], str) and Path(ref['path']).is_absolute(), 'Need absolute local evidence path')
    require(type(ref['bytes']) is int and ref['bytes'] > 0 and valid_sha(ref['sha256']), 'Unbound size/SHA')
    require(file_ref(ref['path']) == ref, 'Evidence bytes changed: ' + ref['path'])
    return Path(ref['path'])


def write_new(path, value):
    # Exclusive create makes accidental repeated scoring or frozen-output overwrite fail.
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_adapter_files(files):
    require(isinstance(files, dict) and set(files) == set(ADAPTER_NAMES), 'Require exactly two bound adapter files')
    for name in ADAPTER_NAMES:
        item = files[name]
        require(isinstance(item, dict) and set(item) == {'bytes', 'sha256'}, 'Malformed adapter binding')
        require(type(item['bytes']) is int and item['bytes'] > 0 and valid_sha(item['sha256']), 'Unbound adapter: ' + name)


def validate_training(complete, checkpoint, actual, manifest, data, training_plan, files, transfer, model_files):
    validate_adapter_files(files)
    formal = training_plan['formal']
    require(formal['rows'] == 265 and formal['documents'] == 64 and formal['updates'] == 68, 'Wrong frozen training plan')
    require(formal['fresh_base'] is True and formal['continue_probe_or_48_adapter'] is False, 'Fresh full64 required')
    require(complete['smoke_only'] is False and complete['epochs'] == 2 and complete['global_step'] == 68, 'Training not complete epoch002/68')
    require(complete['source_sha256'] == training_plan['source_sha256']['train_qwen35_full64.py'], 'Wrong trainer source')
    require(complete['base_parameters'] == BASE_PARAMETERS and complete['trainable_adapter_parameters'] == ADAPTER_PARAMETERS, 'Wrong trained parameter count')
    require(complete['total_parameters'] == BASE_PARAMETERS + ADAPTER_PARAMETERS, 'Wrong total parameters')
    require(checkpoint['smoke_only'] is False and checkpoint['completed_epochs'] == 2 and checkpoint['global_step'] == 68, 'Wrong checkpoint epoch/step')
    fp = checkpoint['fingerprint']
    require(fp == manifest['fingerprint'], 'Checkpoint/run manifest fingerprint mismatch')
    require(fp['schema'] == 'patientphex-qwen35-full64-relation-lora-v1', 'Not full64 checkpoint schema')
    require(fp['source_sha256'] == complete['source_sha256'], 'Checkpoint trainer mismatch')
    require(fp['train_sha256'] == training_plan['data_sha256']['train.jsonl'] == data['training_sha256'], 'Wrong 265-row training bytes')
    require(fp['full64_preparation_sha256'] == training_plan['data_sha256']['preparation-manifest.json'], 'Wrong full64 provenance')
    require(data['training_rows'] == 265 and data['training_documents'] == 64, 'Wrong training row/document count')
    require(data['fresh_base_required'] is True and data['old_adapter_continuation_forbidden'] is True, 'Data does not require fresh training')
    require(data['all_target_documents_excluded_from_own_generators'] is True, 'Candidate provenance exclusion missing')
    require(fp['training_document_ids'] == data['training_document_ids'] and len(set(fp['training_document_ids'])) == 64, 'Wrong train64 document IDs')
    require(fp['excluded_document_ids'] == data['excluded_document_ids'] and len(set(fp['excluded_document_ids'])) == 16, 'Wrong excluded16 IDs')
    require(not set(fp['training_document_ids']) & set(fp['excluded_document_ids']), 'Training/dev overlap')
    require(fp['excluded_input_sha256'] == training_plan['excluded_sha256'] == data['excluded_dev_input_sha256'], 'Wrong excluded input')
    require(fp['base_files']['all15_files_verified'] is True and fp['base_files']['actual_files'] == model_files, 'Training base bytes unverified')
    require(fp['base_files']['source_manifest_sha256'] == training_plan['source_manifest_sha256'], 'Training model manifest mismatch')
    expected_manifest = dict(examples=265, max_tokens=6153, total_tokens=864937, supervised_tokens=15410,
                             steps_per_epoch=34, epochs=2, total_optimizer_steps=68, warmup_steps=4,
                             smoke_only=False, fresh_base_on_initial_run=True, validation_labels_used=False)
    for name, expected in expected_manifest.items():
        require(manifest.get(name) == expected, 'Training manifest mismatch: ' + name)
    fixed = fp['configuration']['fixed']
    for name, expected in dict(rank=16, alpha=32, dropout=.05, epochs=2, max_seq_length=6400,
                               batch_size=1, gradient_accumulation=8, seed=20261023,
                               learning_rate=1e-4, quantization=False, packing=False).items():
        require(fixed.get(name) == expected, 'Training recipe mismatch: ' + name)
    require(set(fixed['targets']) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
                                     'in_proj_b', 'in_proj_a', 'out_proj', 'gate_proj', 'up_proj', 'down_proj'}, 'Wrong LoRA targets')
    measured = actual['epoch-002']
    require(measured['completed_epochs'] == 2 and measured['global_step'] == 68, 'Actual hashes are not epoch002/68')
    require(transfer['status'] == 'complete_full_sha' and transfer['source_epoch'] == 2 and transfer['source_training_global_step'] == 68, 'Incomplete adapter transfer')
    require(transfer['source_files'] == files == transfer['destination_files'], 'Source/destination adapter copy SHA mismatch')
    for name in ADAPTER_NAMES:
        require(checkpoint['files']['adapter/' + name] == measured['files']['adapter/' + name] == files[name]['sha256'], 'Checkpoint/actual/transfer adapter mismatch: ' + name)


def validate_fingerprint(manifest, checkpoint_fp, expected, old_fp, files):
    fp = manifest['fingerprint']
    require(fp == expected == checkpoint_fp, 'Actual/checkpoint/prelaunch fingerprint mismatch')
    for name in ['source_sha256', 'prompts_sha256', 'source_manifest_sha256', 'model_files', 'runtime', 'precision', 'batch_size', 'thinking']:
        require(fp[name] == old_fp[name], 'Changed same-A30 comparison field: ' + name)
    require(fp['adapter_files'] == {k: files[k]['sha256'] for k in ADAPTER_NAMES}, 'Wrong new adapter inference SHA')
    require(fp['adapter_files']['adapter_model.safetensors'] != old_fp['adapter_files']['adapter_model.safetensors'], 'Old48 adapter reused')
    config = fp['configuration']
    require(set(config) == set(old_fp['configuration']), 'Runner argument schema changed')
    for name in ['model', 'max_input_tokens', 'max_new_tokens']:
        require(config[name] == old_fp['configuration'][name], 'Changed model/decoder: ' + name)
    require(isinstance(config['adapter'], str) and config['adapter'] and config['adapter'] != old_fp['configuration']['adapter'], 'Need distinct new adapter directory')
    require(Path(config['output']).name == 'full64-adapter.responses.jsonl', 'Unexpected new raw output')
    require(manifest['base_parameters'] == BASE_PARAMETERS and manifest['total_parameters'] == BASE_PARAMETERS + ADAPTER_PARAMETERS, 'Wrong inference parameter count')
    require(manifest['expected_rows'] == 64 and manifest['eos_token_ids'] == [248044, 248046] and manifest['pad_token_id'] == 248044, 'Changed EOS/coverage')
    loading = manifest['load_information']
    require(not loading.get('missing_keys') and not loading.get('mismatched_keys') and not loading.get('error_msgs'), 'Incomplete model loading')
    require(all(k.startswith(('mtp.', 'model.mtp.', 'model.visual.', 'visual.')) for k in loading.get('unexpected_keys', [])), 'Unexpected model keys')


def audit_responses(prompts, responses, token_rows, parse_json):
    known = {p['row_id']: p for p in prompts}
    require(len(prompts) == len(known) and len(responses) == len(known), 'Incomplete response coverage')
    require(len({r['row_id'] for r in responses}) == len(responses), 'Duplicate response row')
    require({r['row_id']: str(r['pmc_id']) for r in responses} == {k: str(p['pmc_id']) for k, p in known.items()}, 'Response identity mismatch')
    require({r['row_id']: r['input_tokens'] for r in responses} == token_rows, 'Per-row tokenizer mismatch')
    counts = collections.Counter()
    reports = []
    for response in responses:
        require(isinstance(response['response'], str), 'Response is not text')
        require(response['removed_demonstrations'] == 0, 'Input removed demonstrations')
        require(type(response['hit_token_limit']) is bool, 'Token-limit status not boolean')
        require(type(response['output_tokens']) is int and 0 < response['output_tokens'] <= 1600, 'Invalid generated-token count')
        prompt = known[response['row_id']]
        parsed = None if response['hit_token_limit'] else parse_json(response['response'])
        try:
            strict = isinstance(json.loads(response['response']), dict)
        except (ValueError, TypeError):
            strict = False
        patients = set(prompt['patient_ids'])
        valid, bad = [], []
        for code in prompt['code_map']:
            value = parsed.get(code) if isinstance(parsed, dict) else None
            (valid if isinstance(value, list) and all(str(pid) in patients for pid in value) else bad).append(code)
        counts.update(dict(rows=1, strict_json_object_rows=int(strict), hit_token_limit_rows=int(response['hit_token_limit']),
                           valid_candidate_decisions=len(valid), unresolved_candidate_decisions=len(bad),
                           rows_with_fallback=int(bool(bad)), output_tokens=response['output_tokens']))
        reports.append(dict(row_id=response['row_id'], valid_candidate_decisions=len(valid), fallback_codes=bad,
                            extra_codes=sorted(set(parsed) - set(prompt['code_map'])) if isinstance(parsed, dict) else [],
                            strict_json_object=strict, hit_token_limit=response['hit_token_limit']))
    return dict(counts=dict(counts), rows=reports)


def validate_prediction(predictions, inputs, baseline):
    require([str(x['pmc_id']) for x in predictions] == [str(x['pmc_id']) for x in inputs], 'Prediction document order/coverage mismatch')
    base = {str(d['pmc_id']): d for d in baseline}
    for prediction, item in zip(predictions, inputs):
        require(prediction['entities'] == base[str(item['pmc_id'])]['entities'], 'Changed fixed entity array')
        require(prediction.get('pmid') == item.get('pmid'), 'Changed PMID')
        require([str(p['patient_id']) for p in prediction['association']] == [str(p['patient_id']) for p in item['patient']], 'Changed/duplicate patient IDs')


def load_parser(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('fixed_full64_merge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_json


def preflight(directory):
    proposal_path = ROOT / 'proposal.json'
    require(sha(proposal_path) == PROPOSAL_SHA, 'Fixed proposal changed')
    proposal = read(proposal_path)
    references = dict(proposal['references'])
    # All proposal references are input, predictions, source or prior evidence;
    # none points at originaldev gold. Do not add gold to this map.
    for ref in references.values():
        check_ref(ref)
    old_summary = read(references['old_summary']['path'])
    old_freeze = read(references['old_prediction_freeze']['path'])
    for arm, control in proposal['frozen_controls'].items():
        for key, ref in control.items():
            if isinstance(ref, dict) and 'sha256' in ref:
                check_ref(ref)
                references['old_' + arm + '_' + key] = ref
        prior = old_summary['arms'][arm]
        metric = read(control['metrics']['path'])
        require(prior['metrics_sha256'] == control['metrics']['sha256'], 'Old metric SHA differs from frozen summary')
        require(prior['prediction_sha256'] == control['prediction']['sha256'], 'Old prediction differs from frozen summary')
        require(metric['metrics'] == prior['metrics'] and metric['counts'] == prior['counts'], 'Old metrics no longer agree')
        if arm != 'full_lr':
            require(old_freeze['arms'][arm]['prediction_sha256'] == prior['prediction_sha256'], 'Old prediction freeze mismatch')
            require(old_freeze['arms'][arm]['raw_response_sha256'] == control['raw']['sha256'], 'Old raw freeze mismatch')
        else:
            require(old_freeze['baseline_prediction_sha256'] == prior['prediction_sha256'], 'Old fullLR freeze mismatch')
    plan_path = directory / 'run-plan.json'
    plan = read(plan_path)
    require(plan['status'] == 'frozen_single_new_adapter_arm_no_gold', 'Runplan is not frozen for this one arm')
    require(plan['execution_cap_seconds'] == 900 and plan['kill_grace_seconds'] == 30 and plan['expected_rows'] == 64, 'Changed execution contract')
    require(plan['proposal_sha256'] == PROPOSAL_SHA, 'Wrong authorized proposal')
    require(plan['old_control_prediction_freeze_sha256'] == references['old_prediction_freeze']['sha256'], 'Old freeze not bound by runplan')
    require(plan['old_control_summary_sha256'] == references['old_summary']['sha256'], 'Old summary not bound by runplan')
    for filename, key in [('run_fixed_arm.py', 'driver_sha256'), ('gpu-handoff.json', 'gpu_handoff_sha256'),
                          ('source-runtime-preflight.json', 'source_runtime_preflight_sha256')]:
        path = directory / filename
        require(valid_sha(plan.get(key)) and sha(path) == plan[key], 'Changed prelaunch evidence: ' + filename)
        references['new_' + key] = file_ref(path)
    contract = plan['evaluation_contract']
    require(contract['schema'] == SCHEMA, 'Wrong evaluator contract')
    require(contract['evaluator_source_sha256'] == sha(__file__), 'Evaluator source not bound by runplan')
    require(set(contract['training']) == {'complete', 'checkpoint', 'actual_hashes', 'run_manifest'}, 'Incomplete training evidence bindings')
    training = {}
    for name, ref in contract['training'].items():
        training[name] = read(check_ref(ref))
        references['new_training_' + name] = ref
    transfer = read(check_ref(contract['adapter_transfer']))
    references['new_adapter_transfer'] = contract['adapter_transfer']
    old_manifest = read(proposal['frozen_controls']['adapter']['runner_manifest']['path'])
    data = read(references['new_training_data_manifest']['path'])
    trainplan = read(references['new_training_frozen_plan']['path'])
    validate_training(training['complete'], training['checkpoint'], training['actual_hashes'], training['run_manifest'],
                      data, trainplan, contract['adapter_files'], transfer, old_manifest['fingerprint']['model_files'])
    complete = read(directory / 'complete.json')
    require(complete['status'] == 'complete_pending_frozen_cpu_evaluation', 'Inference not completed')
    require(complete['run_plan_sha256'] == sha(plan_path), 'Completion bound to another plan')
    require(valid_sha(complete.get('released_gpu_sha256')) and sha(directory / 'released-gpu.json') == complete['released_gpu_sha256'], 'GPU release evidence not bound')
    references['new_gpu_release'] = file_ref(directory / 'released-gpu.json')
    new_paths = {'raw': directory / 'full64-adapter.responses.jsonl',
                 'manifest': directory / 'full64-adapter.responses.manifest.json',
                 'checkpoint': directory / 'full64-adapter.responses.checkpoint.json'}
    for name, path in new_paths.items():
        require(valid_sha(complete.get(name + '_sha256')) and sha(path) == complete[name + '_sha256'], 'Incomplete or changed collected ' + name)
        references['new_' + name] = file_ref(path)
    manifest = read(new_paths['manifest'])
    require(manifest['output_sha256'] == complete['raw_sha256'], 'Raw/manifest mismatch')
    validate_fingerprint(manifest, read(new_paths['checkpoint']), contract['expected_runner_fingerprint'],
                         old_manifest['fingerprint'], contract['adapter_files'])
    inputs = rows(references['input']['path'])
    baseline = rows(references['fixed_entities_and_fallback']['path'])
    prompts = rows(references['frozen_prompts']['path'])
    require(len(inputs) == len(baseline) == 16 and sum(len(d['patient']) for d in inputs) == 36, 'Input coverage changed')
    require(sum(len(d['entities']) for d in baseline) == 955, 'Wrong fixed entity count')
    require(all('entities' not in d and 'association' not in d for d in inputs), 'Input includes annotations')
    require({str(d['pmc_id']) for d in inputs} == set(data['excluded_document_ids']), 'Training excluded docs differ from evaluated inputs')
    require(len(prompts) == len({p['row_id'] for p in prompts}) == 64 and sum(len(p['code_map']) for p in prompts) == 466, 'Changed prompt/candidate counts')
    require(sum(len(p['code_map']) * len(p['patient_ids']) for p in prompts) == 1006, 'Changed candidate patient pairs')
    require(all('answer' not in p and [m['role'] for m in p['messages']] == ['system', 'user'] for p in prompts), 'Unexpected prompt labels/roles')
    tokens = {r['row_id']: r['input_tokens'] for r in read(references['old_tokenizer_audit']['path'])['rows']}
    require(sum(tokens.values()) == 202743 and max(tokens.values()) == 5194, 'Changed input tokens')
    response_rows = rows(new_paths['raw'])
    formats = audit_responses(prompts, response_rows, tokens, load_parser(Path(references['merger']['path'])))
    require(formats['counts']['valid_candidate_decisions'] + formats['counts']['unresolved_candidate_decisions'] == 466, 'Incomplete CID accounting')
    references.update(run_plan=file_ref(plan_path), inference_complete=file_ref(directory / 'complete.json'),
                      evaluator_source=file_ref(__file__), fixed_proposal=file_ref(proposal_path))
    return dict(references=references, formats=formats, old_summary=old_summary, inputs=inputs, baseline=baseline, manifest=manifest,
                contract=contract, new_raw=new_paths['raw'])


def run_command(argv, directory, stage):
    result = subprocess.run(argv, capture_output=True, text=True)
    write_new(directory / (stage + '-command.json'), dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    require(result.returncode == 0, stage + ' failed; preserve evidence, do not automatically rerun')


def freeze_arm(directory):
    require(not (directory / 'full64-predictions-freeze.json').exists(), 'Frozen output already exists')
    prediction = directory / 'full64-adapter.pred.jsonl'
    require(not prediction.exists() and not prediction.with_suffix('.manifest.json').exists(), 'Prediction already exists; no overwrite')
    prepared = preflight(directory)
    refs = prepared['references']
    run_command([PY, refs['merger']['path'], '--input', refs['input']['path'], '--entities', refs['fixed_entities_and_fallback']['path'],
                 '--prompts', refs['frozen_prompts']['path'], '--responses', str(prepared['new_raw']), '--fallback', 'baseline', '--output', str(prediction)], directory, 'merge')
    validate_prediction(rows(prediction), prepared['inputs'], prepared['baseline'])
    merge_meta = read(prediction.with_suffix('.manifest.json'))
    require(merge_meta['code_sha256'] == refs['merger']['sha256'], 'Wrong merger code')
    for key, role in [('input', 'input'), ('entities', 'fixed_entities_and_fallback'), ('prompts', 'frozen_prompts'), ('responses', 'new_raw')]:
        require(merge_meta['files'][key] == refs[role]['sha256'], 'Merger input binding changed: ' + key)
    require(merge_meta['files']['output'] == sha(prediction), 'Merger output SHA mismatch')
    require(merge_meta['configuration']['fallback'] == 'baseline' and merge_meta['configuration']['final_entities'] is None, 'Changed merger policy')
    for key in ['valid_candidate_decisions', 'unresolved_candidate_decisions']:
        require(merge_meta['stats'].get(key, 0) == prepared['formats']['counts'][key], 'Format/merge accounting differs')
    # Rehash every frozen input/control after assembly, before permitting gold.
    for ref in refs.values():
        check_ref(ref)
    write_new(directory / 'full64-format-audit.json', prepared['formats'])
    freeze = dict(schema=SCHEMA, frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gold_read=False, evaluation_performed=False, references=refs,
                  prediction=file_ref(prediction), merge_manifest=file_ref(prediction.with_suffix('.manifest.json')),
                  format_audit=file_ref(directory / 'full64-format-audit.json'), adapter_files=prepared['contract']['adapter_files'])
    write_new(directory / 'full64-predictions-freeze.json', freeze)
    return dict(status='frozen_no_gold_access', freeze=file_ref(directory / 'full64-predictions-freeze.json'))


def evaluate_arm(directory):
    freeze_path = directory / 'full64-predictions-freeze.json'
    require(freeze_path.is_file(), 'No frozen prediction; gold access forbidden')
    require(not (directory / 'evaluation-started.json').exists(), 'Evaluation already attempted; no automatic re-score')
    freeze = read(freeze_path)
    require(freeze['schema'] == SCHEMA and freeze['gold_read'] is False and freeze['evaluation_performed'] is False, 'Invalid pre-gold freeze')
    for ref in [*freeze['references'].values(), freeze['prediction'], freeze['merge_manifest'], freeze['format_audit']]:
        check_ref(ref)
    prepared = preflight(directory)
    require(prepared['references'] == freeze['references'], 'Preflight references changed after freeze')
    require(prepared['formats'] == read(freeze['format_audit']['path']), 'Format audit changed')
    prediction = Path(freeze['prediction']['path'])
    validate_prediction(rows(prediction), prepared['inputs'], prepared['baseline'])
    for name in ['full64-adapter.metrics.json', 'full64-adapter.validation.json', 'full64-summary.json']:
        require(not (directory / name).exists(), 'Evaluation output already exists: ' + name)
    # This persisted boundary is before the first access to the actual gold file.
    write_new(directory / 'evaluation-started.json', dict(freeze=file_ref(freeze_path),
              at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), old_controls_reused=True, new_arms_evaluated=1))
    refs = prepared['references']
    validation = directory / 'full64-adapter.validation.json'
    metric_path = directory / 'full64-adapter.metrics.json'
    run_command([PY, refs['validator']['path'], '--input', refs['input']['path'], '--pred', str(prediction), '--hpo', refs['hpo']['path'], '--output', str(validation)], directory, 'validation')
    checked = read(validation)
    require(checked['valid'], 'New prediction format invalid')
    # FIRST actual gold read; historical recorded gold hash is only a binding.
    require(sha(GOLD) == prepared['old_summary']['gold_sha256_after_freeze'], 'Gold bytes differ from fixed prior evaluator input')
    run_command([PY, refs['evaluator']['path'], '--gold', str(GOLD), '--pred', str(prediction), '--body-scope', 'all', '--no-id-doc', 'text', '--output', str(metric_path)], directory, 'evaluation')
    metric = read(metric_path)
    require(metric['documents'] == 16 and metric['patients'] == 36, 'Wrong evaluated coverage')
    require(sha(prediction) == freeze['prediction']['sha256'], 'Prediction changed during evaluation')
    for arm in ['base', 'adapter', 'full_lr']:
        for name in ['menF1', 'docF1']:
            require(metric['metrics'][name] == prepared['old_summary']['arms'][arm]['metrics'][name], 'Fixed-entity score changed')
    for ref in refs.values():
        check_ref(ref)
    summary = dict(status='complete_evaluated_single_new_arm', scope='Originaldev16 public v2 all/text; fixed955entities/36patients; prior3controls reused without re-evaluation',
                   freeze=file_ref(freeze_path), new_arm=dict(metrics=metric['metrics'], counts=metric['counts'], prediction=freeze['prediction'],
                   metrics_file=file_ref(metric_path), validation_file=file_ref(validation), format=prepared['formats']['counts'],
                   runner_manifest=refs['new_manifest'], adapter_files=prepared['contract']['adapter_files']),
                   old_controls={arm: prepared['old_summary']['arms'][arm] for arm in ['base', 'adapter', 'full_lr']},
                   delta_to_old={arm: {key: metric['metrics'][key] - prepared['old_summary']['arms'][arm]['metrics'][key] for key in metric['metrics']} for arm in ['base', 'adapter', 'full_lr']},
                   official_submission=False, parameter_sweep=False, schema_projection=False,
                   limitation='Document supervision48->64 AND candidate-generator32->48 plus derived training schedule change; fixed dev comparison after prior development, not blind. Entity pool remains CRF/Sap/LR64+GLiNER48.')
    write_new(directory / 'full64-summary.json', summary)
    return dict(status=summary['status'], summary=file_ref(directory / 'full64-summary.json'), metrics=metric['metrics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    parser.add_argument('--directory', type=Path, default=ROOT)
    args = parser.parse_args()
    directory = args.directory.resolve()
    print(json.dumps(freeze_arm(directory) if args.phase == 'freeze' else evaluate_arm(directory), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
