#!/usr/bin/env python3
"""Collect one already-complete F0 endpoint; never run inference or evaluation.

Only local metadata and hashes are read. Gold, model/state bytes, tokenizer,
training data and prompt data are never opened. Existing full predictions/raw
are hashed in place, not copied or interpreted. This is an owner evidence index,
not a replacement for the independent terminal review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


CONTRACT = dict(rows=58, documents=16, patients=47, entities=947, candidates=415, pairs=1166)
KEEP_RULE = 'delta(micF1+macF1)>0 and delta(macF1)>=0'
CONTROL = {'driver-started.json', 'wrapper-complete.json', 'stdout.log', 'stderr.log'}
TRAIN = CONTROL | {'run_manifest.json', 'token-mask-audit.json', 'gpu-preflight.json',
    'model-coverage.json', 'base-loading-info.json', 'first-nonzero-update.json',
    'progress.json', 'training_complete.json', 'epoch-001/checkpoint.json',
    'epoch-002/checkpoint.json', 'durability-epoch-001.json', 'durability-epoch-002.json'}
INFER = CONTROL | {'responses.jsonl', 'responses.manifest.json',
    'responses.checkpoint.json', 'responses.loading.json', 'complete.json'}
ENDPOINT = ['assembled/freeze.json', 'assembled/patient-first.jsonl',
    'assembled/parsed-decisions.json', 'secondary-union/common-freeze.json',
    'secondary-union/union.jsonl', 'secondary-union/before-both-quality.json',
    'evaluation/before-gold.json', 'evaluation/validation.json', 'evaluation/metrics.json',
    'evaluation/results.json', 'secondary-union/evaluation/before-gold.json',
    'secondary-union/evaluation/validation.json', 'secondary-union/evaluation/metrics.json',
    'secondary-union/evaluation/results.json', 'secondary-union/results.json']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'Missing regular evidence file: ' + str(path))
    require('.gold.' not in path.name.lower() and path.suffix not in
            {'.pt', '.pth', '.bin', '.safetensors', '.h5', '.pkl', '.pickle'},
            'Forbidden gold/model/state read: ' + str(path))
    require(path.stat().st_size <= 2 * 1024 * 1024, 'Not small endpoint evidence: ' + str(path))
    return path


def ref(path):
    path = safe_path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(262144), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def read(path):
    path = safe_path(path)
    require(path.suffix == '.json', 'Only JSON metadata may be parsed')
    return json.loads(path.read_text())


def match(item, actual):
    require(all(item[k] == actual[k] for k in ('bytes', 'sha256')), 'Evidence SHA/size differs')
    if 'path' in item:
        require(Path(item['path']).resolve() == Path(actual['path']), 'Evidence path differs')


def when(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timestamp must include timezone')
    return result


def collect_phase(b, phase, allowed, plan, refs):
    prefix = 'training' if phase == 'train' else 'inference'
    collection_path = b / (prefix + '-collection.json')
    collection = read(collection_path)
    require(collection['phase'] == phase and collection['direct_existing_files_only'], 'Collection contract')
    require(set(collection['files']) == allowed, 'Collection incomplete or has unexpected/failure member')
    docs = {}
    for name, item in collection['files'].items():
        key = prefix + '-terminal/' + name
        actual = ref(b / key); match(item, actual); refs[key] = actual
        if name.endswith('.json'):
            docs[name] = read(b / key)
    wrapper = docs['wrapper-complete.json']
    require(wrapper == collection['wrapper'] and wrapper['returncode'] == 0
            and not wrapper.get('endpoint_error'), 'Phase must have a successful complete wrapper')
    require(wrapper['phase'] == phase and wrapper['no_retry'], 'Phase identity/retry contract')
    require(wrapper['cap_seconds'] == plan['cap_seconds'] and wrapper['grace_seconds'] == 30, 'Budget differs')
    require(wrapper['output_dir'] == plan['output_dir'], 'Output namespace differs')
    release = wrapper['gpu_release']
    require(release['target'] == plan['gpu_uuid'] and release['target_has_no_process'], 'Actual GPU release missing')
    require(docs['driver-started.json']['plan_sha256'] == refs[phase + '-plan.json']['sha256'], 'Driver plan SHA differs')
    require(when(docs['driver-started.json']['at_utc']) <= when(wrapper['at_utc'])
            <= when(release['at_utc']) <= when(collection['at_utc']), 'Phase chronology differs')
    refs[collection_path.name] = ref(collection_path)
    return collection, docs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); began = time.monotonic()
    b = args.directory.resolve(strict=True); output = args.output.resolve()
    require(output.parent.is_dir() and not output.exists(), 'Output needs existing parent and new filename')
    require(output.name == 'owner-terminal-manifest.json' and output.parent == b,
            'Write only the new owner manifest in formal-preparation')
    # Inventory first: missing/partial endpoints produce no final output/directory.
    required = ['training-collection.json', 'inference-collection.json'] + ENDPOINT
    required += ['training-terminal/' + n for n in TRAIN]
    required += ['inference-terminal/' + n for n in INFER]
    missing = [name for name in required if not (b / name).is_file()]
    require(not missing, 'Endpoint incomplete; no manifest written: ' + ', '.join(sorted(missing)))
    bindings_path = Path(__file__).with_name('source-bindings.json')
    bindings = read(bindings_path)['refs']; refs = {}
    # The bindings enumerate small source/plans only; do not recursively follow JSON refs.
    for name, expected in bindings.items():
        current = ref(expected['path']); match(expected, current)
        snapshot = ref(expected['snapshot'])
        require(snapshot['sha256'] == current['sha256'] and snapshot['bytes'] == current['bytes'], 'Snapshot differs')
        if name in {'policy.json', 'endpoint-plan.json', 'train-plan.json', 'infer-plan.json', 'resource.json'}:
            require(Path(current['path']) == b / name, 'Use the frozen original preparation directory')
        refs[name] = current
    policy = read(b/'policy.json'); plan = read(b/'endpoint-plan.json')
    train_plan = read(b/'train-plan.json'); infer_plan = read(b/'infer-plan.json')
    require(policy['contract'] == CONTRACT and policy['keep_rule'] == plan['keep_rule'] == KEEP_RULE, 'Policy differs')
    require(not policy['new_F0_quality_read'] and not policy['new_F0_responses_read'], 'Policy was not pre-quality')
    match(plan['policy'], refs['policy.json']); match(plan['secondary_source'], refs['secondary_union.py'])
    require(plan['source_sha256'] == refs['finish.py']['sha256'], 'Primary source differs')
    require(train_plan['cap_seconds'] == 3900 and infer_plan['cap_seconds'] == 600, 'Frozen budgets differ')
    require(train_plan['epochs'] == 2 and train_plan['optimizer_steps'] == 52, 'Training accounting differs')
    for arm_plan, source_name in ((train_plan, 'trainer.py'), (infer_plan, 'infer.py')):
        item = arm_plan['files'][arm_plan['argv'][2]]
        require(item['sha256'] == refs[source_name]['sha256'] and item['bytes'] == refs[source_name]['bytes'], 'Actual phase source differs')
    tc, t = collect_phase(b, 'train', TRAIN, train_plan, refs)
    ic, i = collect_phase(b, 'infer', INFER, infer_plan, refs)
    require(when(tc['at_utc']) <= when(i['driver-started.json']['at_utc']), 'Inference preceded durable training collection')
    tw = t['wrapper-complete.json']; training = t['training_complete.json']
    require(tw['all_epochs_persistent_and_readback_verified'] and tw['no_SHM_or_pending_export'], 'Durable full state missing')
    require(training['epochs'] == 2 and training['global_step'] == 52 and not training['smoke_only'], 'Incomplete training')
    require(training['source_sha256'] == refs['trainer.py']['sha256'], 'Training source differs')
    require(tw['training_complete_sha256'] == refs['training-terminal/training_complete.json']['sha256'], 'Training complete SHA differs')
    manifest = t['run_manifest.json']
    require((manifest['examples'], manifest['epochs'], manifest['total_optimizer_steps'], manifest['warmup_steps'])
            == (204, 2, 52, 3), 'Actual run accounting differs')
    durable = {}
    for epoch in (1, 2):
        name = 'durability-epoch-%03d.json' % epoch; receipt = t[name]
        metadata_name = 'epoch-%03d/checkpoint.json' % epoch; metadata = t[metadata_name]
        require(receipt == tw['durability'][str(epoch)]['receipt'], 'Wrapper/durability receipt differs')
        require(tw['durability'][str(epoch)]['receipt_sha256'] == refs['training-terminal/'+name]['sha256'], 'Durability SHA differs')
        require(receipt['completed_epochs'] == epoch and receipt['global_step'] == 26 * epoch
                and metadata['completed_epochs'] == epoch and metadata['global_step'] == 26 * epoch, 'Epoch accounting differs')
        require(receipt['file_directory_fsync'] and receipt['reopened_SHA_all_exact'], 'Durability actual verification missing')
        require(receipt['source_sha256'] == training['source_sha256'], 'Durability trainer differs')
        require(receipt['checkpoint'] == train_plan['output_dir'] + '/epoch-%03d' % epoch, 'Checkpoint namespace differs')
        require(receipt['persistent_root'] == train_plan['storage_parent'], 'Persistence root differs')
        require(receipt['ancestor_directories_fsynced'][0] == train_plan['output_dir']
                and receipt['ancestor_directories_fsynced'][-1] == train_plan['storage_parent'], 'Ancestor fsync evidence missing')
        expected_metadata = refs['training-terminal/'+metadata_name]['sha256']
        require(receipt['checkpoint_metadata_sha256'] == expected_metadata, 'Checkpoint metadata SHA differs')
        require(set(receipt['files']) == set(metadata['files']) | {'checkpoint.json'}, 'Durable member inventory differs')
        require({'training_state.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json'} <= set(metadata['files']), 'Full state inventory missing')
        for name, item in receipt['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe member name')
            expected = expected_metadata if name == 'checkpoint.json' else metadata['files'][name]
            require(item['sha256'] == expected and re.fullmatch('[0-9a-f]{64}', expected)
                    and isinstance(item['bytes'], int) and item['bytes'] > 0, 'Durable member digest differs')
        durable[str(epoch)] = dict(receipt=refs['training-terminal/durability-epoch-%03d.json' % epoch],
            checkpoint=receipt['checkpoint'], files=receipt['files'],
            verification='Copied from actual fsync/reopen receipt and checked against metadata; no model/state bytes reopened here')
    complete = i['complete.json']; inference = i['responses.manifest.json']; fingerprint = inference['fingerprint']
    require(complete['rows'] == inference['expected_rows'] == 58 and complete['complete_coverage']
            and complete['no_quality'] and inference['strict_all58_EOS_and_PID_CID_schema']
            and inference['fallback_rows'] == 0, 'Incomplete/invalid final58 inference')
    require(i['wrapper-complete.json']['inference_complete_sha256'] == refs['inference-terminal/complete.json']['sha256'], 'Inference complete SHA differs')
    require(complete['responses_sha256'] == inference['output_sha256'] == refs['inference-terminal/responses.jsonl']['sha256'], 'Raw58 SHA differs')
    require(complete['manifest_sha256'] == refs['inference-terminal/responses.manifest.json']['sha256']
            and complete['inference_plan_sha256'] == refs['infer-plan.json']['sha256'], 'Inference manifest/plan differs')
    require(fingerprint == i['responses.checkpoint.json'], 'Inference fingerprint differs')
    require(fingerprint['source_sha256'] == refs['infer.py']['sha256']
            and fingerprint['training_complete_sha256'] == refs['training-terminal/training_complete.json']['sha256']
            and fingerprint['training_checkpoint_sha256'] == refs['training-terminal/epoch-002/checkpoint.json']['sha256'], 'Final inference training identity differs')
    for name, value in fingerprint['adapter_files'].items():
        require(value == durable['2']['files']['adapter/'+name]['sha256'], 'Inference adapter differs from durable epoch2')
    for name in ENDPOINT:
        refs[name] = ref(b/name)
    primary = read(b/'assembled/freeze.json'); common = read(b/'secondary-union/common-freeze.json')
    before = read(b/'secondary-union/before-both-quality.json')
    require(not primary['gold_read'] and primary['all58_strict'] and primary['all_nonassociation_exact']
            and primary['no_baseline_fallback'], 'Primary full freeze incomplete')
    require((primary['entities'], primary['candidates'], primary['pairs']) == (947, 415, 1166), 'Primary pool differs')
    require(not common['gold_read'] and common['new_quality_calls_before_freeze'] == 0, 'Common freeze is not before quality')
    require((common['rows'], common['documents'], common['entities'], common['candidates'], common['pairs'])
            == (58, 16, 947, 415, 1166), 'Common pool differs')
    require(common['contract']['pool_pairs'] == 1166 and common['contract']['removed_from_old'] == 0
            and common['contract']['all_nonassociation_exact'] and common['contract']['all_patient_metadata_exact'], 'Union contract differs')
    primary_links = dict(plan='endpoint-plan.json', policy='policy.json', source='finish.py',
        prediction='assembled/patient-first.jsonl', responses='inference-terminal/responses.jsonl',
        parsed='assembled/parsed-decisions.json', inference_manifest='inference-terminal/responses.manifest.json')
    common_links = dict(plan='endpoint-plan.json', policy='policy.json', source='secondary_union.py',
        primary_source='finish.py', primary_freeze='assembled/freeze.json', primary='assembled/patient-first.jsonl',
        secondary='secondary-union/union.jsonl', responses='inference-terminal/responses.jsonl',
        parsed='assembled/parsed-decisions.json', inference_manifest='inference-terminal/responses.manifest.json')
    for key, name in primary_links.items(): match(primary[key], refs[name])
    for key, name in common_links.items(): match(common[key], refs[name])
    for key, plan_key in [('baseline_prediction', 'baseline_pred'), ('baseline_metrics', 'baseline_metrics')]:
        require(common[key] == plan['references'][plan_key], 'Old baseline identity differs')
    # Only old metrics are opened, never any gold reference from plan/results.
    refs['baseline_metrics'] = ref(plan['references']['baseline_metrics']['path'])
    match(plan['references']['baseline_metrics'], refs['baseline_metrics'])
    baseline = read(refs['baseline_metrics']['path'])['metrics']
    for key, name in [('common_freeze', 'secondary-union/common-freeze.json'),
                      ('policy', 'policy.json'), ('plan', 'endpoint-plan.json')]: match(before[key], refs[name])
    require(before['gold_not_yet_read'] and before['exactly_two_new_arms'] and before['old_baseline_reused'], 'Before-both marker invalid')
    pbefore = read(b/'evaluation/before-gold.json'); sbefore = read(b/'secondary-union/evaluation/before-gold.json')
    for key, name in [('freeze', 'assembled/freeze.json'), ('common_freeze', 'secondary-union/common-freeze.json'),
                      ('before_both_quality', 'secondary-union/before-both-quality.json')]: match(pbefore[key], refs[name])
    match(sbefore['common_freeze'], refs['secondary-union/common-freeze.json'])
    require(pbefore['gold_not_yet_read'] and sbefore['secondary_gold_not_yet_read'], 'Before-gold marker invalid')
    both = read(b/'secondary-union/results.json')
    require(both['new_evaluator_calls'] == 2 and both['baseline_evaluator_calls'] == 0, 'Quality call accounting differs')
    match(both['common_freeze'], refs['secondary-union/common-freeze.json']); match(both['policy'], refs['policy.json'])
    results = {}
    for arm, directory in [('primary', 'evaluation'), ('secondary', 'secondary-union/evaluation')]:
        result = read(b/directory/'results.json'); metrics = read(b/directory/'metrics.json')
        validation = read(b/directory/'validation.json')
        require(validation['valid'], 'Format invalid: ' + arm)
        require(metrics['evaluator_version'] == 'patientphex-public-approx-v2'
                and metrics['configuration']['body_scope'] == 'all'
                and metrics['configuration']['unlinked_document_mode'] == 'text'
                and metrics['documents'] == 16 and metrics['patients'] == 47, 'Stored evaluator contract differs')
        require(result == both[arm] and result['metrics'] == metrics['metrics'], 'Stored results differ: ' + arm)
        require(result['baseline_reused_not_rescored'] and result['baseline'] == baseline
                and result['keep_rule'] == KEEP_RULE, 'Baseline/keep policy differs')
        require(result['delta']['menF1'] == result['delta']['docF1'] == 0, 'Entity scores changed')
        results[arm] = dict(result=result, metrics=refs[directory+'/metrics.json'],
                           validation=refs[directory+'/validation.json'], results=refs[directory+'/results.json'])
    events = [('policy', policy['frozen_at_utc']), ('train_start', t['driver-started.json']['at_utc']),
        ('epoch1_durable', t['durability-epoch-001.json']['at_utc']),
        ('epoch2_durable', t['durability-epoch-002.json']['at_utc']),
        ('train_collection', tc['at_utc']), ('infer_start', i['driver-started.json']['at_utc']),
        ('infer_complete', complete['at_utc']), ('infer_release', i['wrapper-complete.json']['gpu_release']['at_utc']),
        ('infer_collection', ic['at_utc']), ('primary_freeze', primary['frozen_at_utc']),
        ('common_freeze', common['frozen_at_utc']), ('before_both_quality', before['at_utc']),
        ('primary_before_gold', pbefore['at_utc']), ('primary_complete', results['primary']['result']['completed_at_utc']),
        ('secondary_before_gold', sbefore['at_utc']), ('secondary_complete', results['secondary']['result']['completed_at_utc'])]
    require(all(when(left[1]) <= when(right[1]) for left, right in zip(events, events[1:])), 'Freeze/quality chronology is not ordered')
    report = dict(status='complete_owner_evidence_index_pending_independent_terminal_review',
        at_utc=datetime.now(timezone.utc).isoformat(), source=ref(__file__), source_bindings=ref(bindings_path),
        contract=CONTRACT, references=refs, chronology=[dict(event=n, at_utc=v) for n,v in events],
        persistent_checkpoints=durable, results=results,
        training=dict(actual_seconds=training['elapsed_seconds'], wrapper_seconds=tw['elapsed_seconds'],
            epochs=2, optimizer_calls=52, max_gpu_allocated_bytes=training['max_gpu_allocated_bytes'],
            max_gpu_reserved_bytes=training['max_gpu_reserved_bytes'], gpu_release=tw['gpu_release']),
        inference=dict(actual_seconds=inference['elapsed_seconds'], wrapper_seconds=i['wrapper-complete.json']['elapsed_seconds'],
            rows=58, max_gpu_allocated_bytes=inference['max_gpu_allocated_bytes'], gpu_release=i['wrapper-complete.json']['gpu_release']),
        limits=dict(gold_opened_or_hashed=False, model_or_training_state_opened=False,
            evaluator_calls=0, tokenizer_or_schema_tests_rerun=False, MCP_GPU_or_network_calls=False,
            new_quality_calls_reported_by_frozen_endpoint=2, old_baseline_calls_reported=0,
            chronology='Existing metadata times and exact hash links, not independent proof of first gold access',
            artifact_retention='Raw/fullpred/logs/weights stay outside Git; this script copies no artifact bytes'),
        collection_seconds=time.monotonic()-began)
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    temporary = output.with_name(output.name + '.pending')
    with temporary.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, output)  # Exclusive atomic publication; no overwrite.
    temporary.unlink()
    descriptor = os.open(str(output.parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    print(json.dumps(ref(output)))


if __name__ == '__main__':
    main()
