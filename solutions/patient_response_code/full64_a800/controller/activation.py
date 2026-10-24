"""Pure actual-evidence gate; a template or old C32 result cannot enable full64."""
from pathlib import Path
import hashlib, json, math


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def check(record):
    actual = ref(record['path'])
    if actual != record:
        raise ValueError('Evidence identity changed: ' + record['path'])
    return json.loads(Path(record['path']).read_text())


def check_probe_host(training, probe_plan):
    """The old A100 result cannot activate the migrated A800 campaign."""
    host = training['resource_migration']
    if (host['mount_id'] != '69b991ee-183f-4add-af1a-579af92479c9'
        or probe_plan['gpu_uuid'] != host['gpu_uuid']
        or probe_plan['python'] != training['python']
        or probe_plan['model'] != training['model']
        or probe_plan['environment']['PYTHONPATH'] != training['pythonpath']
        or probe_plan['environment']['CUDA_VISIBLE_DEVICES'] != host['gpu_uuid']
        or probe_plan['resource_migration']['actual_torch_expected'] != training['formal_runtime']['torch']
        or probe_plan['actual_native_CPU'] != training['data_arguments']['native_receipt']):
        raise ValueError('New A800 probe host/native/runtime identity differs')
    inventory = check(host['actual_inventory'])
    if (probe_plan['resource_migration']['actual_A800_inventory'] != host['actual_inventory']
        or host['source_private_IP'] not in inventory['host_addresses']
        or inventory['model'] != training['model'] or inventory['python'] != training['python']):
        raise ValueError('New A800 inventory binding differs')
    return True


def require_actual_probe(training, wrapper_path, decision_path):
    # Caller invokes this before creating a deployable controller directory or
    # GPU authorization. Missing actual files fail without a fake placeholder.
    wrapper = json.loads(Path(wrapper_path).read_text())
    decision = json.loads(Path(decision_path).read_text())
    if (wrapper['returncode'] != 0 or wrapper['timed_out']
        or not wrapper['all12_backwards_and_one_update_verified'] or not wrapper['target_has_no_process']):
        raise ValueError('New full64 actual pressure result did not pass/release')
    if decision['probe_wrapper'] != ref(wrapper_path):
        raise ValueError('Cost decision does not bind actual wrapper')
    step = check(decision['probe_step']); done = check(decision['probe_complete'])
    probe_plan = check(training['expected_probe_plan'])
    check_probe_host(training, probe_plan)
    if (step['actual_forwards'] != 12 or step['actual_backwards'] != 12 or step['actual_optimizer_steps'] != 1
        or not step['max_supervision128_and_longest6462_completed']
        or not step['final_inventory']['all_grad_finite'] or not step['final_inventory']['frozen_base_no_gradient']
        or done['step_sha256'] != decision['probe_step']['sha256'] or not done['fresh_state_discarded']):
        raise ValueError('Full64 H51 evidence differs, incomplete or probe state reused')
    # Exact probe source is checked in its immutable runtime plan and observed
    # manifest; the owner cost decision binds that same plan explicitly.
    if decision['probe_plan'] != training['expected_probe_plan']:
        raise ValueError('Cost decision used another probe plan')
    manifest_path = Path(decision['probe_step']['path']).parent / 'run-manifest.json'
    manifest_ref = ref(manifest_path)
    observed = json.loads(manifest_path.read_text())
    runtime = observed['runtime']
    if (any(runtime[name] != training['formal_runtime'][name] for name in ['torch', 'transformers', 'peft'])
        or runtime['gpu_uuid'] != training['resource_migration']['gpu_uuid']
        or observed['FLA'] is not False):
        raise ValueError('Actual A800 probe runtime/GPU differs')
    if (done['manifest_sha256'] != manifest_ref['sha256']
        or observed['plan_sha256'] != training['expected_probe_plan']['sha256']
        or observed['source_sha256'] != probe_plan['payload_sources']['probe.py']['sha256']):
        raise ValueError('Actual probe runtime manifest/source differs from full64 plan')
    if probe_plan['diagnostic_rows'] != 12 or probe_plan['budget_seconds'] != 240 or probe_plan['grace_seconds'] != 30:
        raise ValueError('Unexpected pressure budget/shape schedule')
    estimate = decision['estimated_complete_campaign_seconds']
    if (type(estimate) not in (float, int) or not math.isfinite(estimate) or not 0 < estimate <= 5400
        or decision['formal_cost_fits_5400'] is not True or decision['hard_shared_budget_seconds'] != 5400
        or decision.get('actual_training_already_performed', False)):
        raise ValueError('Fresh two-epoch campaign does not fit fixed 5400 seconds')
    for item in training['source_files'].values():
        if ref(item['path']) != item:
            raise ValueError('Frozen trainer source changed')
    return {'status': 'ACTUAL_FULL64_H51_AND_COST_BOUND_BEFORE_AUTHORIZATION',
            'wrapper': ref(wrapper_path), 'decision': ref(decision_path),
            'probe_plan': training['expected_probe_plan'], 'steps_per_epoch': 34,
            'total_updates': 68, 'warmup_steps': 4, 'fresh_base': True,
            'estimated_campaign_seconds': estimate, 'shared_budget_seconds': 5400}
