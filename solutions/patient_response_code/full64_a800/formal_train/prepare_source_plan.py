"""Freeze only migrated source/config; does not prepare an executable campaign."""
from pathlib import Path
import hashlib
import json


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    here = Path(__file__).resolve().parent
    original = here.parents[1] / 'full64/formal_train/training-plan.json'
    probe = here.parent / 'probe/plan.json'
    assert ref(probe)['sha256'] == '21255251ba4bb0ab432c182ee45b4f2d9729177575cdbaf1f21a5347a35ddfce'
    plan = json.loads(original.read_text())
    project = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
    external = Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-a800-v1')
    old_model, old_output = plan['model'], plan['output_directory']
    plan.update(
        status='PREPARED_DISABLED_AWAITING_ACTUAL_A800_FULL64_H51_PASS_AND_COST_LE5400',
        output_directory='/dev/shm/patientphex-response-code-20261024/full64-a800-formal2ep',
        persistent_directory=project + '/patient-response-code/full64-a800-v1/checkpoints',
        remote_source_directory=project + '/patient-response-code/full64-a800-v1',
        python=project + '/ontology-round2/qwen-venv/bin/python',
        pythonpath=project + '/case-retrieval/qwen35-preparation/a800-migration/runtime',
        model='/dev/shm/patientphex-qwen35-1045-c20223623576',
        expected_probe_plan=ref(probe),
        default_local_controller=str(external / 'controller'))
    plan['formal_runtime']['torch'] = '2.7.1+cu126'
    plan['prior_native_census_runtime']['note'] = (
        'Actual .100 CPU no model/CUDA, same Torch2.7.1+cu126 version; receipt reused. '
        'All265 A800 actual train input IDs and labels are checked before first model forward. '
        'Same package version does not establish cross-host numerical equality.')
    plan['common_args'] = [plan['model'] if value == old_model else
        plan['output_directory'] if value == old_output else value for value in plan['common_args']]
    plan['source_files'] = {name: ref(here / name) for name in plan['source_files']}
    plan['source_files']['prepare_source_plan.py'] = ref(__file__)
    plan['data_arguments']['model_source_manifest'] = ref(here / 'model-source-manifest.json')
    plan['resource_migration'] = dict(
        original_training_plan=ref(original),
        original_full64_CPU_contract=ref(original.parent / 'CPU-contract-result.json'),
        actual_inventory=ref(external / 'initial-inventory.json'),
        mount_id='69b991ee-183f-4add-af1a-579af92479c9',
        gpu_index=2, gpu_uuid='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc',
        source_private_IP='10.5.4.7',
        actual_new_probe_terminal=str(external / 'probe-v1/package-runtime-v2/terminal'),
        A100_formal_never_started=ref(Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/controller/A100-unexecuted-closure.json')),
        source_project_disk_min_bytes=512*1024**2,
        private_SHM_min_free_bytes=4*1024**3,
        persistent_target_disk_min_bytes=2*1024**3,
        GPU_free_min_MiB=70000, GPU_total_min_MiB=80000,
        state_location='privateSHM_then_two_complete_.100_ACKs',
        all_original_265_train_and_64_inference_rows_retained=True,
        native_CPU_tokenization_repeated=False,
        old_A100_probe_not_a_new_A800_result=True,
        new_probe_terminal_or_cost_bound=False,
        helper_remote_GPU_Git_calls=0)
    plan['disabled_activation']['reason'] = (
        'New A800 probe plan/source bound only. Actual A800 H51 terminal and measured '
        'cost must pass controller/activation.py; no deployment or GPU authorization exists here.')
    target = here / 'training-plan.json'
    with target.open('x') as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(ref(target)))


if __name__ == '__main__':
    main()
