"""Freeze one complete unscored component handoff and private evidence archive.

Only reads completed local artifacts. It cannot start a model, contact a
server, read gold, score, select an epoch, or regenerate a response.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tarfile
import time


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    source = Path(__file__).resolve().parent
    controller = root / 'controller'
    inference = root / 'originaldev64-inference-v1'
    probe = root / 'probe-v1/package-runtime-v2'
    old = root.parent / 'patient-response-code-full64-readiness-v1'
    old_controller = root.parent / 'patient-response-code-full64-formal-v1/controller'
    campaign, complete = read(controller / 'campaign.json'), read(controller / 'campaign-complete.json')
    assert campaign['cap_seconds'] == 5400 and campaign['grace_seconds'] == 30
    assert complete['status'] == 'TWO_COMPLETE_EPOCHS_68_STEPS_DURABLE_ACKS_VERIFIED'
    assert complete['campaign'] == campaign
    assert campaign['started_unix'] <= complete['at_epoch'] <= campaign['absolute_deadline_unix']
    acknowledgments = []
    for epoch in (1, 2):
        path = controller / f'epoch-{epoch:03d}-export-ack.json'
        ack = read(path)
        assert ack['status'] == 'persistent_checkpoint_verified'
        assert ack['completed_epochs'] == epoch and ack['global_step'] == 34 * epoch
        assert ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files']
        assert ack['destination_directories_fsynced']
        assert complete[f'epoch{epoch}_ack_sha256'] == ref(path)['sha256']
        assert read(controller / f'epoch-{epoch:03d}-source-ack-copy.json')['sha256'] == ref(path)['sha256']
        acknowledgments.append(ref(path))
    training_checks = read(controller / 'training-terminal/checks.json')
    assert training_checks['all_original_members_SHA']
    restore = read(controller / 'training-terminal/collected/resume-state-verified.json')
    assert restore['status'] == 'PASS_EXACT_LOGICAL_FULL_STATE_RESTORE'
    assert restore['completed_epochs'] == 1 and restore['global_step'] == 34
    assert restore['restored_state_digest'] == restore['serialized_state_digest'] and restore['all_RNG_restored']
    closure = controller / 'export-service-final-closure.json'
    if not closure.exists(): closure = controller / 'export-service-cleanup.json'
    if not closure.exists(): closure = controller / 'export-service-closure.json'
    assert read(closure)['all_exact_servers_absent']
    result = read(inference / 'terminal/summary.json')
    assert result['expected_rows'] == result['fresh_rows'] == 64 and result['cached_rows'] == 0
    assert result['strict_complete_rows'] + result['whole_row_abstentions'] == 64
    assert read(inference / 'terminal/wrapper-complete.json')['returncode'] == 0
    assert not read(inference / 'terminal/gpu-release.json')['target_compute_rows']
    component = read(inference / 'pure-component/manifest.json')
    assert component['documents'] == 16 and component['patients'] == 36 and component['entities'] == 955
    assert component['all_nonrelation_bytes_preserved'] and component['quality_calls'] == 0
    assert component['prediction'] == ref(inference / 'pure-component/pred.jsonl')
    entry = dict(status='COMPLETE_FULL64_C_PURE_COMPONENT_NO_LOCAL_QUALITY', at_unix=time.time(),
        source=ref(__file__), training_plan=ref(source / 'formal_train/training-plan.json'),
        controller_entry=ref(controller / 'entry.json'), campaign=ref(controller / 'campaign.json'),
        training_complete=ref(controller / 'campaign-complete.json'), persistent_full_state_ACKs=acknowledgments,
        training_metadata=ref(controller / 'training-terminal/manifest.json'),
        training_archive=ref(controller / 'training-terminal/terminal.zip'),
        actual_full_state_restore=ref(controller / 'training-terminal/collected/resume-state-verified.json'),
        CPU329=ref(old / 'native100-attempt2/native-terminal/native-result.json'),
        actual_A800_probe=ref(probe / 'collection.json'), cost=ref(root / 'formal-cost-decision.json'),
        initial_A100_probe=ref(old / 'probe-v1/package/collection.json'),
        A100_formal_closed_unexecuted=ref(old_controller / 'A100-unexecuted-closure.json'),
        abandoned_HTTP_closed=ref(old_controller / 'source-bridge/closure.json'),
        inference_plan=ref(inference / 'plan.json'), inference_manifest=ref(inference / 'terminal-manifest.json'),
        inference_archive=ref(inference / 'terminal.zip'), inference_summary=ref(inference / 'terminal/summary.json'),
        pure_component=ref(inference / 'pure-component/pred.jsonl'), component_manifest=ref(inference / 'pure-component/manifest.json'),
        rows=64, strict_rows=result['strict_complete_rows'], whole_row_abstentions=result['whole_row_abstentions'],
        originaldev_quality_calls=0, originaldev_gold_read=False, source_entity_frame_only=955,
        root_expected_final_entity_frame=1061, no_global_best_claim=True,
        actual_training_runtime='A800 Torch2.7.1+cu126/native Transformers5.20.2',
        earlier_C32_internal_runtime='A100 Torch2.6.0+cu124; no cross-host bitwise claim',
        purpose='Pure C supplied only to root prelocked full64 fusion; no standalone originaldev assessment')
    save(root / 'final-review-entry.json', entry)
    destination = root / 'terminal-backup'
    destination.mkdir(exist_ok=False)
    files = {}
    def add(name, path):
        path = Path(path)
        assert path.is_file() and not path.is_symlink() and name not in files
        assert path.name not in ('access-token', 'training_state.pt', 'adapter_model.safetensors')
        files[name] = path
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.diff') and '__pycache__' not in path.parts:
            add('source/' + path.relative_to(source).as_posix(), path)
    add('source-dependencies/original-F0-store-terminal.py', source.parent / 'f0/store_terminal.py')
    for name in ('final-review-entry.json', 'formal-cost-decision.json', 'initial-inventory.json'):
        add(name, root / name)
    for name in ('entry.json', 'controller-plan.json', 'campaign.json', 'campaign-complete.json', 'GPU-AUTHORIZATION.json',
                 'source-deployment.json', 'target-deployment.json', 'bootstrap-entry.json', 'watch-job.json',
                 'watch-lifecycle-observed.json', 'watch-lock-contract.json',
                 'export-service-closure-intent.json', 'export-service-closure-tool.json', 'export-service-closure.json'):
        add('controller/' + name, controller / name)
    for name in ('export-service-cleanup-intent.json', 'export-service-cleanup-tool.json', 'export-service-cleanup.json',
                 'export-service-final-closure-intent.json', 'export-service-final-closure-tool.json', 'export-service-final-closure.json'):
        if (controller / name).exists(): add('controller/' + name, controller / name)
    for epoch in (1, 2):
        for suffix in ('job', 'state', 'export-ack', 'export-verification', 'source-ack-copy', 'server-ready'):
            name = f'epoch-{epoch:03d}-{suffix}.json'
            add('controller/' + name, controller / name)
    for name in ('manifest.json', 'checks.json', 'terminal.zip'):
        add('training/' + name, controller / 'training-terminal' / name)
    add('probe/collection.json', probe / 'collection.json')
    add('probe/terminal.zip', probe / 'collection-transfer/probe-output.zip')
    for name in ('plan.json', 'entry.json', 'terminal-manifest.json', 'terminal.zip', 'held.response-code.jsonl',
                 'pure-component/pred.jsonl', 'pure-component/manifest.json'):
        add('inference/' + name, inference / name)
    add('original-CPU/structural-result.json', old / 'structural-result.json')
    add('original-CPU/native-result.json', old / 'native100-attempt2/native-terminal/native-result.json')
    add('original-CPU/native-row-audit.json', old / 'native100-attempt2/native-terminal/native-row-audit.json')
    add('original-A100/closure.json', old_controller / 'A100-unexecuted-closure.json')
    add('original-A100/HTTP-closure.json', old_controller / 'source-bridge/closure.json')
    add('original-A100/probe.zip', old / 'probe-v1/package/collection-transfer/probe-output.zip')
    members = {name: {key: value for key, value in ref(path).items() if key != 'path'} for name, path in files.items()}
    archive = destination / 'terminal.tar.gz'
    with tarfile.open(archive, 'x:gz') as output:
        for name, path in sorted(files.items()):
            output.add(path, arcname=name, recursive=False)
    with archive.open('rb') as handle:
        os.fsync(handle.fileno())
    with tarfile.open(archive) as check:
        assert len(check.getnames()) == len(members) and set(check.getnames()) == set(members)
        for name, expected in members.items():
            data = check.extractfile(name).read()
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256']
    save(destination / 'archive-manifest.json', dict(archive=ref(archive), members=members,
        private_clinical_inputs_and_responses=True, no_held_gold=True, no_model_optimizer_arrays=True,
        model_states_already_durable_via_two_ACKs=True, new_model_or_quality_calls=0))
    descriptor = os.open(str(destination), os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps(dict(entry=ref(root / 'final-review-entry.json'), archive=ref(archive), members=len(members))))


if __name__ == '__main__':
    main()
