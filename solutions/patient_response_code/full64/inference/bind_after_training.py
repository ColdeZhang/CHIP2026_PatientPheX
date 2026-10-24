"""Bind actual full64 campaign completion and both durable ACKs; local stdlib only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tarfile

SCHEMA = 'patientphex-qwen35-full64-response-code-indexedHead-lora-v1'
HELD_SHA = 'b76e4b3de00432c692c0cefd84baa44275cf62404e9887f1e051ec0982db1f9b'
DEV_INPUT_SHA = 'c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e'
TRAIN_SHA = '4fdc4c6684f43bed88551cef1e2b72d67877efda49c896e5f34b855ca6d48d07'
NATIVE_SHA = '0ef29c1252c53527e8ee6947d05006e7179edd60a62dcfaea3f530ac87b2d627'
AUDIT_SHA = '51e7a3327d09966cc08a250571e03d3a97705a5e53a0abe2a062e810d4704adb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, default=Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-readiness-v1'))
    parser.add_argument('--dev-input', type=Path, default=Path('/workspace/patientphex-data/day1/dev16.input.jsonl'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    controller, destination, readiness = args.controller, args.directory.resolve(), args.readiness
    # Full prompts/artifacts must remain outside this checkout.
    assert source.parents[3] not in destination.parents
    cp = json.loads((controller / 'controller-plan.json').read_text())
    done = json.loads((controller / 'campaign-complete.json').read_text())
    assert cp['fingerprint_schema'] == SCHEMA
    assert cp['train_sha256'] == TRAIN_SHA and cp['no_probe_state_reuse'] is True
    assert cp['remote_relative'] == 'patient-response-code/full64-v1'
    assert cp['remote_absolute'] == '/home/zhangyuheng/projects/CHIP2026_PatientPheX/patient-response-code/full64-v1'
    assert cp['output_dir'] == '/dev/shm/patientphex-response-code-20261024/full64-formal2ep'
    assert cp['persistent_checkpoint_root'] == cp['remote_absolute'] + '/checkpoints'
    assert done['status'] == 'TWO_COMPLETE_EPOCHS_68_STEPS_DURABLE_ACKS_VERIFIED'
    assert done['campaign']['plan_sha256'] == sha(controller / 'controller-plan.json')
    assert done['campaign']['started_unix'] <= done['at_epoch'] <= done['campaign']['absolute_deadline_unix']
    trainer_sha = sha(source.parent / 'formal_train/train.py')
    assert trainer_sha == cp['trainer_sha256']
    acks = []
    evidence = {'training-controller-plan.json': controller / 'controller-plan.json',
                'training-campaign-complete.json': controller / 'campaign-complete.json'}
    for epoch in (1, 2):
        tag = f'epoch-{epoch:03d}'
        ack_path = controller / (tag + '-export-ack.json')
        ack = json.loads(ack_path.read_text())
        assert ack['status'] == 'persistent_checkpoint_verified'
        assert ack['completed_epochs'] == epoch and ack['global_step'] == 34 * epoch
        assert ack['trainer_sha256'] == trainer_sha
        assert ack['source_checkpoint_directory'] == cp['output_dir'] + '/' + tag
        assert ack['persistent_directory'] == cp['persistent_checkpoint_root'] + '/' + tag
        assert ack['persistent_mount_id'] == cp['persistent_mount_id']
        assert ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files'] and ack['destination_directories_fsynced']
        assert sha(ack_path) == done[f'epoch{epoch}_ack_sha256']
        verification_path = controller / (tag + '-export-verification.json')
        verified = json.loads(verification_path.read_text())
        assert verified['all_files_reopened_SHA'] and verified['ack'] == ack
        copy_path = controller / (tag + '-source-ack-copy.json')
        assert json.loads(copy_path.read_text())['sha256'] == sha(ack_path)
        ready_path = controller / (tag + '-server-ready.json')
        ready = json.loads(ready_path.read_text())
        assert ack['files'] == ready['manifest']['files']
        assert ack['transport_manifest_sha256'] == ready['manifest_sha256']
        assert ack['files']['_run/' + cp['export_trainer_basename']]['sha256'] == trainer_sha
        assert ack['files']['checkpoint.json']['sha256'] == ack['checkpoint_sha256']
        assert ack['files']['_run/run_manifest.json']['sha256'] == ack['run_manifest_sha256']
        if epoch == 2:
            assert '_run/training_complete.json' in ack['files']
        for path in (ack_path, verification_path, copy_path, ready_path):
            evidence['training-' + path.name] = path
        acks.append(ack)
    native_path = readiness / 'native100-attempt2/native-terminal/native-result.json'
    audit_path = readiness / 'native100-attempt2/native-terminal/native-row-audit.json'
    held_path = readiness / 'held.response-code.jsonl'
    assert sha(native_path) == NATIVE_SHA and sha(audit_path) == AUDIT_SHA
    assert sha(held_path) == HELD_SHA and sha(args.dev_input) == DEV_INPUT_SHA
    native = json.loads(native_path.read_text())
    assert native['status'] == 'PASS_NATIVE_CPU_RESPONSE_CODE_FULL64_CENSUS_NO_MODEL'
    assert native['audit_sha256'] == AUDIT_SHA and native['model_loaded'] is False
    assert native['runtime'] == dict(torch='2.7.1+cu126', transformers='5.20.2', tokenizers='0.23.1')
    held_census = native['groups']['held']['arms']['response_code']
    assert held_census['rows'] == 64 and held_census['total_tokens'] == 523821
    assert held_census['max_tokens'] == 13425 and held_census['longest_indices'] == [1]
    audit = [row for row in json.loads(audit_path.read_text()) if row['group'] == 'held' and row['arm'] == 'response_code']
    rows = [json.loads(line) for line in held_path.read_text().splitlines()]
    assert len(rows) == len(audit) == 64
    assert len({row['pmc_id'] for row in rows}) == 16
    assert len({(row['pmc_id'], pid) for row in rows for pid in row['patient_ids']}) == 36
    assert sum(len(row['code_map']) for row in rows) == 466
    assert sum(len(row['code_map']) * len(row['patient_ids']) for row in rows) == 1006
    for index, (row, expected) in enumerate(zip(rows, audit)):
        assert 'answer' not in row
        assert expected['index'] == index and expected['row_id'] == row['row_id'] and expected['pmc_id'] == row['pmc_id']
        assert expected['published_native_Auto_template_decode_exact'] is True
    destination.mkdir(exist_ok=False)
    sources = {name: source / name for name in ('infer_full64.py', 'probe_runner.py', 'legacy_runner.py', 'launch.py', 'response_codec.py', 'model-source-manifest.json')}
    sources.update({'held.response-code.jsonl': held_path, 'native-result.json': native_path, **evidence})
    for name, path in sources.items():
        with (destination / name).open('xb') as handle:
            handle.write(path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    save(destination / 'row-audit.json', audit)
    files = {name: {key: value for key, value in ref(destination / name).items() if key != 'path'} for name in [*sources, 'row-audit.json']}
    plan = dict(schema='patient-response-code-full64-two-epoch-originaldev64-native-v1',
        status='TWO_ACTUAL_DURABLE_ACKS_BOUND_NO_MODEL', mount_id=cp['mount_id'], gpu_uuid=cp['gpu_uuid'],
        python=cp['python'], model=cp['model'], remote='patient-response-code/full64-v1/originaldev64-inference-v1',
        adapter=cp['output_dir'] + '/epoch-002/adapter',
        adapter_files={name: acks[1]['files']['adapter/' + name]['sha256'] for name in ('adapter_config.json', 'adapter_model.safetensors')},
        env=cp['env'], cap_seconds=1200, term_grace_seconds=30, input_limit=13824, generation_limit=1600,
        cached_original_indices=[], fresh_rows=64, training_completed_epochs=2, training_global_step=68,
        configured_eos_ids=[248044, 248046],
        expected_runtime=dict(torch='2.6.0+cu124', transformers='5.20.2', peft='0.21.2', tokenizers='0.23.1',
            actual_tokenizer_class='Qwen3_5Tokenizer', matmul_TF32=False, cudnn_TF32=True),
        native_CPU_runtime=native['runtime'], native_census=ref(native_path), native_full_audit=ref(audit_path),
        originaldev_input=ref(args.dev_input), files=files,
        training_provenance=dict(controller=ref(controller / 'controller-plan.json'), campaign=ref(controller / 'campaign-complete.json'),
            ACKs=[ref(controller / f'epoch-{epoch:03d}-export-ack.json') for epoch in (1, 2)], epoch2_full_files=acks[1]['files']),
        raw_first=True, whole_row_abstention=True, no_repair=True, canonical_only=True)
    save(destination / 'plan.json', plan)
    archived = {name: ref(destination / name) for name in [*files, 'plan.json']}
    with tarfile.open(destination / 'deployment.tar.gz', 'x:gz') as archive:
        for name in archived:
            archive.add(destination / name, arcname=name, recursive=False)
    with tarfile.open(destination / 'deployment.tar.gz') as archive:
        assert len(archive.getnames()) == len(archived) and set(archive.getnames()) == set(archived)
        for name, record in archived.items():
            data = archive.extractfile(name).read()
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    save(destination / 'deployment-manifest.json', dict(archive=ref(destination / 'deployment.tar.gz'), files=archived))
    save(destination / 'entry.json', dict(plan=ref(destination / 'plan.json'), operations=ref(source / 'operations.py'),
        deployment=ref(destination / 'deployment-manifest.json'), all64fresh=True, old_cache_reuse=False,
        complete_canonical_handoff='terminal/canonical.jsonl'))
    print(json.dumps(ref(destination / 'entry.json')))


if __name__ == '__main__':
    main()
