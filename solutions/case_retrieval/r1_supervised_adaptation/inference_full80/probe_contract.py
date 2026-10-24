"""Small metadata/input contract for a single final-adapter R1 probe; no ML imports."""
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(value):
    path = Path(value['path'])
    assert not path.is_symlink() and path.stat().st_size == value['bytes'] and sha(path) == value['sha256']
    return path


def ids_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def check_sources(plan, directory):
    for name, expected in plan['sources'].items():
        assert Path(name).name == name
        path = Path(directory) / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']


def fixed_inputs(plan):
    original = gzip.decompress(checked(plan['inputs']['prompts']).read_bytes())
    assert hashlib.sha256(original).hexdigest() == plan['prompts_uncompressed_sha256']
    rows = [json.loads(line) for line in original.decode().splitlines()]
    encoded = [json.loads(line) for line in gzip.decompress(checked(plan['inputs']['encoded']).read_bytes()).decode().splitlines()]
    audits = json.loads(checked(plan['inputs']['audit']).read_text())
    cpu = json.loads(checked(plan['inputs']['cpu_result']).read_text())
    assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact']
    assert cpu['outputs']['all80-published-encoded.jsonl.gz'] == plan['inputs']['encoded']
    assert cpu['outputs']['all80-input-audit.json'] == plan['inputs']['audit']
    assert len(rows) == len(encoded) == len(audits) == 80
    assert len({row['row_id'] for row in rows}) == 80
    assert {str(row['pmc_id']) for row in rows} == set(plan['F1_ids'])
    assert not set(plan['F1_ids']) & set(plan['train32_ids'])
    for index, (row, item, audit) in enumerate(zip(rows, encoded, audits)):
        assert set(row) == {'row_id', 'pmc_id', 'code_map', 'patient_ids', 'messages'}
        assert row['row_id'] == item['row_id'] == audit['row_id'] and index == audit['index']
        assert hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == audit['original_row_sha256']
        assert ids_sha(item['input_ids']) == audit['native_ids_sha256']
        assert len(item['input_ids']) == audit['native_input_tokens']
        assert all(type(value) is int and 0 <= value < 151936 for value in item['input_ids'])
        assert [message['role'] for message in row['messages']] == ['system', 'user']
    lengths = [len(item['input_ids']) for item in encoded]
    assert sum(lengths) == 249937 and max(lengths) == 5689
    assert [i for i, length in enumerate(lengths) if length == max(lengths)] == [73]
    assert audits[73] == plan['selected'] and rows[73]['row_id'] == '6063015:relations:0'
    return rows, encoded, audits


def completed_binding(plan, training_directory):
    directory = Path(training_directory).resolve()
    assert str(directory) == plan['training_directory']
    terminal_path = directory / 'terminal.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['exit_code'] == 0 and terminal['timeout_triggered'] is False
    def terminal_file(relative):
        expected = terminal['files'][relative]
        path = directory / relative
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        return path
    training_plan_path = terminal_file('plan.json')
    training_source_path = terminal_file('formal_train.py')
    assert sha(training_plan_path) == plan['training_plan_sha256']
    assert sha(training_source_path) == plan['training_source_sha256']
    training_plan = json.loads(training_plan_path.read_text())
    assert training_plan['train32_ids'] == plan['train32_ids']
    assert set(plan['F1_ids']) <= set(training_plan['excluded_ids'])
    checkpoint_path = terminal_file('actual/epoch-002/checkpoint.json')
    checkpoint = json.loads(checkpoint_path.read_text())
    result_path = terminal_file('actual/result.json')
    result = json.loads(result_path.read_text())
    assert checkpoint['status'] == 'complete_epoch' and checkpoint['completed_epochs'] == 2 and checkpoint['global_step'] == 32
    assert result['status'] == 'complete_fresh_R1_124x2_32steps' and result['completed_epochs'] == 2 and result['global_step'] == 32
    assert checkpoint['fingerprint'] == result['fingerprint'] == plan['expected_fingerprint']
    assert checkpoint['base_frozen'] and checkpoint['no_validation_or_quality'] and checkpoint['all_train_rows_used']
    assert checkpoint['total_rows_seen'] == result['rows_seen'] == 248
    assert checkpoint['total_tokens_seen'] == result['input_tokens_seen'] == 802582
    assert checkpoint['total_supervised_seen'] == result['supervised_tokens_seen'] == 14092
    assert result['fresh_public_base'] and result['probe_state_not_loaded'] and result['inference_or_quality'] is False
    assert result['first_nonzero_update_verified'] and result['no_truncation_or_quantization']
    required = {'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'training_state.pt'}
    assert required <= set(checkpoint['files'])
    files = {}
    for relative, expected in checkpoint['files'].items():
        name = Path(relative)
        assert not name.is_absolute() and '..' not in name.parts
        path = terminal_file('actual/epoch-002/' + relative)
        assert not path.is_symlink() and ref(path) == dict(path=str(path.resolve()), **expected)
        files[relative] = ref(path)
    assert {str(p.relative_to(checkpoint_path.parent)) for p in checkpoint_path.parent.rglob('*') if p.is_file()} == set(files) | {'checkpoint.json'}
    ack_path = terminal_file('actual/epoch-002-persistent-ack.json')
    ack = json.loads(ack_path.read_text())
    assert ack['status'] == 'COMPLETE_EPOCH_DURABLE_FSYNC_REOPEN_ALL_SHA_ADAPTER_AND_ADAM_RNG_EXACT'
    assert ack['completed_epochs'] == 2 and ack['global_step'] == 32 and ack == result['acknowledgements'][-1]
    assert ack['checkpoint'] == ref(checkpoint_path)
    assert ack['adapter'] == files['adapter/adapter_model.safetensors']
    assert ack['training_state'] == files['training_state.pt'] and ack['adapter_config'] == files['adapter/adapter_config.json']
    assert ack['adapter_tensors_reopened'] == ack['optimizer_states_reopened'] == 504 and ack['all_rng_reopened_exact']
    config = json.loads(checked(files['adapter/adapter_config.json']).read_text())
    for key in ['r', 'lora_alpha', 'lora_dropout', 'bias', 'task_type']:
        assert config[key] == plan['lora'][key]
    assert set(config['target_modules']) == set(plan['lora']['target_modules']) and config['peft_type'] == 'LORA'
    model_manifest_path = terminal_file('actual/model-manifest.json')
    model_manifest = json.loads(model_manifest_path.read_text())
    assert model_manifest['source_sha256'] == plan['training_source_sha256'] and model_manifest['plan_sha256'] == plan['training_plan_sha256']
    assert model_manifest['precision'] == 'BF16 base + FP32 LoRA, unquantized full head'
    for name, expected in plan['model_files'].items():
        actual = model_manifest['asset_files'][name]
        assert actual['bytes'] == expected['bytes'] and actual['sha256'] == expected['sha256']
    return dict(status='BOUND_COMPLETE_EPOCH2_32STEP_ALL_CHECKPOINT_FILES_SHA',
                terminal=ref(terminal_path), result=ref(result_path), checkpoint=ref(checkpoint_path),
                adapter=files['adapter/adapter_model.safetensors'], training_state=files['training_state.pt'],
                adapter_config=files['adapter/adapter_config.json'], files=files,
                persistent_ack=ref(ack_path), model_manifest=ref(model_manifest_path),
                fingerprint=checkpoint['fingerprint'], all_checkpoint_files_verified=True)
