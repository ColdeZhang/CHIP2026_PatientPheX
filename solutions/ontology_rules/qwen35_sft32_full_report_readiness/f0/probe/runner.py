"""One fixed original-SFT48 F0 full-report input: native greedy generation, raw first."""
from pathlib import Path
import hashlib
import json
import os
import time

import legacy_runner as legacy


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=legacy.json_default)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_complete(text, row):
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != set(row['code_map']):
        raise ValueError('Expected exactly the supplied candidate keys')
    allowed = set(row['patient_ids'])
    for patients in value.values():
        if not isinstance(patients, list):
            raise ValueError('Patient value must be a list')
        if any(type(pid) is not str or pid not in allowed for pid in patients):
            raise ValueError('Unknown patient ID or wrong type')
        if len(patients) != len(set(patients)):
            raise ValueError('Duplicate patient ID')
    return value


def main():
    started = time.monotonic()
    plan = json.loads(Path('plan.json').read_text())
    for name, ref in plan['files'].items():
        assert Path(name).stat().st_size == ref['bytes'] and legacy.digest(name) == ref['sha256']
    assert plan['input_limit'] == ((plan['row_audit']['expanded']['tokens'] + 511) // 512) * 512
    assert plan['row_audit']['expanded']['tokens'] <= 32768 and plan['generation_limit'] == 1600
    assert 0 <= plan['probe_index'] < 58 and plan['cap_seconds'] == 240
    rows = legacy.read_prompts('probe.jsonl')
    assert len(rows) == 1
    row = rows[0]
    assert jsha(row) == plan['row_audit']['expanded_row_sha256']
    verified_base = legacy.verify_model(plan['model'], 'model-source-manifest.json')
    verified_adapter = {}
    for name, digest in plan['adapter_files'].items():
        path = Path(plan['adapter']) / name
        assert legacy.digest(path) == digest
        verified_adapter[name] = {'bytes': path.stat().st_size, 'sha256': digest}
    import torch
    import transformers
    import peft
    import tokenizers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from peft import PeftModel
    assert transformers.__version__ == '5.20.2' and peft.__version__ == '0.21.2'
    assert torch.__version__ == '2.7.1+cu126'
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    expected = plan['row_audit']['expanded']
    assert len(ids) == expected['tokens'] <= plan['input_limit']
    assert jsha(ids) == expected['input_ids_sha256']
    assert hashlib.sha256(rendered.encode()).hexdigest() == expected['rendered_sha256']
    text_config = Qwen3_5TextConfig.from_pretrained(plan['model'], local_files_only=True)
    assert text_config.max_position_embeddings >= len(ids) + plan['generation_limit']
    save('actual-input-and-assets.json', {
        'row_id': row['row_id'], 'index': plan['probe_index'], 'input_tokens': len(ids),
        'input_ids_sha256': jsha(ids), 'rendered_sha256': expected['rendered_sha256'],
        'max_position_embeddings': text_config.max_position_embeddings,
        'base_files': verified_base, 'adapter_files': verified_adapter,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'peft': peft.__version__, 'tokenizers': tokenizers.__version__,
        'actual_tokenizer_class': type(tokenizer).__name__,
        'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_TF32': torch.backends.cudnn.allow_tf32,
        'no_truncation': True, 'gold_read': False})
    loading_started = time.monotonic()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        plan['model'], config=text_config, dtype=torch.bfloat16, device_map='cuda:0',
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    save('base-loading.json', loading)
    assert not any(loading.get(key) for key in ['missing_keys', 'mismatched_keys', 'error_msgs'])
    assert all(key.startswith(('mtp.', 'model.mtp.', 'model.visual.', 'visual.'))
               for key in loading.get('unexpected_keys', []))
    base_count = sum(parameter.numel() for parameter in model.parameters())
    assert base_count == legacy.TEXT_PARAMETERS
    model = PeftModel.from_pretrained(model, plan['adapter'], is_trainable=False)
    model.eval()
    total_count = sum(parameter.numel() for parameter in model.parameters())
    assert total_count == 8_997_081_600 < 10_000_000_000
    loading_seconds = time.monotonic() - loading_started
    configured_eos = model.generation_config.eos_token_id
    eos_ids = set(configured_eos if isinstance(configured_eos, list) else [configured_eos])
    eos_ids.add(tokenizer.eos_token_id)
    eos_ids.discard(None)
    assert eos_ids
    encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda:0')
    assert encoded['input_ids'][0].tolist() == ids
    torch.cuda.synchronize()
    generation_started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=False, use_cache=True,
            max_new_tokens=plan['generation_limit'], eos_token_id=sorted(eos_ids),
            pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - generation_started
    generated_ids = generated[0, len(ids):].tolist()
    natural_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
    hit_cap = len(generated_ids) >= plan['generation_limit'] and not natural_eos
    # Durable IDs are saved before decoding or any response validation.
    save('generated-ids.json', {'row_id': row['row_id'], 'input_ids_sha256': jsha(ids),
        'generated_ids': generated_ids, 'configured_eos_ids': sorted(eos_ids),
        'natural_eos': natural_eos, 'hit_token_limit': hit_cap,
        'generation_seconds': generation_seconds})
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    save('raw-response.json', {'row_id': row['row_id'], 'pmc_id': row['pmc_id'],
        'response': response, 'input_tokens': len(ids), 'output_tokens': len(generated_ids),
        'hit_token_limit': hit_cap, 'natural_eos': natural_eos,
        'batch_seconds': generation_seconds, 'removed_demonstrations': 0})
    parsed, error = None, None
    try:
        if not natural_eos or hit_cap:
            raise ValueError('No natural EOS or generation reached cap')
        parsed = parse_complete(response, row)
    except (ValueError, TypeError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    save('format-result.json', {'strict_complete': parsed is not None,
        'error': error, 'parsed': parsed, 'gold_read': False, 'quality_calls': 0})
    result = {'status': 'PASS_NATIVE_LONGEST_STRICT_COMPLETE' if parsed is not None else 'FORMAT_FAILURE_AFTER_RAW_SAVED',
        'row_id': row['row_id'], 'index': plan['probe_index'], 'input_tokens': len(ids),
        'generated_tokens': len(generated_ids), 'generation_seconds': generation_seconds,
        'generation_tokens_per_second_including_prefill': len(generated_ids) / generation_seconds,
        'loading_seconds': loading_seconds, 'elapsed_seconds': time.monotonic() - started,
        'base_parameters': base_count, 'total_parameters': total_count,
        'strict_complete': parsed is not None, 'natural_eos': natural_eos, 'hit_token_limit': hit_cap,
        'max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_reserved_bytes': torch.cuda.max_memory_reserved(),
        'source_sha256': legacy.digest(__file__), 'plan_sha256': legacy.digest('plan.json'),
        'raw_ids_saved_before_decode': True, 'raw_response_saved_before_parse': True,
        'gold_read': False, 'quality_calls': 0}
    save('summary.json', result)
    print(json.dumps(result), flush=True)
    assert parsed is not None, error


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', {'exception': type(exc).__name__, 'message': str(exc),
                             'source_sha256': legacy.digest(__file__)})
        raise
