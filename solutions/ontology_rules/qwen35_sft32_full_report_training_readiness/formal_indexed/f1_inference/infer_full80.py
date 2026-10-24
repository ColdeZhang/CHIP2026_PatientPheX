"""Fixed complete80 newly fulltext-trained SFT32 inference, all 80 fresh."""
from pathlib import Path
import hashlib
import json
import os
import time

import probe_runner as probe
legacy = probe.legacy
save = probe.save
jsha = probe.jsha
progress = {'completed': 0, 'current_original_index': None, 'current_row_id': None}


def jsonl(path, records):
    with Path(path).open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open('.', os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def copy_exact(source, destination):
    path = Path(destination)
    with path.open('xb') as handle:
        handle.write(Path(source).read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def main():
    started = time.monotonic()
    plan = json.loads(Path('plan.json').read_text())
    assert plan['cap_seconds'] == 1200 and plan['term_grace_seconds'] == 30
    assert plan['input_limit'] == 14400 and plan['generation_limit'] == 1600
    assert plan['cached_original_indices'] == [] and plan['fresh_rows'] == 80
    assert plan['training_completed_epochs'] == 2 and plan['training_global_step'] == 32
    for name, ref in plan['files'].items():
        assert Path(name).stat().st_size == ref['bytes'] and legacy.digest(name) == ref['sha256']
    rows = legacy.read_prompts('expanded80.jsonl')
    audit = json.loads(Path('row-audit.json').read_text())
    assert len(rows) == len(audit) == 80 and len({row['pmc_id'] for row in rows}) == 16
    for index, (row, expected) in enumerate(zip(rows, audit)):
        assert expected['index'] == index and row['row_id'] == expected['row_id']
        assert jsha(row) == expected['expanded_row_sha256']
    assert sum(len(row['code_map']) for row in rows) == 587
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
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
        'peft': peft.__version__, 'tokenizers': tokenizers.__version__,
        'actual_tokenizer_class': type(tokenizer).__name__,
        'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_TF32': torch.backends.cudnn.allow_tf32}
    assert runtime == plan['expected_runtime']
    prepared = []
    for index, row in enumerate(rows):
        rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        expected = audit[index]['expanded']
        assert len(ids) == expected['tokens'] <= plan['input_limit']
        assert jsha(ids) == expected['input_ids_sha256']
        assert hashlib.sha256(rendered.encode()).hexdigest() == expected['rendered_sha256']
        prepared.append((len(ids), row['row_id'], index, rendered, ids))
    assert sum(item[0] for item in prepared) == 670122
    assert max(prepared)[0] == 14390
    prepared.sort(key=lambda item: (item[0], item[1]))
    text_config = Qwen3_5TextConfig.from_pretrained(plan['model'], local_files_only=True)
    assert text_config.max_position_embeddings >= 15990
    save('actual-input-and-assets.json', {
        'rows': 80, 'documents': 16, 'candidate_codes': 587,
        'input_tokens': 670122, 'max_input_tokens': 14390,
        'all80_actual_ids_equal_prior_native_CPU': True,
        'base_files': verified_base, 'adapter_files': verified_adapter,
        'max_position_embeddings': text_config.max_position_embeddings,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'peft': peft.__version__, 'tokenizers': tokenizers.__version__,
        'actual_tokenizer_class': type(tokenizer).__name__,
        'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_TF32': torch.backends.cudnn.allow_tf32,
        'cached_original_indices': [], 'no_truncation': True, 'gold_read': False})
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
    assert sorted(eos_ids) == plan['configured_eos_ids']
    for name in ['generated-ids', 'raw-responses', 'canonical-records']:
        Path(name).mkdir(exist_ok=False)
    raw_records, canonical_records = [], []
    fresh_count = 0
    generation_seconds_total = 0.0
    for length, _, index, rendered, ids in prepared:
        row = rows[index]
        progress.update(current_original_index=index, current_row_id=row['row_id'])
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
        generation_seconds_total += generation_seconds
        generated_ids = generated[0, len(ids):].tolist()
        natural_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
        hit_cap = len(generated_ids) >= plan['generation_limit'] and not natural_eos
        save(f'generated-ids/{index:03d}.json', {'row_id': row['row_id'],
            'input_ids_sha256': jsha(ids), 'generated_ids': generated_ids,
            'configured_eos_ids': sorted(eos_ids), 'natural_eos': natural_eos,
            'hit_token_limit': hit_cap, 'generation_seconds': generation_seconds})
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        raw = {'row_id': row['row_id'], 'pmc_id': row['pmc_id'],
            'response': response, 'input_tokens': len(ids), 'output_tokens': len(generated_ids),
            'hit_token_limit': hit_cap, 'natural_eos': natural_eos,
            'batch_seconds': generation_seconds, 'removed_demonstrations': 0}
        save(f'raw-responses/{index:03d}.json', raw)
        parsed, error = None, None
        try:
            if not natural_eos or hit_cap:
                raise ValueError('No natural EOS or generation reached cap')
            parsed = probe.parse_complete(response, row)
        except (ValueError, TypeError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        provenance = 'fresh'
        fresh_count += 1
        canonical = {'row_id': row['row_id'], 'pmc_id': row['pmc_id'],
            'candidate_patients': parsed if parsed is not None else {},
            'strict_complete': parsed is not None, 'abstention': error,
            'provenance': provenance}
        save(f'canonical-records/{index:03d}.json', canonical)
        raw_records.append(raw)
        canonical_records.append(canonical)
        progress['completed'] += 1
        print(json.dumps({'completed': progress['completed'], 'original_index': index,
            'provenance': provenance, 'strict_complete': parsed is not None,
            'output_tokens': raw['output_tokens']}), flush=True)
    assert fresh_count == 80 and len(raw_records) == len(canonical_records) == 80
    assert {record['row_id'] for record in raw_records} == {row['row_id'] for row in rows}
    assert all(record['provenance'] == 'fresh' for record in canonical_records)
    jsonl('responses.jsonl', raw_records)
    jsonl('canonical.jsonl', canonical_records)
    result = {'status': 'COMPLETE_NEW_FULLTEXT_SFT32_FIXED80_ALL_FRESH_NO_QUALITY',
        'expected_rows': 80, 'fresh_rows': 80, 'cached_rows': 0, 'cached_original_indices': [],
        'strict_complete_rows': sum(record['strict_complete'] for record in canonical_records),
        'whole_row_abstentions': sum(not record['strict_complete'] for record in canonical_records),
        'fresh_generation_seconds': generation_seconds_total, 'loading_seconds': loading_seconds,
        'elapsed_seconds': time.monotonic() - started, 'base_parameters': base_count,
        'total_parameters': total_count, 'max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_reserved_bytes': torch.cuda.max_memory_reserved(),
        'responses_sha256': legacy.digest('responses.jsonl'),
        'canonical_sha256': legacy.digest('canonical.jsonl'),
        'plan_sha256': legacy.digest('plan.json'), 'source_sha256': legacy.digest(__file__),
        'raw_ids_saved_before_decode': True, 'raw_response_saved_before_parse': True,
        'gold_read': False, 'quality_calls': 0}
    save('summary.json', result)
    save('complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', {'exception': type(exc).__name__, 'message': str(exc),
            'source_sha256': legacy.digest(__file__), 'progress': progress,
            'incomplete_no_quality': True})
        raise
