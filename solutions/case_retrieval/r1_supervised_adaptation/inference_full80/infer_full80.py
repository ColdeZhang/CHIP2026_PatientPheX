"""One fixed80 R1-adapted inference:79 fresh, one exact final-adapter cache; no quality."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback
from probe_contract import check_sources, checked, completed_binding, fixed_inputs, ids_sha, ref, sha
from raw_first import atomic, persist_fresh
from full80_cache import check_cache, copy_cached, fsync_chain, write_responses, compress_responses


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--deadline-monotonic', type=float, required=True)
    args = parser.parse_args(); plan_path = args.plan.resolve(); plan = json.loads(plan_path.read_text())
    start = time.monotonic(); output = plan_path.parent / 'actual'; output.mkdir(exist_ok=False); fsync_chain(output, Path(plan['project_root']))
    report = dict(status='RUNNING_FIXED80_FINAL_ADAPTER', started_utc=datetime.now(timezone.utc).isoformat(),
                  source=ref(__file__), plan=ref(plan_path), new_generations=0, quality_calls=0,
                  cached_generations=0, expected_rows=80, completed_rows=0, torch_model_loaded=False)
    stage = 'metadata_source_and_completed_epoch_binding'; torch = None
    def terminate(*_):
        raise TimeoutError('Authorized900second full80 deadline reached')
    signal.signal(signal.SIGTERM, terminate)
    try:
        assert plan['execution_ready'] and plan['adapter_binding'] is not None
        assert plan['budget_seconds'] == 900 and plan['grace_seconds'] == 30 and plan['soft_stop_seconds'] == 870
        assert plan['max_new_tokens'] == 4096 and plan['seed'] == 20261024
        assert plan['temperature'] == .6 and plan['top_p'] == .95 and plan['top_k'] == 0
        assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        check_sources(plan, plan_path.parent)
        binding = completed_binding(plan, plan['training_directory']); assert binding == plan['adapter_binding']
        rows, encoded, audits = fixed_inputs(plan)
        cached = check_cache(plan, rows, encoded, binding)
        manifest = json.loads(checked(binding['model_manifest']).read_text())
        model_directory = Path(plan['model_dir']); before = {}
        for name, expected in plan['model_files'].items():
            path = model_directory / name; stat = path.stat()
            assert not path.is_symlink() and stat.st_size == expected['bytes']
            assert stat.st_mtime_ns == manifest['asset_files'][name]['mtime_ns']
            if not name.endswith('.safetensors'):
                assert sha(path) == expected['sha256']
            before[name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino,
                                sha256_from_completed_formal_full_hash=expected['sha256'])
        report['asset_integrity'] = 'Reuse completed formal full-base SHA receipt plus unchanged bytes/mtime; rehash small configs and every final checkpoint member. Validate all actual base tensor shapes/dtypes/finite. No extra16GB base SHA scan claimed.'
        report['base_asset_stats'] = before; report['adapter_binding'] = binding
        import torch as torch_module
        import transformers
        import tokenizers
        import peft
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, StoppingCriteria, StoppingCriteriaList
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        from full80_contract import parse_item
        torch = torch_module
        assert torch.__version__ == '2.7.1+cu126' and transformers.__version__ == '5.20.2'
        assert tokenizers.__version__ == '0.23.1' and peft.__version__ == '0.21.2'
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        report['runtime'] = dict(torch=torch.__version__, transformers=transformers.__version__,
                                 tokenizers=tokenizers.__version__, peft=peft.__version__,
                                 GPU=torch.cuda.get_device_name(0), precision='BF16 base + exact saved FP32 LoRA')
        stage = 'published_native_all80_input_contract'
        config_json = json.loads((model_directory / 'tokenizer_config.json').read_text())
        def special(name):
            value = config_json.get(name)
            return value.get('content') if isinstance(value, dict) else value
        native = tokenizers.Tokenizer.from_file(str(model_directory / 'tokenizer.json'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_directory / 'tokenizer.json'),
            bos_token=special('bos_token'), eos_token=special('eos_token'), pad_token=special('pad_token'),
            unk_token=special('unk_token'), chat_template=config_json['chat_template'],
            clean_up_tokenization_spaces=False, model_max_length=config_json['model_max_length'])
        raw_backend, generic_backend = json.loads(native.to_str()), json.loads(tokenizer.backend_tokenizer.to_str())
        assert all(raw_backend[key] == generic_backend[key] for key in ['model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens'])
        assert tokenizer.eos_token_id == native.token_to_id('<｜end▁of▁sentence｜>') == 151645
        assert native.token_to_id('<｜Assistant｜>') == 151670
        for row, item, audit in zip(rows, encoded, audits):
            rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            native_chat = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, return_dict=False)
            assert ids == native_chat == native.encode(rendered, add_special_tokens=False).ids == item['input_ids']
            assert ids_sha(ids) == audit['native_ids_sha256']
            assert hashlib.sha256(rendered.encode()).hexdigest() == audit['native_prompt_sha256']
            assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == native.decode(ids, skip_special_tokens=False) == rendered
            assert ids[-1] == 151670
        report['all80_published_native_generic_chat_IDs_and_roundtrip_exact'] = True
        derived_path = checked(plan['derived_config']); derived = json.loads(derived_path.read_text())
        expected = json.loads((model_directory / 'config.json').read_text())
        expected['rope_scaling']['attention_factor'] = 1.0; assert derived == expected
        config = AutoConfig.from_pretrained(str(derived_path), local_files_only=True, trust_remote_code=False)
        stage = 'fresh_base_load'; load_start = time.monotonic()
        base, loading = AutoModelForCausalLM.from_pretrained(model_directory, config=config, dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='sdpa', local_files_only=True,
            trust_remote_code=False, output_loading_info=True)
        assert not any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        assert type(base).__name__ == 'Qwen3ForCausalLM' and sum(p.numel() for p in base.parameters()) == 8190735360
        shapes = json.loads(checked(plan['derived_shapes']).read_text()); parameters = dict(base.named_parameters())
        assert set(parameters) == set(shapes) and len(parameters) == 399
        for name, parameter in parameters.items():
            assert list(parameter.shape) == shapes[name]['shape_derived_from_config_and_native_source']
            assert parameter.dtype == torch.bfloat16 and parameter.device.type == 'cuda' and bool(torch.isfinite(parameter).all())
        assert base.get_input_embeddings().weight.data_ptr() != base.get_output_embeddings().weight.data_ptr()
        rotary = base.model.rotary_emb; native_rotary = ROPE_INIT_FUNCTIONS['yarn']
        assert sha(inspect.getsourcefile(native_rotary)) == plan['rotary']['native_function_source_sha256']
        gpu_ref, gpu_scale = native_rotary(base.config, device=rotary.inv_freq.device)
        cpu_ref, cpu_scale = native_rotary(base.config, device=torch.device('cpu'))
        assert rotary.inv_freq.dtype == gpu_ref.dtype == cpu_ref.dtype == torch.float32
        assert rotary.inv_freq.shape == gpu_ref.shape == cpu_ref.shape == (64,)
        assert torch.equal(rotary.inv_freq, rotary.original_inv_freq)
        assert hashlib.sha256(cpu_ref.numpy().tobytes()).hexdigest() == plan['rotary']['CPU_reference_bytes_sha256']
        assert torch.allclose(rotary.inv_freq, gpu_ref, rtol=8 * torch.finfo(torch.float32).eps, atol=0)
        assert torch.allclose(gpu_ref.cpu(), cpu_ref, rtol=8 * torch.finfo(torch.float32).eps, atol=0)
        assert float(rotary.attention_scaling) == float(gpu_scale) == float(cpu_scale) == 1.0
        stage = 'fresh_saved_FP32_adapter_load_and_value_comparison'
        model = PeftModel.from_pretrained(base, str(checked(binding['adapter']).parent),
            is_trainable=False, autocast_adapter_dtype=True, local_files_only=True)
        saved_adapter = load_file(str(checked(binding['adapter'])), device='cpu')
        actual_adapter = get_peft_model_state_dict(model)
        assert set(saved_adapter) == set(actual_adapter) and len(saved_adapter) == 504
        assert sum(tensor.numel() for tensor in saved_adapter.values()) == 43646976
        for name, tensor in saved_adapter.items():
            actual = actual_adapter[name]
            assert tensor.dtype == actual.dtype == torch.float32 and tensor.shape == actual.shape
            assert bool(torch.isfinite(tensor).all()) and bool(torch.isfinite(actual).all())
            assert torch.equal(tensor, actual.detach().cpu())
        assert sum(parameter.numel() for parameter in model.parameters()) == 8234382336
        assert not any(parameter.requires_grad for parameter in model.parameters())
        for name, parameter in model.named_parameters():
            assert parameter.dtype == (torch.float32 if '.lora_' in name else torch.bfloat16)
        del saved_adapter, actual_adapter
        model.eval(); model.config.use_cache = True
        assert not model.training and not model.is_gradient_checkpointing
        torch.cuda.synchronize(); report['load_seconds'] = time.monotonic() - load_start
        report['torch_model_loaded'] = True
        report['adapter_verification'] = dict(tensors=504, parameters=43646976, saved_actual_FP32_shapes_values_exact=True,
                                             autocast_adapter_dtype=True, no_merge_or_quantization=True)
        report['actual_rotary'] = dict(dtype=str(rotary.inv_freq.dtype), scaling=float(rotary.attention_scaling),
                                      CPU_sha256=hashlib.sha256(cpu_ref.numpy().tobytes()).hexdigest(), native_tolerance_pass=True)
        atomic(output / 'model-ready.json', report)
        # The launcher supplies one absolute deadline; load/setup does not reset it.
        deadline = args.deadline_monotonic
        if time.monotonic() >= deadline:
            raise TimeoutError('Preflight/load consumed fixed full80 budget before generation')
        generation = dict(do_sample=True, temperature=.6, top_p=.95, top_k=0, repetition_penalty=1.,
                          max_new_tokens=4096, num_return_sequences=1, use_cache=True,
                          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                          return_dict_in_generate=True, output_scores=False)
        report['generation_arguments'] = generation
        report['cached_indices'] = [73]; report['new_indices'] = plan['new_indices']
        report['row_sources'] = []
        items = []; maximum_allocated = maximum_reserved = 0
        def decode(values):
            text = tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert text == native.decode(values, skip_special_tokens=False)
            return text
        for index, (row, item) in enumerate(zip(rows, encoded)):
            stage = 'row_%d' % index
            if index == 73:
                parsed = copy_cached(plan, output, cached, decode)
                report['cached_generations'] += 1
                source_kind = 'exact_final_adapter_probe_cache'
            else:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Fixed full80 deadline reached before next row')
                torch.manual_seed(plan['seed']); torch.cuda.manual_seed_all(plan['seed'])
                ids = torch.tensor([item['input_ids']], device='cuda', dtype=torch.long); mask = torch.ones_like(ids)
                prefix_length = len(item['input_ids']); generation_start = time.monotonic()
                class EvidenceStop(StoppingCriteria):
                    def __init__(self):
                        self.wall_stop = False; self.last_count = 0
                    def __call__(self, input_ids, scores, **kwargs):
                        count = input_ids.shape[1] - prefix_length; self.wall_stop = time.monotonic() >= deadline
                        if count - self.last_count >= 32 or self.wall_stop:
                            partial = input_ids[0, prefix_length:].tolist()
                            atomic(output / ('row%d.partial.json' % index), dict(
                                row_id=row['row_id'], original_index=index, generated_ids=partial,
                                generated_tokens=len(partial), wall_stop=self.wall_stop,
                                seconds=time.monotonic() - generation_start), replace=True)
                            self.last_count = count
                        return self.wall_stop
                stop = EvidenceStop()
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); generation_start = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(input_ids=ids, attention_mask=mask,
                        stopping_criteria=StoppingCriteriaList([stop]), **generation)
                torch.cuda.synchronize(); elapsed = time.monotonic() - generation_start
                sequence = generated.sequences[0].tolist(); values = sequence[prefix_length:]
                eos = bool(values and values[-1] == tokenizer.eos_token_id)
                raw_item = dict(row_id=row['row_id'], original_index=index, fresh_NN=True, seed=plan['seed'],
                    input_ids_sha256=ids_sha(item['input_ids']), input_tokens=prefix_length,
                    generated_ids=values, generated_tokens=len(values), input_prefix_exact=sequence[:prefix_length] == item['input_ids'],
                    generated_length_within_cap=len(values) <= 4096, native_EOS_observed=eos,
                    stopped_at_token_cap=not eos and len(values) >= 4096, stopped_at_wall_budget=stop.wall_stop,
                    generation_seconds=elapsed, tokens_per_second_including_prefill=len(values) / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    adapter_binding=binding, quality_evaluated=False)
                parsed = persist_fresh(output, index, raw_item, decode, parse_item, row)
                report['new_generations'] += 1
                maximum_allocated = max(maximum_allocated, torch.cuda.max_memory_allocated())
                maximum_reserved = max(maximum_reserved, torch.cuda.max_memory_reserved())
                source_kind = 'fresh_final_adapter_generation'
                # Save evidence and counts before deciding whether a stopped row ends this run.
                report['last_row'] = ref(output / ('row%d.json' % index))
                if stop.wall_stop:
                    atomic(output / 'timeout-row-evidence.json', dict(original_index=index,
                        row=report['last_row'], complete_raw_preserved=True, no_quality=True))
                    raise TimeoutError('Fixed full80 wall budget exhausted; raw retained, no retry')
                del generated, ids, mask
            response = dict(parsed)
            if index == 73:
                response.update(fresh_NN=False, original_probe_fresh_NN=parsed['fresh_NN'],
                                cache_provenance=plan['final_adapter_cache'])
            items.append(response)
            report['completed_rows'] = len(items)
            report['row_sources'].append(dict(original_index=index, row_id=row['row_id'], kind=source_kind,
                row=ref(output / ('row%d.json' % index)), normalized_valid=parsed['normalized_valid'],
                ids=ref(output / ('row%d.ids.json' % index)), raw=ref(output / ('row%d.raw.json' % index)),
                strict_final_valid=parsed['strict_final_valid'], native_EOS_observed=parsed['native_EOS_observed']))
            atomic(output / 'progress.json', dict(completed_rows=len(items), new_generations=report['new_generations'],
                cached_generations=report['cached_generations'], last_index=index,
                program_seconds=time.monotonic()-start), replace=True)
        assert report['new_generations'] == 79 and report['cached_generations'] == 1
        assert [item['row_id'] for item in items] == [row['row_id'] for row in rows]
        report['responses'] = write_responses(output / 'responses.jsonl', items)
        report['responses_gzip'] = compress_responses(output / 'responses.jsonl')
        report['responses_uncompressed_sha256'] = report['responses']['sha256']
        report['response_row_identity_policy'] = 'All rowN files preserve original immutable provenance. The combined row73 response wrapper marks fresh_NN=False, original_probe_fresh_NN=True and binds the exact cache provenance; row_sources and row73.cache-use identify reuse.'
        report.update(normalized_valid_rows=sum(item['normalized_valid'] for item in items),
            strict_valid_rows=sum(item['strict_final_valid'] for item in items),
            natural_EOS_rows=sum(item['native_EOS_observed'] for item in items),
            token_cap_rows=sum(item['stopped_at_token_cap'] for item in items),
            invalid_whole_row_abstentions=sum(not item['normalized_valid'] for item in items),
            provided_CIDs=sum(len(item.get('normalized_evidence', {}).get('provided_mapping', {})) for item in items),
            total_CIDs=sum(len(row['code_map']) for row in rows), missing_CIDs_are_negative=False,
            no_semantic_repair_or_missing_CID_zero_fill=True, peak_allocated_bytes=maximum_allocated,
            peak_reserved_bytes=maximum_reserved)
        for name, original in before.items():
            stat = (model_directory / name).stat()
            assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (original['bytes'], original['mtime_ns'], original['inode'])
        assert ref(plan_path) == report['plan']; check_sources(plan, plan_path.parent)
        report['status'] = 'COMPLETE_FIXED80_FINAL_ADAPTER_79FRESH_1CACHE_NO_QUALITY'
        report['scope'] = 'All80 F1 inputs complete. Original normalized codec retained; invalid/missing rows abstain, never fill negative. Quality assembly and scoring are separate.'
    except BaseException as error:
        report.update(status='FAILED_FIXED80_PRESERVED_NO_RETRY', stage=stage,
                      error_type=type(error).__name__, error=str(error)[:3000], traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(), program_seconds=time.monotonic() - start)
        atomic(output / 'result.json', report)
        fsync_chain(output, Path(plan['project_root']))
        print(json.dumps({key: report.get(key) for key in ['status', 'completed_rows', 'new_generations',
                                                        'cached_generations', 'program_seconds', 'error_type']}), flush=True)


if __name__ == '__main__':
    main()
