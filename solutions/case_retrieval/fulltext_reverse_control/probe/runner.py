"""One CPU-selected fulltext-reverse input with original Qwen greedy/strict policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback
import legacy_runner as legacy
from codec import parse_complete
from common import checked, check_sources, ref, save, fsync_chain


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--soft-deadline-monotonic', type=float, required=True)
    args = parser.parse_args(); started = time.monotonic(); plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text()); output = plan_path.parent / 'actual'
    output.mkdir(exist_ok=False); fsync_chain(output, Path(plan['project_root']))
    report = dict(status='RUNNING_ONE_FULLTEXT_REVERSE_PROBE', source=ref(__file__), plan=ref(plan_path),
                  started_utc=datetime.now(timezone.utc).isoformat(), gold_read=False, quality_calls=0,
                  fresh_rows=0, original_index=plan['probe_index'], all80_automatically_started=False)
    stage = 'source_CPU_input_and_assets'
    def terminate(*_): raise TimeoutError('Fixed240second probe deadline')
    signal.signal(signal.SIGTERM, terminate)
    try:
        assert plan['execution_ready'] and plan['cap_seconds'] == 240 and plan['term_grace_seconds'] == 30
        assert plan['soft_stop_seconds'] == 210 and plan['generation_limit'] == 1600
        assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        check_sources(plan, plan_path.parent)
        cpu = json.loads(checked(plan['CPU_result']).read_text())
        assert cpu['status'] == 'PASS_CPU_FIXED_FULLTEXT_REVERSE_NO_MODEL_NO_QUALITY'
        assert cpu['rows'] == 80 and cpu['all_inverse_and_suffix_bytes_exact'] and cpu['raw_Auto_template_IDs_and_roundtrip_all80']
        assert cpu['selected_longest_index'] == plan['probe_index']
        assert cpu['input_cap_ceil512'] == plan['input_limit'] == ((cpu['maximum_input_tokens'] + 511) // 512) * 512
        row_audit = json.loads(checked(plan['CPU_audit']).read_text())[plan['probe_index']]
        assert row_audit == plan['row_audit']
        rows = legacy.read_prompts(checked(plan['probe_input'])); assert len(rows) == 1
        row = rows[0]; assert jsha(row) == row_audit['reversed_row_sha256']
        assert row['row_id'] == cpu['selected_row_id'] == row_audit['row_id']
        verified_base = legacy.verify_model(plan['model'], checked(plan['model_source_manifest']))
        verified_adapter = {}
        for name, digest in plan['adapter_files'].items():
            path = Path(plan['adapter']) / name
            assert legacy.digest(path) == digest
            verified_adapter[name] = {'bytes': path.stat().st_size, 'sha256': digest}
        import torch, transformers, peft, tokenizers
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig, StoppingCriteria, StoppingCriteriaList
        from transformers.generation.utils import GenerationMixin
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        from peft import PeftModel
        assert transformers.__version__ == '5.20.2' and peft.__version__ == '0.21.2'
        assert torch.__version__ == '2.7.1+cu126' and tokenizers.__version__ == '0.23.1'
        assert importlib.util.find_spec('fla') is None
        assert legacy.digest(q.__file__) == plan['runtime_sha256']['qwen_modeling']
        assert legacy.digest(inspect.getsourcefile(GenerationMixin)) == plan['runtime_sha256']['generation_utils']
        bindings = {name: dict(module=getattr(q,name).__module__, unwrapped=inspect.unwrap(getattr(q,name)).__module__)
                    for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']}
        assert all(v['module'] == v['unwrapped'] == q.__name__ for v in bindings.values())
        torch.set_num_threads(1)
        # Reproduce actual normal-probe flags; R1's cudnn=False is not this recipe.
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        native = tokenizers.Tokenizer.from_file(str(Path(plan['model']) / 'tokenizer.json'))
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False); expected = row_audit['reverse']
        assert ids == native.encode(rendered, add_special_tokens=False).ids
        assert ids == tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == native.decode(ids, skip_special_tokens=False) == rendered
        assert len(ids) == expected['tokens'] == cpu['maximum_input_tokens'] <= plan['input_limit']
        assert jsha(ids) == expected['input_ids_sha256']
        assert hashlib.sha256(rendered.encode()).hexdigest() == expected['rendered_sha256']
        text_config = Qwen3_5TextConfig.from_pretrained(plan['model'], local_files_only=True)
        assert text_config.max_position_embeddings >= plan['input_limit'] + plan['generation_limit']
        save(output / 'actual-input-and-assets.json', dict(row_id=row['row_id'], original_index=plan['probe_index'],
            input_tokens=len(ids), input_ids_sha256=jsha(ids), rendered_sha256=expected['rendered_sha256'],
            max_position_embeddings=text_config.max_position_embeddings, base_files=verified_base,
            adapter_files=verified_adapter, runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
            peft=peft.__version__, tokenizers=tokenizers.__version__, actual_tokenizer_class=type(tokenizer).__name__,
            matmul_TF32=False, cudnn_TF32=True, native_GDN_bindings=bindings, FLA_installed=False),
            no_truncation=True, gold_read=False, normal_A800_reverse_A30_disclosure=plan['hardware_disclosure']))
        stage = 'original_Qwen_text_only_BF16_SDPA_load'
        loading_started = time.monotonic()
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            plan['model'], config=text_config, dtype=torch.bfloat16, device_map='cuda:0',
            attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
        save(output / 'base-loading.json', loading)
        assert not any(loading.get(key) for key in ['missing_keys', 'mismatched_keys', 'error_msgs'])
        assert all(key.startswith(('mtp.', 'model.mtp.', 'model.visual.', 'visual.'))
                   for key in loading.get('unexpected_keys', []))
        base_count = sum(parameter.numel() for parameter in model.parameters())
        assert base_count == legacy.TEXT_PARAMETERS
        model = PeftModel.from_pretrained(model, plan['adapter'], is_trainable=False)
        model.eval()
        assert type(model.get_base_model()).__name__ == 'Qwen3_5ForCausalLM'
        assert model.get_base_model().config._attn_implementation == 'sdpa'
        total_count = sum(parameter.numel() for parameter in model.parameters())
        assert total_count == 8_997_081_600 < 10_000_000_000
        loading_seconds = time.monotonic() - loading_started
        configured_eos = model.generation_config.eos_token_id
        eos_ids = set(configured_eos if isinstance(configured_eos, list) else [configured_eos])
        eos_ids.add(tokenizer.eos_token_id); eos_ids.discard(None)
        assert sorted(eos_ids) == [248044, 248046] and tokenizer.pad_token_id == 248044
        save(output / 'generation-config.json', dict(actual=model.generation_config.to_dict(),
            configured_eos_ids=sorted(eos_ids), actual_model_class=type(model.get_base_model()).__name__,
            actual_attention_implementation=model.get_base_model().config._attn_implementation,
            do_sample=False, use_cache=True, max_new_tokens=1600, no_explicit_logits_override=True))
        encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda:0')
        assert encoded['input_ids'][0].tolist() == ids
        assert time.monotonic() < args.soft_deadline_monotonic
        stage = 'one_longest_greedy_generation'; generation_started = time.monotonic()
        class EvidenceStop(StoppingCriteria):
            def __init__(self): self.last_count=0; self.wall_stop=False
            def __call__(self, input_ids, scores, **kwargs):
                count=input_ids.shape[1]-len(ids); self.wall_stop=time.monotonic()>=args.soft_deadline_monotonic
                if count-self.last_count>=32 or self.wall_stop:
                    values=input_ids[0,len(ids):].tolist()
                    save(output/'partial-generated-ids.json',dict(row_id=row['row_id'], original_index=plan['probe_index'],
                        generated_ids=values, generated_tokens=len(values), wall_stop=self.wall_stop,
                        seconds=time.monotonic()-generation_started),replace=True)
                    self.last_count=count
                return self.wall_stop
        stop=EvidenceStop(); torch.cuda.synchronize(); generation_started=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(**encoded, do_sample=False, use_cache=True,
                max_new_tokens=plan['generation_limit'], eos_token_id=sorted(eos_ids),
                pad_token_id=tokenizer.pad_token_id, stopping_criteria=StoppingCriteriaList([stop]))
        torch.cuda.synchronize(); seconds=time.monotonic()-generation_started
        generated_ids=generated[0,len(ids):].tolist()
        natural_eos=bool(generated_ids and generated_ids[-1] in eos_ids)
        hit_cap=len(generated_ids)>=plan['generation_limit'] and not natural_eos
        prefix_exact=generated[0,:len(ids)].tolist()==ids
        save(output/'generated-ids.json',dict(row_id=row['row_id'],original_index=plan['probe_index'],
            input_ids_sha256=jsha(ids),generated_ids=generated_ids,configured_eos_ids=sorted(eos_ids),
            natural_eos=natural_eos,hit_token_limit=hit_cap,wall_stop=stop.wall_stop,
            generation_seconds=seconds,input_prefix_exact=prefix_exact))
        report['fresh_rows']=1
        assert prefix_exact and len(generated_ids)<=plan['generation_limit']
        response=tokenizer.decode(generated_ids,skip_special_tokens=True)
        native_response=native.decode(generated_ids,skip_special_tokens=True)
        with_special=tokenizer.decode(generated_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        native_with_special=native.decode(generated_ids,skip_special_tokens=False)
        save(output/'raw-response.json',dict(row_id=row['row_id'],pmc_id=row['pmc_id'],original_index=plan['probe_index'],
            response=response,native_response=native_response,raw_with_special_tokens=with_special,
            native_raw_with_special_tokens=native_with_special,input_tokens=len(ids),output_tokens=len(generated_ids),
            natural_eos=natural_eos,hit_token_limit=hit_cap,wall_stop=stop.wall_stop,batch_seconds=seconds,removed_demonstrations=0))
        assert response==native_response and with_special==native_with_special
        parsed=error=None
        try:
            if not natural_eos or hit_cap or stop.wall_stop: raise ValueError('No natural EOS or generation reached cap/deadline')
            parsed=parse_complete(response,row)
        except (ValueError,TypeError) as exc: error=type(exc).__name__+': '+str(exc)
        save(output/'format-result.json',dict(strict_complete=parsed is not None,error=error,parsed=parsed,
            whole_row_abstention=parsed is None,missing_CID_negative_fill=False,fallback=False,gold_read=False,quality_calls=0))
        report.update(status='PASS_NATIVE_LONGEST_STRICT_COMPLETE' if parsed is not None else 'FORMAT_FAILURE_AFTER_RAW_SAVED',
            row_id=row['row_id'],input_tokens=len(ids),generated_tokens=len(generated_ids),generation_seconds=seconds,
            loading_seconds=loading_seconds,base_parameters=base_count,total_parameters=total_count,
            strict_complete=parsed is not None,natural_eos=natural_eos,hit_token_limit=hit_cap,wall_stop=stop.wall_stop,
            max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
            raw_ids_saved_before_decode=True,raw_response_saved_before_parse=True,raw_native_decode_exact=True)
        save(output/'format-cost-summary.json',report)
        assert parsed is not None,error
        assert ref(plan_path)==report['plan'];check_sources(plan,plan_path.parent)
    except BaseException as error:
        report.update(status='FAILED_PROBE_PRESERVED_NO_RETRY',stage=stage,error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(output/'summary.json',report);fsync_chain(output,Path(plan['project_root']))
        print(json.dumps({key:report.get(key) for key in ['status','strict_complete','generation_seconds','elapsed_seconds']}),flush=True)


if __name__ == '__main__': main()
