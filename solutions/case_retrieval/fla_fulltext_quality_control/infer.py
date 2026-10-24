"""Independent FLA fulltext-normal variant: all80 fresh, no native cache/quality."""
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
import legacy_runner as legacy
from codec import parse_complete
from common import checked, check_sources, ref, save, fsync_chain
from full80_io import jsonl_with_gzip
from backend import verify_fla_backend


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--soft-deadline-monotonic',type=float,required=True)
    args=parser.parse_args();started=time.monotonic();plan_path=args.plan.resolve()
    plan=json.loads(plan_path.read_text());output=plan_path.parent/'actual'
    output.mkdir(exist_ok=False);fsync_chain(output,Path(plan['project_root']))
    report=dict(status='RUNNING_FIXED80_FULLTEXT_FLA',source=ref(__file__),plan=ref(plan_path),
        started_utc=datetime.now(timezone.utc).isoformat(),gold_read=False,quality_calls=0,
        fresh_rows=0,cached_rows=0,completed_rows=0,cached_original_indices=[])
    stage='sources_original_normal_CPU_and_assets'
    def terminate(*_):raise TimeoutError('Fixed1200second full80 deadline')
    signal.signal(signal.SIGTERM,terminate)
    try:
        assert plan['execution_ready'] and plan['cap_seconds']==1200 and plan['term_grace_seconds']==30
        assert plan['soft_stop_seconds']==1170 and plan['generation_limit']==1600 and plan['input_limit']==14848
        assert plan['cached_original_indices']==[] and plan['expected_rows']==80
        assert args.soft_deadline_monotonic <= started + 1170
        assert plan['output_order']=='ascending(input_tokens,row_id)'
        assert plan['adapter_files']=={
            'adapter_config.json':'96564441328e6759568e60c49ea87db432b9ec45cfd57001ef81640d4ff9c4f9',
            'adapter_model.safetensors':'4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'}
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        check_sources(plan,plan_path.parent)
        cpu=json.loads(checked(plan['CPU_result']).read_text())
        audit=json.loads(checked(plan['CPU_audit']).read_text())
        rows=legacy.read_prompts(checked(plan['full80_prompts']))
        assert cpu['status']=='PASS_ACTUAL_80_APPEND_ONLY_NATIVE_TOKENIZER_INPUT_CENSUS_NO_MODEL_OR_QUALITY'
        assert plan['full80_prompts']['sha256']=='2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085'
        assert plan['CPU_audit']['sha256']=='35f9a00ecabf0b0114d756d31b13fc5a947155cb3c3ec8f6a36392f1f0674abb'
        assert cpu['rows']==len(rows)==len(audit)==80 and cpu['expanded']['longest_indices']==[19]
        assert cpu['expanded']['total']==670122 and cpu['expanded']['maximum']==14390
        assert len({str(row['pmc_id']) for row in rows})==16
        assert sum(len(row['code_map']) for row in rows)==587
        assert sum(len(row['code_map'])*len(row['patient_ids']) for row in rows)==1557
        for index,(row,expected) in enumerate(zip(rows,audit)):
            assert expected['index']==index and row['row_id']==expected['row_id']
            assert jsha(row)==expected['expanded_row_sha256']
        verified_base=legacy.verify_model(plan['model'],checked(plan['model_source_manifest']))
        verified_adapter={}
        for name,digest in plan['adapter_files'].items():
            path=Path(plan['adapter'])/name;assert legacy.digest(path)==digest
            verified_adapter[name]={'bytes':path.stat().st_size,'sha256':digest}
        import torch, transformers, peft, tokenizers
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig, StoppingCriteria, StoppingCriteriaList
        from transformers.generation.utils import GenerationMixin
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        from peft import PeftModel
        assert transformers.__version__ == '5.20.2' and peft.__version__ == '0.21.2'
        assert torch.__version__ == '2.7.1+cu126' and tokenizers.__version__ == '0.23.1'
        assert legacy.digest(q.__file__) == plan['runtime_sha256']['qwen_modeling']
        assert legacy.digest(inspect.getsourcefile(GenerationMixin)) == plan['runtime_sha256']['generation_utils']
        fused, backend_evidence = verify_fla_backend(plan, q)
        bindings = backend_evidence['fused']
        save(output/'actual-FLA-bindings-and-dependencies.json',backend_evidence)
        report['backend_evidence']=ref(output/'actual-FLA-bindings-and-dependencies.json')
        torch.set_num_threads(1)
        # Reproduce actual normal-probe flags; R1's cudnn=False is not this recipe.
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
        native = tokenizers.Tokenizer.from_file(str(Path(plan['model']) / 'tokenizer.json'))
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        prepared=[]
        for index,row in enumerate(rows):
            rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(rendered,add_special_tokens=False);expected=audit[index]['expanded']
            assert ids==native.encode(rendered,add_special_tokens=False).ids
            assert ids==tokenizer.apply_chat_template(row['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            assert tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(ids,skip_special_tokens=False)==rendered
            assert len(ids)==expected['tokens']<=plan['input_limit']
            assert jsha(ids)==expected['input_ids_sha256']
            assert hashlib.sha256(rendered.encode()).hexdigest()==expected['rendered_sha256']
            prepared.append((len(ids),row['row_id'],index,rendered,ids))
        assert sum(item[0] for item in prepared)==670122 and max(item[0] for item in prepared)==14390
        # Exact original normal9bb2b322 policy: ascending length, then row_id.
        prepared.sort(key=lambda item:(item[0],item[1]))
        report['processing_original_indices']=[item[2] for item in prepared]
        text_config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True)
        assert text_config.max_position_embeddings>=plan['input_limit']+plan['generation_limit']
        save(output/'actual-input-and-assets.json',dict(rows=80,documents=16,candidate_codes=587,
            input_tokens=670122,max_input_tokens=14390,all80_actual_ids_equal_prior_native_CPU=True,
            base_files=verified_base,adapter_files=verified_adapter,max_position_embeddings=text_config.max_position_embeddings,
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                tokenizers=tokenizers.__version__,actual_tokenizer_class=type(tokenizer).__name__,
                matmul_TF32=False,cudnn_TF32=True,FLA_GDN_bindings=bindings,FLA_installed=True,conv_backend='native'),
            cached_original_indices=[],no_truncation=True,gold_read=False,hardware_disclosure=plan['hardware_disclosure']))
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
        for name in ['generated-ids','raw-responses','format-records','canonical-records','partial-generated-ids']:
            (output/name).mkdir(exist_ok=False)
        fsync_chain(output,Path(plan['project_root']))
        records=[];raw_records=[];canonical_records=[];generation_seconds_total=0.
        save(output/'cache-use.json',dict(cached_rows=0,all80_fresh=True,native_responses_forbidden=True))
        for length,_,index,rendered,ids in prepared:
            stage='row_%d'%index;row=rows[index]
            if time.monotonic()>=args.soft_deadline_monotonic:
                raise TimeoutError('Fixed full80 soft deadline reached before next row')
            assert all(getattr(q,name) is fn for name,fn in fused.items())
            encoded=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to('cuda:0')
            assert encoded['input_ids'][0].tolist()==ids
            generation_started=time.monotonic()
            class EvidenceStop(StoppingCriteria):
                def __init__(self):self.last_count=0;self.wall_stop=False
                def __call__(self,input_ids,scores,**kwargs):
                    count=input_ids.shape[1]-len(ids);self.wall_stop=time.monotonic()>=args.soft_deadline_monotonic
                    if count-self.last_count>=32 or self.wall_stop:
                        values=input_ids[0,len(ids):].tolist()
                        save(output/'partial-generated-ids'/('%03d.json'%index),dict(
                            row_id=row['row_id'],original_index=index,generated_ids=values,generated_tokens=len(values),
                            wall_stop=self.wall_stop,seconds=time.monotonic()-generation_started),replace=True)
                        self.last_count=count
                    return self.wall_stop
            stop=EvidenceStop();torch.cuda.synchronize();generation_started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=False,use_cache=True,
                    max_new_tokens=plan['generation_limit'],eos_token_id=sorted(eos_ids),
                    pad_token_id=tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([stop]))
            torch.cuda.synchronize();seconds=time.monotonic()-generation_started;generation_seconds_total+=seconds
            values=generated[0,len(ids):].tolist();natural_eos=bool(values and values[-1] in eos_ids)
            hit_cap=len(values)>=plan['generation_limit'] and not natural_eos
            prefix_exact=generated[0,:len(ids)].tolist()==ids;wall_stop=stop.wall_stop
            save(output/'generated-ids'/('%03d.json'%index),dict(row_id=row['row_id'],original_index=index,
                input_ids_sha256=jsha(ids),generated_ids=values,configured_eos_ids=sorted(eos_ids),
                natural_eos=natural_eos,hit_token_limit=hit_cap,wall_stop=wall_stop,
                generation_seconds=seconds,input_prefix_exact=prefix_exact))
            report['fresh_rows']+=1
            response=tokenizer.decode(values,skip_special_tokens=True);native_response=native.decode(values,skip_special_tokens=True)
            with_special=tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            native_with_special=native.decode(values,skip_special_tokens=False)
            raw=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],original_index=index,response=response,
                native_response=native_response,raw_with_special_tokens=with_special,
                native_raw_with_special_tokens=native_with_special,input_tokens=len(ids),output_tokens=len(values),
                natural_eos=natural_eos,hit_token_limit=hit_cap,wall_stop=wall_stop,batch_seconds=seconds,removed_demonstrations=0)
            save(output/'raw-responses'/('%03d.json'%index),raw)
            assert prefix_exact and len(values)<=plan['generation_limit']
            assert response==native_response and with_special==native_with_special
            parsed=error=None
            try:
                if not natural_eos or hit_cap or wall_stop:raise ValueError('No natural EOS or generation reached cap/deadline')
                parsed=parse_complete(response,row)
            except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
            save(output/'format-records'/('%03d.json'%index),dict(strict_complete=parsed is not None,error=error,
                parsed=parsed,whole_row_abstention=parsed is None,missing_CID_negative_fill=False,
                fallback=False,gold_read=False,quality_calls=0))
            provenance='fresh'
            del generated,encoded
            canonical=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],candidate_patients=parsed if parsed is not None else {},
                           strict_complete=parsed is not None,abstention=error,provenance=provenance)
            save(output/'canonical-records'/('%03d.json'%index),canonical)
            raw_records.append(raw);canonical_records.append(canonical)
            records.append(dict(original_index=index,row_id=row['row_id'],provenance=provenance,
                ids=ref(output/'generated-ids'/('%03d.json'%index)),raw=ref(output/'raw-responses'/('%03d.json'%index)),
                format=ref(output/'format-records'/('%03d.json'%index)),canonical=ref(output/'canonical-records'/('%03d.json'%index))))
            report['completed_rows']+=1
            save(output/'progress.json',dict(completed=report['completed_rows'],fresh=report['fresh_rows'],cached=report['cached_rows'],
                current_original_index=index,current_row_id=row['row_id'],elapsed_seconds=time.monotonic()-started),replace=True)
            if wall_stop:raise TimeoutError('Fixed full80 wall stop; complete current IDs/raw/canonical retained')
        assert report['fresh_rows']==80 and report['cached_rows']==0 and report['completed_rows']==80
        assert len({row['row_id'] for row in raw_records})==80
        assert {row['row_id'] for row in raw_records}=={row['row_id'] for row in rows}
        assert [row['row_id'] for row in raw_records]==[row['row_id'] for row in canonical_records]
        assert [row['original_index'] for row in records]==report['processing_original_indices']
        responses=jsonl_with_gzip(output/'responses.jsonl',raw_records)
        canonical_output=jsonl_with_gzip(output/'canonical.jsonl',canonical_records)
        save(output/'record-manifest.json',records)
        report.update(status='COMPLETE_FIXED80_SFT32_FULLTEXT_FLA_NO_QUALITY',expected_rows=80,
            strict_complete_rows=sum(row['strict_complete'] for row in canonical_records),
            whole_row_abstentions=sum(not row['strict_complete'] for row in canonical_records),
            natural_EOS_rows=sum(row['natural_eos'] for row in raw_records),
            token_cap_rows=sum(row['hit_token_limit'] for row in raw_records),
            fresh_generation_seconds=generation_seconds_total,loading_seconds=loading_seconds,
            base_parameters=base_count,total_parameters=total_count,max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(),responses=responses,canonical=canonical_output,
            responses_sha256=responses['plain']['sha256'],canonical_sha256=canonical_output['plain']['sha256'],
            record_manifest=ref(output/'record-manifest.json'),input_tokens=670122,max_input_tokens=14390,
            source_sha256=legacy.digest(__file__),plan_sha256=legacy.digest(plan_path),
            raw_ids_saved_before_decode=True,raw_response_saved_before_parse=True,all80_raw_native_decode_exact=True,
            invalid_rows_are_whole_row_abstentions=True,missing_CIDs_are_negative=False,fallback=False)
        assert all(getattr(q,name) is fn for name,fn in fused.items())
        report['FLA_two_GDN_bindings_maintained']=True
        report['backend_variant_not_transparent_replacement']=True
        assert ref(plan_path)==report['plan'];check_sources(plan,plan_path.parent)
        save(output/'complete.json',report)
    except BaseException as error:
        report.update(status='FAILED_FIXED80_PRESERVED_NO_RETRY',stage=stage,error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(output/'summary.json',report);fsync_chain(output,Path(plan['project_root']))
        print(json.dumps({key:report.get(key) for key in ['status','completed_rows','fresh_rows','cached_rows','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
