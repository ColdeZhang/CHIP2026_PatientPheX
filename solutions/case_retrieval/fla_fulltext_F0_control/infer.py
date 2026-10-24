"""Independent F0 FLA fulltext-normal: longest30 gate, all58 fresh, no quality."""
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
    report=dict(status='RUNNING_FIXED58_FULLTEXT_FLA',source=ref(__file__),plan=ref(plan_path),
        started_utc=datetime.now(timezone.utc).isoformat(),gold_read=False,quality_calls=0,
        fresh_rows=0,cached_rows=0,completed_rows=0,cached_original_indices=[],longest_engineering_passed=False)
    stage='sources_original_normal_CPU_and_assets'
    def terminate(*_):raise TimeoutError('Fixed900second full58 deadline')
    signal.signal(signal.SIGTERM,terminate)
    try:
        assert plan['execution_ready'] and plan['cap_seconds']==900 and plan['term_grace_seconds']==30
        assert plan['soft_stop_seconds']==870 and plan['generation_limit']==1600 and plan['input_limit']==16384
        assert plan['cached_original_indices']==[] and plan['expected_rows']==58
        assert args.soft_deadline_monotonic <= started + 870
        assert plan['output_order']=='longest-first-then-ascending(input_tokens,row_id)'
        assert plan['longest_original_index']==30
        assert plan['adapter_files']=={
            'adapter_config.json':'165dbc59c340d68b9756443331cdc1a5f3e81c1834b1237a99d004c33e14cbb9',
            'adapter_model.safetensors':'8e32daeea3c078af25a7d01e9d9cebf1fe8d99bbfaf2ca5e2c172e643d346513'}
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        check_sources(plan,plan_path.parent)
        cpu=json.loads(checked(plan['CPU_result']).read_text())
        audit=json.loads(checked(plan['CPU_audit']).read_text())
        rows=legacy.read_prompts(checked(plan['full58_prompts']))
        assert cpu['status']=='PASS_ACTUAL_58_APPEND_ONLY_NATIVE_TOKENIZER_INPUT_CENSUS_NO_MODEL_OR_QUALITY'
        assert plan['full58_prompts']['sha256']=='d633225256dbf1cd5b64e94c74d7ce7f220b834cd748158082b306f12191fab2'
        assert plan['CPU_audit']['sha256']=='2fa5c495f8341ba4bf5f6ea4f731f75128f159f28ecb6302abaf7ead67872d02'
        assert cpu['rows']==len(rows)==len(audit)==58 and cpu['expanded']['longest_indices']==[30]
        assert cpu['patients']==47 and cpu['candidates']==415 and cpu['pairs']==1166
        assert plan['CPU_result']['sha256']=='ccdcd461fdb8dc7119bc0755ed7c253bcee64c5ae7738c0c53a0259a30517116'
        assert cpu['expanded']['total']==505090 and cpu['expanded']['maximum']==16245
        assert len({str(row['pmc_id']) for row in rows})==16
        assert sum(len(row['code_map']) for row in rows)==415
        assert sum(len(row['code_map'])*len(row['patient_ids']) for row in rows)==1166
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
        report['CUDA_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
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
        assert sum(item[0] for item in prepared)==505090 and max(item[0] for item in prepared)==16245
        # New authorized engineering order: maximum row30 first, then original ascending policy.
        prepared.sort(key=lambda item:(item[0],item[1]))
        longest=[item for item in prepared if item[2]==30]
        assert len(longest)==1 and longest[0][0]==16245
        prepared=longest+[item for item in prepared if item[2]!=30]
        report['processing_original_indices']=[item[2] for item in prepared]
        text_config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True)
        assert text_config.max_position_embeddings>=plan['input_limit']+plan['generation_limit']
        save(output/'actual-input-and-assets.json',dict(rows=58,documents=16,patients=47,candidate_codes=415,
            input_tokens=505090,max_input_tokens=16245,all58_actual_ids_equal_prior_native_CPU=True,
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
        save(output/'cache-use.json',dict(cached_rows=0,all58_fresh=True,native_responses_forbidden=True))
        for length,_,index,rendered,ids in prepared:
            stage='row_%d'%index;row=rows[index]
            if time.monotonic()>=args.soft_deadline_monotonic:
                raise TimeoutError('Fixed full58 soft deadline reached before next row')
            if index!=30:assert report['longest_engineering_passed']
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
            parsed=error=None
            try:
                if not natural_eos or hit_cap or wall_stop:raise ValueError('No natural EOS or generation reached cap/deadline')
                parsed=parse_complete(response,row)
            except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
            save(output/'format-records'/('%03d.json'%index),dict(strict_complete=parsed is not None,error=error,
                parsed=parsed,whole_row_abstention=parsed is None,missing_CID_negative_fill=False,
                fallback=False,gold_read=False,quality_calls=0))
            if index==30:
                gate=dict(original_index=30,row_id=row['row_id'],input_tokens=len(ids),input_cap=16384,
                    generation_limit=1600,output_tokens=len(values),natural_eos=natural_eos,
                    strict_complete=parsed is not None,hit_token_limit=hit_cap,wall_stop=wall_stop,
                    input_prefix_exact=prefix_exact,native_decode_exact=(response==native_response and with_special==native_with_special),
                    CUDA_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                    max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                    generation_seconds=seconds,fresh_NN=True,old_native_cache_used=False,
                    ids=ref(output/'generated-ids'/('%03d.json'%index)),raw=ref(output/'raw-responses'/('%03d.json'%index)),
                    format=ref(output/'format-records'/('%03d.json'%index)),
                    passed=bool(natural_eos and parsed is not None and not hit_cap and not wall_stop
                        and prefix_exact and len(values)<=1600 and response==native_response and with_special==native_with_special),
                    gold_read=False,quality_calls=0)
                save(output/'longest-engineering-gate.json',gate)
                report['longest_engineering_gate']=ref(output/'longest-engineering-gate.json')
                report['longest_engineering_passed']=gate['passed']
                assert gate['passed'],'Longest F0 row engineering gate failed; preserve raw, no next row or retry'
            assert prefix_exact and len(values)<=plan['generation_limit']
            assert response==native_response and with_special==native_with_special
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
            if wall_stop:raise TimeoutError('Fixed full58 wall stop; complete current IDs/raw/canonical retained')
        assert report['fresh_rows']==58 and report['cached_rows']==0 and report['completed_rows']==58
        assert len({row['row_id'] for row in raw_records})==58
        assert {row['row_id'] for row in raw_records}=={row['row_id'] for row in rows}
        assert [row['row_id'] for row in raw_records]==[row['row_id'] for row in canonical_records]
        assert [row['original_index'] for row in records]==report['processing_original_indices']
        responses=jsonl_with_gzip(output/'responses.jsonl',raw_records)
        canonical_output=jsonl_with_gzip(output/'canonical.jsonl',canonical_records)
        save(output/'record-manifest.json',records)
        report.update(status='COMPLETE_FIXED58_SFT48_FULLTEXT_FLA_NO_QUALITY',expected_rows=58,
            strict_complete_rows=sum(row['strict_complete'] for row in canonical_records),
            whole_row_abstentions=sum(not row['strict_complete'] for row in canonical_records),
            natural_EOS_rows=sum(row['natural_eos'] for row in raw_records),
            token_cap_rows=sum(row['hit_token_limit'] for row in raw_records),
            fresh_generation_seconds=generation_seconds_total,loading_seconds=loading_seconds,
            base_parameters=base_count,total_parameters=total_count,max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(),responses=responses,canonical=canonical_output,
            responses_sha256=responses['plain']['sha256'],canonical_sha256=canonical_output['plain']['sha256'],
            record_manifest=ref(output/'record-manifest.json'),input_tokens=505090,max_input_tokens=16245,
            source_sha256=legacy.digest(__file__),plan_sha256=legacy.digest(plan_path),
            raw_ids_saved_before_decode=True,raw_response_saved_before_parse=True,all58_raw_native_decode_exact=True,
            invalid_rows_are_whole_row_abstentions=True,missing_CIDs_are_negative=False,fallback=False)
        assert all(getattr(q,name) is fn for name,fn in fused.items())
        report['FLA_two_GDN_bindings_maintained']=True
        report['backend_variant_not_transparent_replacement']=True
        assert ref(plan_path)==report['plan'];check_sources(plan,plan_path.parent)
        save(output/'complete.json',report)
    except BaseException as error:
        if 'torch' in locals():
            try:
                report['CUDA_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
                report['failure_max_allocated_bytes']=torch.cuda.max_memory_allocated()
                report['failure_max_reserved_bytes']=torch.cuda.max_memory_reserved()
            except BaseException as memory_error:
                report['failure_memory_observation_error']=type(memory_error).__name__+': '+str(memory_error)
        report.update(status='FAILED_FIXED58_PRESERVED_NO_RETRY',stage=stage,error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(output/'summary.json',report);fsync_chain(output,Path(plan['project_root']))
        print(json.dumps({key:report.get(key) for key in ['status','completed_rows','fresh_rows','cached_rows','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
