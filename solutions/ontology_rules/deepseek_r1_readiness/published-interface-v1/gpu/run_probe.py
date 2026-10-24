"""Published tokenizer first+longest native R1 format/cost pilot; max4096, no quality."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,time,traceback
from strict_final import parse_final,tests
from sparse_final import parse_sparse,tests as sparse_tests

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
def checked(r):
    p=Path(r['path']);assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'];return p

def atomic(p,value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def ids_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();planpath=Path(a.plan).resolve();base=planpath.parent;plan=json.loads(planpath.read_text());start=time.monotonic();out=base/'terminal';out.mkdir(exist_ok=False)
    report=dict(status='running_format_cost_only',started_at=now(),source=ref(__file__),plan=ref(planpath),rows=[],quality=False,original80_not_inferred=True)
    torch=None
    try:
        assert plan['execution_ready'] is True and plan['process_budget_seconds']==480 and plan['TERM_grace_seconds']==30
        assert plan['max_new_tokens']==4096 and plan['seed']==20261024 and plan['temperature']==.6 and plan['top_p']==.95 and plan['top_k']==0
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and os.getuid()==1006
        for name,sha in plan['sources'].items():assert digest(base/name)==sha
        assets=json.loads(checked(plan['assets_result']).read_text());expected=json.loads(checked(plan['expected_files']).read_text());cpu=json.loads(checked(plan['CPU_result']).read_text());yarn=json.loads(checked(plan['yarn_result']).read_text())
        assert assets['status']=='ASSET_FILES_AND_ACTUAL_HEADERS_VERIFIED' and assets['MS_revision']==plan['MS_revision'] and cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT' and yarn['status']=='PASS_CPU_ROTARY_FUNCTION_REFERENCE_EQUIVALENCE'
        inventory=json.loads(checked(plan['header_inventory']).read_text());assert inventory['tensor_keys']==399 and inventory['full_bundle_parameters']==8190735360
        actual={x['name']:x for x in assets['files']};modeldir=Path(plan['model_dir']);assert str(modeldir)==assets['model_dir']
        stat_inventory={}
        for e in expected['files']:
            x=actual[e['name']];p=modeldir/e['name'];assert x['status'] in ['REUSED_FULL_SHA_MATCH','DOWNLOADED_FULL_SHA_MATCH'] and x['sha256']==e['sha256'] and x['bytes']==e['bytes']==p.stat().st_size
            if not e['name'].endswith('.safetensors'):assert digest(p)==e['sha256']
            stat_inventory[e['name']]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino,'full_SHA_source':x['sha256']}
        report['asset_integrity_policy']='Full weight SHA from immutable just-completed asset receipt; probe checks size/inode/mtime stability, rehashes small configs, verifies native loading and all parameter keys/shapes/dtypes. No second16GB SHA scan falsely claimed.'
        report['asset_stat_before']=stat_inventory
        rows=[json.loads(x) for x in checked(plan['selected_prompts']).read_text().splitlines()];encoded=json.loads(checked(plan['selected_encoded']).read_text())['rows'];assert 1<=len(rows)==len(encoded)==len(plan['row_ids'])<=2
        assert [x['row_id'] for x in rows]==plan['row_ids']==[x['row_id'] for x in encoded]==[x['row_id'] for x in cpu['selected']]
        assert [x['native_input_tokens'] for x in encoded]==plan['input_tokens']==[x['native_input_tokens'] for x in cpu['selected']]
        assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact'] and cpu['selected_indices']==plan['selected_CPU_indices']
        report['fixed_selected_row_count']=len(rows);report['fresh_prompt_not_old_KV_or_RNG_continuation']=True
        import torch as torch_module,transformers,tokenizers
        torch=torch_module
        from transformers import PreTrainedTokenizerFast,AutoConfig,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        report['runtime']={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'GPU':torch.cuda.get_device_name(0),'uuid':plan['gpu_uuid'],'precision':'native_BF16_SDPA_no_quantization','TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32}
        tc=json.loads((modeldir/'tokenizer_config.json').read_text())
        def token_content(n):
            x=tc.get(n);return x.get('content') if isinstance(x,dict) else x
        published_raw=tokenizers.Tokenizer.from_file(str(modeldir/'tokenizer.json'))
        tok=PreTrainedTokenizerFast(tokenizer_file=str(modeldir/'tokenizer.json'),bos_token=token_content('bos_token'),eos_token=token_content('eos_token'),pad_token=token_content('pad_token'),unk_token=token_content('unk_token'),chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tc['model_max_length'])
        raw_backend=json.loads(published_raw.to_str());chosen_backend=json.loads(tok.backend_tokenizer.to_str());assert all(raw_backend[k]==chosen_backend[k] for k in ['model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'])
        derived_path=checked(plan['derived_config']);derived=json.loads(derived_path.read_text());original_config=json.loads((modeldir/'config.json').read_text());expected_derived=json.loads(json.dumps(original_config));expected_derived['rope_scaling']['attention_factor']=1.0;assert derived==expected_derived
        load_config=AutoConfig.from_pretrained(str(derived_path),local_files_only=True,trust_remote_code=False);assert load_config.rope_parameters['attention_factor']==1.0
        report['tokenizer_policy']='Published JSON raw/generic full backend identical; selected input IDs and roundtrip checked against new all80CPU contract. No AutoTokenizer Llama reconstruction.'
        report['explicit_derived_config']=plan['derived_config'];report['two_interface_fixes_not_single_variable_causal_claim']=True
        for row,enc in zip(rows,encoded):
            rendered=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True);ids=tok.encode(rendered,add_special_tokens=False)
            assert rendered==enc['rendered_prompt'] and ids==enc['input_ids']==published_raw.encode(rendered,add_special_tokens=False).ids and ids_sha(ids)==enc['native_ids_sha256'] and all(k in row for k in ['code_map','patient_ids'])
            assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==published_raw.decode(ids,skip_special_tokens=False)==rendered
        report['parser_tests']=tests();report['sparse_parser_tests']=sparse_tests();report['input_preflight_seconds']=time.monotonic()-start;atomic(out/'progress.json',dict(report,phase='loading_native_model'))
        load_start=time.monotonic()
        model,loading=AutoModelForCausalLM.from_pretrained(modeldir,config=load_config,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',output_loading_info=True)
        model.eval();torch.cuda.synchronize();report['load_seconds']=time.monotonic()-load_start;report['loading_information']={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        shapes=json.loads(checked(plan['derived_shapes']).read_text());parameters=dict(model.named_parameters());assert set(parameters)==set(shapes) and len(parameters)==399
        finite_start=time.monotonic();total=0
        for name,p in parameters.items():
            e=shapes[name];assert list(p.shape)==e['shape_derived_from_config_and_native_source'] and p.numel()==e['numel'] and p.dtype==torch.bfloat16 and p.device.type=='cuda' and bool(torch.isfinite(p).all());total+=p.numel()
        assert total==8190735360 and model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr()
        # Native FP32 nonpersistent buffers may be initialized/copied on a different device.
        # Preserve the CPU-v4.51 byte proof; compare actual GPU math with a same-device native reference.
        import inspect
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        rotary=model.model.rotary_emb;inv=rotary.inv_freq.detach();original=rotary.original_inv_freq.detach()
        assert inv.dtype==original.dtype==torch.float32 and inv.device.type=='cuda' and inv.shape==original.shape==(64,), 'Unexpected native rotary dtype/device/shape; do not silently cast or repair'
        assert bool(torch.isfinite(inv).all()) and bool((inv>0).all()) and torch.equal(inv,original)
        native_fn=ROPE_INIT_FUNCTIONS['yarn'];assert digest(inspect.getsourcefile(native_fn))==yarn['native_full_source_sha256']
        ref_gpu,scale_gpu=native_fn(model.config,device=inv.device);ref_cpu,scale_cpu=native_fn(model.config,device=torch.device('cpu'))
        assert ref_gpu.dtype==ref_cpu.dtype==torch.float32 and ref_gpu.device==inv.device and ref_gpu.shape==ref_cpu.shape==inv.shape
        assert bool(torch.isfinite(ref_gpu).all()) and bool((ref_gpu>0).all()) and bool(torch.isfinite(ref_cpu).all()) and bool((ref_cpu>0).all())
        cpu_hash=hashlib.sha256(ref_cpu.numpy().tobytes()).hexdigest();assert cpu_hash==yarn['native_inv_freq_bytes_sha256']
        inv_cpu=inv.cpu();gpu_cpu=ref_gpu.cpu();rtol=8*torch.finfo(torch.float32).eps;atol=0.0
        def differences(a,b):return {'maximum_abs':float((a-b).abs().max()),'maximum_relative':float(((a-b).abs()/b.abs()).max()),'exact':bool(torch.equal(a,b))}
        report['actual_rotary']={'dtype':str(inv.dtype),'shape':list(inv.shape),'device':str(inv.device),'inv_freq_bytes_sha256':hashlib.sha256(inv_cpu.numpy().tobytes()).hexdigest(),'attention_scaling':float(rotary.attention_scaling),'native_GPU_reference_bytes_sha256':hashlib.sha256(gpu_cpu.numpy().tobytes()).hexdigest(),'CPU_reference_bytes_sha256':cpu_hash,'actual_vs_same_device_native':differences(inv,ref_gpu),'GPU_native_vs_CPU_native':differences(gpu_cpu,ref_cpu),'actual_vs_CPU_native':differences(inv_cpu,ref_cpu),'rtol':rtol,'atol':atol,'tolerance_basis':'8 FP32 eps relative for elementary pow/div/mul evaluation across devices; no absolute floor, no dtype casting or model/config change','native_function_source_sha256':yarn['native_full_source_sha256']}
        assert torch.allclose(inv,ref_gpu,rtol=rtol,atol=atol) and torch.allclose(gpu_cpu,ref_cpu,rtol=rtol,atol=atol)
        assert float(rotary.attention_scaling)==float(scale_gpu)==float(scale_cpu)==cpu['native_CPU_attention_scaling']==plan['derived_attention_scaling']==1.0
        report['parameters']={'tensors':len(parameters),'count':total,'all_actual_shapes_dtype_finite':True,'full_value_comparison_to_files_not_performed':True,'untied_embedding_and_lm_head':True};report['weight_shape_finite_seconds']=time.monotonic()-finite_start
        # A fixed internal wall stop leaves15seconds for serialization/release under the external480s cap.
        soft_deadline=start+465;torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);generation_config={'do_sample':True,'temperature':.6,'top_p':.95,'top_k':0,'repetition_penalty':1.,'max_new_tokens':4096,'num_return_sequences':1,'use_cache':True,'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'return_dict_in_generate':True,'output_scores':False}
        report['generation_arguments']=generation_config;report['soft_deadline_seconds_from_start']=465
        if time.monotonic()>=soft_deadline:raise TimeoutError('Load/validation exhausted fixed probe budget before generation')
        for row_number,(row,enc) in enumerate(zip(rows,encoded)):
            assert time.monotonic()<soft_deadline
            ids=torch.tensor([enc['input_ids']],device='cuda',dtype=torch.long);mask=torch.ones_like(ids);prefix_len=ids.shape[1];gen_start=time.monotonic()
            class BudgetAndEvidence(StoppingCriteria):
                def __init__(self):self.wall_stop=False;self.last_count=0
                def __call__(self,input_ids,scores,**kwargs):
                    count=input_ids.shape[1]-prefix_len;self.wall_stop=time.monotonic()>=soft_deadline
                    if count-self.last_count>=32 or self.wall_stop:
                        values=input_ids[0,prefix_len:].tolist();atomic(out/('row%d.partial.json'%row_number),{'row_id':row['row_id'],'generated_ids':values,'generated_tokens':len(values),'seconds':time.monotonic()-gen_start,'wall_stop':self.wall_stop});self.last_count=count
                    return self.wall_stop
            stopper=BudgetAndEvidence();torch.cuda.synchronize()
            with torch.inference_mode():generated=model.generate(input_ids=ids,attention_mask=mask,stopping_criteria=StoppingCriteriaList([stopper]),**generation_config)
            torch.cuda.synchronize();elapsed=time.monotonic()-gen_start;full=generated.sequences[0].tolist();assert full[:prefix_len]==enc['input_ids'];output_ids=full[prefix_len:];assert len(output_ids)<=4096
            eos=bool(output_ids and output_ids[-1]==tok.eos_token_id);cap=not eos and len(output_ids)>=4096;raw=tok.decode(output_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);content_ids=output_ids[:-1] if eos else output_ids;content=tok.decode(content_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            item={'row_id':row['row_id'],'selection_role':enc['selection_role'],'input_tokens':prefix_len,'generated_ids':output_ids,'generated_tokens':len(output_ids),'native_EOS_observed':eos,'stopped_at_token_cap':cap,'stopped_at_wall_budget':stopper.wall_stop,'raw_generated_text':raw,'final_text_excluding_terminal_EOS':content,'generation_seconds':elapsed,'tokens_per_second_including_prefill':len(output_ids)/elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'quality_evaluated':False}
            item['think_open_count']=raw.count('<think>');item['think_close_count']=raw.count('</think>');item['first_think_close_character_offset']=raw.find('</think>')
            try:item['parsed_final']=parse_final(content,eos_observed=eos,stopped_at_cap=cap or stopper.wall_stop,code_map=row['code_map'],patient_ids=row['patient_ids']);item['strict_final_valid']=True
            except (ValueError,TypeError) as e:item['strict_final_valid']=False;item['strict_invalid_reason']=str(e)
            try:
                sparse=parse_sparse(content,eos_observed=eos,stopped_at_cap=cap or stopper.wall_stop,code_map=row['code_map'],patient_ids=row['patient_ids']);item['legal_sparse_valid']=True;item.update(sparse)
                assert sparse['strict_every_CID']==item['strict_final_valid']
                if item['strict_final_valid']:assert item['parsed_final']==sparse['provided_mapping']
            except (ValueError,TypeError) as e:item['legal_sparse_valid']=False;item['sparse_invalid_reason']=str(e)
            atomic(out/('row%d.json'%row_number),item);report['rows'].append({'path':str(out/('row%d.json'%row_number)),**ref(out/('row%d.json'%row_number)),**{k:item[k] for k in ['row_id','input_tokens','generated_tokens','native_EOS_observed','stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','tokens_per_second_including_prefill','strict_final_valid','legal_sparse_valid']}})
            atomic(out/'progress.json',dict(report,phase='fixed_row_complete'));del generated,ids,mask
            if not item['legal_sparse_valid']:
                report['status']='STOPPED_FIRST_TRUNCATED_OR_INVALID_FINAL_THIS_BUDGET_ONLY';report['stop_reason']=item['sparse_invalid_reason'];break
        else:report['status']='PASS_PUBLISHED_NATIVE_LEGAL_SPARSE_FORMAT_COST_ONLY_NOT_QUALITY'
        report['full_generation_IDs_preserved']=True;report['limitation']='A4096-token or480-second failure only characterizes this fixed budget. No claim this model cannot improve task quality; root decides any separate larger-budget experiment.'
        report['TF32_end']={'matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn':torch.backends.cudnn.allow_tf32}
        report['peak_memory']={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        for name,before in stat_inventory.items():
            p=modeldir/name;assert p.stat().st_size==before['bytes'] and p.stat().st_mtime_ns==before['mtime_ns'] and p.stat().st_ino==before['inode']
        for name,sha in plan['sources'].items():assert digest(base/name)==sha
        assert digest(planpath)==report['plan']['sha256']
    except BaseException as e:report.update(status='FAILED_ENGINEERING_OR_RESOURCE_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        report.update(finished_at=now(),program_seconds=time.monotonic()-start,process_pid=os.getpid());atomic(out/'result.json',report);print(json.dumps({k:report.get(k) for k in ['status','program_seconds','error']}),flush=True)
if __name__=='__main__':main()
