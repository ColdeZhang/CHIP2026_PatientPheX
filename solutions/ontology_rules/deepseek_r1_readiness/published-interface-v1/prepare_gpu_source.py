from pathlib import Path
import json,hashlib,shutil,difflib
P=Path(__file__).parent;D=P.parent;old=D/'gpu-probe-single4096';G=P/'gpu';C=P/'cpu';cpu=json.loads((C/'actual-result.json').read_text());assert cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT'
def change(s,a,b,count=1):assert s.count(a)==count,(a,s.count(a));return s.replace(a,b)
s=(old/'run_probe.py').read_text();original=s
s=change(s,'One frozen first native R1 generation with4096-token cap; no continuation or quality.','Published tokenizer first+longest native R1 format/cost pilot; max4096, no quality.')
s=change(s,'from strict_final import parse_final,tests','from strict_final import parse_final,tests\nfrom sparse_final import parse_sparse,tests as sparse_tests')
s=change(s,"plan['process_budget_seconds']==240","plan['process_budget_seconds']==480")
s=change(s,"cpu['status'].startswith('PASS_actual_native_R1_')","cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT'")
a="""        rows=[json.loads(x) for x in checked(plan['selected_prompts']).read_text().splitlines()];encoded=json.loads(checked(plan['selected_encoded']).read_text())['rows'];assert len(rows)==len(encoded)==3
        assert [x['row_id'] for x in rows]==plan['CPU_frozen_source_row_ids']==[x['row_id'] for x in encoded]
        assert [x['native_input_tokens'] for x in encoded]==[2171,3425,6002]
        assert plan['selected_CPU_index']==0
        rows=rows[:1];encoded=encoded[:1]
        assert [x['row_id'] for x in rows]==plan['row_ids']==['10313222:relations:0'] and plan['input_tokens']==[2171]
        report['fixed_selected_row_count']=1;report['fresh_prompt_not_old_KV_or_RNG_continuation']=True
"""
b="""        rows=[json.loads(x) for x in checked(plan['selected_prompts']).read_text().splitlines()];encoded=json.loads(checked(plan['selected_encoded']).read_text())['rows'];assert 1<=len(rows)==len(encoded)==len(plan['row_ids'])<=2
        assert [x['row_id'] for x in rows]==plan['row_ids']==[x['row_id'] for x in encoded]==[x['row_id'] for x in cpu['selected']]
        assert [x['native_input_tokens'] for x in encoded]==plan['input_tokens']==[x['native_input_tokens'] for x in cpu['selected']]
        assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact'] and cpu['selected_indices']==plan['selected_CPU_indices']
        report['fixed_selected_row_count']=len(rows);report['fresh_prompt_not_old_KV_or_RNG_continuation']=True
"""
s=change(s,a,b)
s=change(s,'from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList','from transformers import PreTrainedTokenizerFast,AutoConfig,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList')
s=change(s,"        tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)","""        tc=json.loads((modeldir/'tokenizer_config.json').read_text())
        def token_content(n):
            x=tc.get(n);return x.get('content') if isinstance(x,dict) else x
        published_raw=tokenizers.Tokenizer.from_file(str(modeldir/'tokenizer.json'))
        tok=PreTrainedTokenizerFast(tokenizer_file=str(modeldir/'tokenizer.json'),bos_token=token_content('bos_token'),eos_token=token_content('eos_token'),pad_token=token_content('pad_token'),unk_token=token_content('unk_token'),chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tc['model_max_length'])
        raw_backend=json.loads(published_raw.to_str());chosen_backend=json.loads(tok.backend_tokenizer.to_str());assert all(raw_backend[k]==chosen_backend[k] for k in ['model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'])
        derived_path=checked(plan['derived_config']);derived=json.loads(derived_path.read_text());original_config=json.loads((modeldir/'config.json').read_text());expected_derived=json.loads(json.dumps(original_config));expected_derived['rope_scaling']['attention_factor']=1.0;assert derived==expected_derived
        load_config=AutoConfig.from_pretrained(str(derived_path),local_files_only=True,trust_remote_code=False);assert load_config.rope_parameters['attention_factor']==1.0
        report['tokenizer_policy']='Published JSON raw/generic full backend identical; selected input IDs and roundtrip checked against new all80CPU contract. No AutoTokenizer Llama reconstruction.'
        report['explicit_derived_config']=plan['derived_config'];report['two_interface_fixes_not_single_variable_causal_claim']=True""")
s=change(s,"            assert rendered==enc['rendered_prompt'] and ids==enc['input_ids'] and ids_sha(ids)==enc['native_ids_sha256'] and all(k in row for k in ['code_map','patient_ids'])","            assert rendered==enc['rendered_prompt'] and ids==enc['input_ids']==published_raw.encode(rendered,add_special_tokens=False).ids and ids_sha(ids)==enc['native_ids_sha256'] and all(k in row for k in ['code_map','patient_ids'])\n            assert tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==published_raw.decode(ids,skip_special_tokens=False)==rendered")
s=change(s,"report['parser_tests']=tests();","report['parser_tests']=tests();report['sparse_parser_tests']=sparse_tests();")
s=change(s,"modeldir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'}","modeldir,config=load_config,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'}")
s=change(s,"assert float(rotary.attention_scaling)==float(scale_gpu)==float(scale_cpu)==yarn['native_attention_scaling']","assert float(rotary.attention_scaling)==float(scale_gpu)==float(scale_cpu)==cpu['native_CPU_attention_scaling']==plan['derived_attention_scaling']==1.0")
s=change(s,'external240s cap.','external480s cap.')
s=change(s,'soft_deadline=start+225','soft_deadline=start+465')
s=change(s,"report['soft_deadline_seconds_from_start']=225","report['soft_deadline_seconds_from_start']=465")
a="""            except (ValueError,TypeError) as e:item['strict_final_valid']=False;item['invalid_reason']=str(e)
"""
b="""            except (ValueError,TypeError) as e:item['strict_final_valid']=False;item['strict_invalid_reason']=str(e)
            try:
                sparse=parse_sparse(content,eos_observed=eos,stopped_at_cap=cap or stopper.wall_stop,code_map=row['code_map'],patient_ids=row['patient_ids']);item['legal_sparse_valid']=True;item.update(sparse)
                assert sparse['strict_every_CID']==item['strict_final_valid']
                if item['strict_final_valid']:assert item['parsed_final']==sparse['provided_mapping']
            except (ValueError,TypeError) as e:item['legal_sparse_valid']=False;item['sparse_invalid_reason']=str(e)
"""
s=change(s,a,b)
s=change(s,"'tokens_per_second_including_prefill','strict_final_valid']","'tokens_per_second_including_prefill','strict_final_valid','legal_sparse_valid']")
s=change(s,"if not item['strict_final_valid']:","if not item['legal_sparse_valid']:")
s=change(s,"report['stop_reason']=item['invalid_reason']","report['stop_reason']=item['sparse_invalid_reason']")
s=change(s,'PASS_SINGLE_NATIVE_FINAL_FORMAT_COST_ONLY_NOT_QUALITY','PASS_PUBLISHED_NATIVE_LEGAL_SPARSE_FORMAT_COST_ONLY_NOT_QUALITY')
s=change(s,'A4096-token or240-second failure','A4096-token or480-second failure')
compile(s,'run_probe.py','exec');(G/'run_probe.py').write_text(s);(G/'run_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='oldTF5/run_probe.py',tofile='published/run_probe.py',n=0)))
s=(old/'launch_probe.py').read_text();original=s
for a,b in [('240s+TERM30','480s+TERM30'),("plan['process_budget_seconds']==240","plan['process_budget_seconds']==480"),('cap_seconds=240','cap_seconds=480'),('only_original_fixed_first_row=True','only_published_first_and_longest=True'),("'240s'","'480s'")]:s=change(s,a,b)
compile(s,'launch_probe.py','exec');(G/'launch_probe.py').write_text(s);(G/'launch_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='oldTF5/launch_probe.py',tofile='published/launch_probe.py',n=0)));shutil.copy2(old/'strict_final.py',G/'strict_final.py')
plan=json.loads((old/'plan.json').read_text());plan.update(status='PUBLISHED_INTERFACE_FIRST_LONGEST_PILOT_READY_NO_GPU_STARTED',process_budget_seconds=480,CPU_result={'path':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1/cpu/actual/result.json','bytes':(C/'actual-result.json').stat().st_size,'sha256':hashlib.sha256((C/'actual-result.json').read_bytes()).hexdigest()},selected_prompts=cpu['outputs']['selected-prompts.jsonl'],selected_encoded=cpu['outputs']['selected-encoded.json'],derived_config=cpu['outputs']['derived-config.json'],derived_attention_scaling=1.0,row_ids=[x['row_id'] for x in cpu['selected']],input_tokens=[x['native_input_tokens'] for x in cpu['selected']],selected_CPU_indices=cpu['selected_indices'])
for k in ['CPU_frozen_source_row_ids','selected_CPU_index']:plan.pop(k,None)
plan['sources']={n:hashlib.sha256((G/n).read_bytes()).hexdigest() for n in ['run_probe.py','launch_probe.py','strict_final.py','sparse_final.py']}
plan['scope']='Only first+actualnewlongest (deduplicate),2014/5689 currentpublishedinput; raw/genericIDscontract. Fresh model/promptseed,4096each,total480+TERM30/internal465softstop. Firstcap/numerical/budget/invalidsparse stops. Legal sparse may continue whileEveryCIDstrictfalse; missingCIDabstention neverfill[]. Noquality/full80/forceclose/retry.'
plan['authorization']='Root conditional explicit afternewCPU complete+freshGPU1: once first/newlongest,4096each,480+TERM30; strictEveryCID andseparatelegalsparse evidence. Two public-interface fixes not causaltest.'
plan['future_authorization']='No full80quality or furtherbudget authorization inthisplan.'
plan['parent_reconstructed_tokenizer_4096_terminal']={'path':str(old/'terminal-manifest.json'),'sha256':hashlib.sha256((old/'terminal-manifest.json').read_bytes()).hexdigest()}
plan['CPU_length_statistic_note']='Recorded3131 isupper-middle sorted[40]; conventionalmedian(3127+3131)/2=3129. Selectionmaxindex73 unchanged.'
(G/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
import sys;sys.path.insert(0,str(G));import sparse_final
(G/'source-check.json').write_text(json.dumps({'status':'LOCAL_SOURCES_COMPILE_AND_SPARSE_CONTRACT_SYNTHETIC_PASS','parser_checks':sparse_final.tests(),'sources':plan['sources'],'plan_sha256':hashlib.sha256((G/'plan.json').read_bytes()).hexdigest(),'no_GPU_new_generation_or_gold':True},indent=2)+'\n');print((G/'source-check.json').read_text())
