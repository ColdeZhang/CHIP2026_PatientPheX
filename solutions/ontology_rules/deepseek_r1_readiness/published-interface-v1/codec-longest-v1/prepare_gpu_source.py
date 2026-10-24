from pathlib import Path
import json,hashlib,shutil,difflib
C=Path(__file__).parent;P=C.parent;old=P/'gpu';G=C/'gpu';cpu=json.loads((C/'actual-cpu-result.json').read_text());assert cpu['status']=='PASS_FIXED_CODEC_CPU_ONLY_ON_EXISTING_FIRST746'
def change(s,a,b,count=1):assert s.count(a)==count,(a,s.count(a));return s.replace(a,b)
s=(old/'run_probe.py').read_text();original=s
s=change(s,'Published tokenizer first+longest native R1 format/cost pilot; max4096, no quality.','New fixed codec longest-only native R1 format/cost; fresh prompt/RNG, no quality.')
s=change(s,'from sparse_final import parse_sparse,tests as sparse_tests','from sparse_final import parse_sparse,tests as sparse_tests\nfrom normalized_final import parse_normalized,tests as normalized_tests')
s=change(s,"plan['process_budget_seconds']==480","plan['process_budget_seconds']==240")
a="""        rows=[json.loads(x) for x in checked(plan['selected_prompts']).read_text().splitlines()];encoded=json.loads(checked(plan['selected_encoded']).read_text())['rows'];assert 1<=len(rows)==len(encoded)==len(plan['row_ids'])<=2
        assert [x['row_id'] for x in rows]==plan['row_ids']==[x['row_id'] for x in encoded]==[x['row_id'] for x in cpu['selected']]
        assert [x['native_input_tokens'] for x in encoded]==plan['input_tokens']==[x['native_input_tokens'] for x in cpu['selected']]
        assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact'] and cpu['selected_indices']==plan['selected_CPU_indices']
        report['fixed_selected_row_count']=len(rows);report['fresh_prompt_not_old_KV_or_RNG_continuation']=True
"""
b="""        rows=[json.loads(x) for x in checked(plan['selected_prompts']).read_text().splitlines()];encoded=json.loads(checked(plan['selected_encoded']).read_text())['rows'];assert len(rows)==len(encoded)==2
        assert [x['row_id'] for x in rows]==plan['published_CPU_pair_row_ids']==[x['row_id'] for x in encoded]==[x['row_id'] for x in cpu['selected']]
        assert [x['native_input_tokens'] for x in encoded]==[x['native_input_tokens'] for x in cpu['selected']]==[2014,5689]
        assert cpu['all80_raw_generic_native_IDs_exact'] and cpu['all80_roundtrip_exact'] and cpu['selected_indices']==[0,73] and plan['selected_pair_index']==1
        rows=rows[1:2];encoded=encoded[1:2]
        assert [x['row_id'] for x in rows]==plan['row_ids']==['6063015:relations:0'] and plan['input_tokens']==[5689] and plan['selected_CPU_indices']==[73]
        codec_receipt=json.loads(checked(plan['codec_CPU_result']).read_text());assert codec_receipt['status']=='PASS_FIXED_CODEC_CPU_ONLY_ON_EXISTING_FIRST746' and codec_receipt['codec_sha256']==plan['sources']['normalized_final.py']
        report['fixed_selected_row_count']=1;report['fresh_prompt_not_old_KV_or_RNG_continuation']=True;report['first_sample_not_regenerated']=True
"""
s=change(s,a,b)
s=change(s,"report['sparse_parser_tests']=sparse_tests();","report['sparse_parser_tests']=sparse_tests();report['normalized_parser_tests']=normalized_tests();")
s=change(s,'external480s cap.','external240s cap.')
s=change(s,'soft_deadline=start+465','soft_deadline=start+225')
s=change(s,"report['soft_deadline_seconds_from_start']=465","report['soft_deadline_seconds_from_start']=225")
a="""            except (ValueError,TypeError) as e:item['legal_sparse_valid']=False;item['sparse_invalid_reason']=str(e)
"""
b="""            except (ValueError,TypeError) as e:item['legal_sparse_valid']=False;item['sparse_invalid_reason']=str(e)
            try:
                normalized=parse_normalized(content,eos_observed=eos,stopped_at_cap=cap or stopper.wall_stop,code_map=row['code_map'],patient_ids=row['patient_ids']);item['normalized_valid']=True;item['normalized_evidence']=normalized
                if item['strict_final_valid']:assert normalized['provided_mapping']==item['parsed_final']
            except (ValueError,TypeError) as e:item['normalized_valid']=False;item['normalized_invalid_reason']=str(e)
"""
s=change(s,a,b)
s=change(s,"'strict_final_valid','legal_sparse_valid']","'strict_final_valid','legal_sparse_valid','normalized_valid']")
s=change(s,"if not item['legal_sparse_valid']:","if not item['normalized_valid']:")
s=change(s,"report['stop_reason']=item['sparse_invalid_reason']","report['stop_reason']=item['normalized_invalid_reason']")
s=change(s,'PASS_PUBLISHED_NATIVE_LEGAL_SPARSE_FORMAT_COST_ONLY_NOT_QUALITY','PASS_PUBLISHED_NORMALIZED_LONGEST_FORMAT_COST_ONLY_NOT_QUALITY')
s=change(s,'A4096-token or480-second failure','A4096-token or240-second failure')
compile(s,'run_probe.py','exec');(G/'run_probe.py').write_text(s);(G/'run_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='published-oldpolicy/run_probe.py',tofile='newcodec-longest/run_probe.py',n=0)))
s=(old/'launch_probe.py').read_text();original=s
for a,b in [('480s+TERM30','240s+TERM30'),("plan['process_budget_seconds']==480","plan['process_budget_seconds']==240"),('cap_seconds=480','cap_seconds=240'),('only_published_first_and_longest=True','only_published_longest_new_codec=True'),("'480s'","'240s'")]:s=change(s,a,b)
compile(s,'launch_probe.py','exec');(G/'launch_probe.py').write_text(s);(G/'launch_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='published-oldpolicy/launch_probe.py',tofile='newcodec-longest/launch_probe.py',n=0)))
for n in ['strict_final.py','sparse_final.py']:shutil.copy2(old/n,G/n)
shutil.copy2(C/'normalized_final.py',G/'normalized_final.py')
plan=json.loads((old/'plan.json').read_text());plan.update(status='NEW_FIXED_CODEC_LONGEST_ONLY_FRESH_RNG_PILOT_READY_NO_GPU_START',process_budget_seconds=240,published_CPU_pair_row_ids=plan['row_ids'],selected_pair_index=1,row_ids=[plan['row_ids'][1]],input_tokens=[5689],selected_CPU_indices=[73],codec_CPU_result={'path':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1/codec-longest-v1/cpu-result.json','bytes':(C/'actual-cpu-result.json').stat().st_size,'sha256':hashlib.sha256((C/'actual-cpu-result.json').read_bytes()).hexdigest()})
plan['sources']={n:hashlib.sha256((G/n).read_bytes()).hexdigest() for n in ['run_probe.py','launch_probe.py','strict_final.py','sparse_final.py','normalized_final.py']}
plan['scope']='Newdistinctlongest-only(index73,5689publishedtokens),first746reusedCPUonly;freshprompt+fixedseed,nooldKV/RNGcontinuation.4096/240+TERM30/internal225softstop. Fixedwholefence/canonicalintPIDcodec,missingCIDabstention. Stopcap/invalid/noEOS;no furtherwidening/gold/full80quality.'
plan['authorization']='Root explicit afternewfixedcodecCPU5valid23negative+frozenfirstonlyCPU:one freshlongest,4096,240+TERM30. Oldpolicy/rawverdictremainunchanged.'
plan['future_authorization']='No full80quality, furthercodecwidening or anotherGPUdiagnostic implied.'
plan['parent_published_oldpolicy_terminal']={'path':str(P/'terminal-manifest.json'),'sha256':hashlib.sha256((P/'terminal-manifest.json').read_bytes()).hexdigest()}
(G/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(G/'source-check.json').write_text(json.dumps({'status':'SOURCE_LOCAL_COMPILE_PASS_CPU_CODEC_RECEIPT_BOUND','sources':plan['sources'],'plan_sha256':hashlib.sha256((G/'plan.json').read_bytes()).hexdigest(),'existing_first_CPU_result_sha256':plan['codec_CPU_result']['sha256'],'no_new_GPU_or_first_NN':True},indent=2)+'\n');print((G/'source-check.json').read_text())
