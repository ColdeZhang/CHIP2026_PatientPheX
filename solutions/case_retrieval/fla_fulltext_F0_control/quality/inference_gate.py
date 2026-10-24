"""Complete fixed all58 fresh FLA evidence validation without decoding/parsing model text."""
import gzip
import json
from pathlib import Path
from common import ACTUAL_NAMES, COMPLETE, checked, jsha, read, ref, rows, same, unique


def verify(plan):
    refs = plan['endpoint_references']; assert set(refs) == set(ACTUAL_NAMES)
    root = Path(refs['inference_plan']['path']).resolve().parent
    for name, relative in ACTUAL_NAMES.items():
        assert Path(refs[name]['path']).resolve() == root/relative; checked(refs[name])
    fixed, summary, complete, terminal = (read(refs[k]['path']) for k in ('inference_plan','summary','complete','terminal'))
    assert summary['status'] == complete['status'] == COMPLETE
    assert set(summary)-set(complete) == {'finished_utc','elapsed_seconds'}
    assert {k:v for k,v in summary.items() if k not in ('finished_utc','elapsed_seconds')} == complete
    assert terminal['exit_code'] == 0 and not terminal['timeout_triggered'] and terminal['target_UUID_has_no_compute_process']
    assert terminal['status'] == 'COMPLETE_FIXED58_FULLTEXT_FLA_PROCESS'
    assert (terminal['cap_seconds'],terminal['soft_stop_seconds'],terminal['term_grace_seconds']) == (900,870,30)
    assert (summary['fresh_rows'],summary['cached_rows'],summary['completed_rows'],summary['expected_rows']) == (58,0,58,58)
    assert summary['cached_original_indices'] == fixed['cached_original_indices'] == []
    assert summary['quality_calls'] == 0 and summary['gold_read'] is False
    assert summary['raw_ids_saved_before_decode'] and summary['raw_response_saved_before_parse'] and summary['all58_raw_native_decode_exact']
    assert summary['invalid_rows_are_whole_row_abstentions'] and not summary['missing_CIDs_are_negative'] and not summary['fallback']
    assert summary['base_parameters'] == 8953803264 and summary['total_parameters'] == 8997081600
    assert summary['input_tokens'] == 505090 and summary['max_input_tokens'] == 16245
    assert fixed['execution_ready'] and fixed['sources'] == plan['expected_inference_sources']
    assert (fixed['cap_seconds'],fixed['soft_stop_seconds'],fixed['term_grace_seconds']) == (900,870,30)
    assert (fixed['input_limit'],fixed['generation_limit'],fixed['expected_rows']) == (16384,1600,58)
    assert fixed['output_order'] == 'longest-first-then-ascending(input_tokens,row_id)'
    assert fixed['longest_original_index'] == 30
    assert fixed['adapter_files'] == plan['adapter_files']
    assert refs['inference_plan']['sha256'] == plan['expected_inference_plan_sha256']
    assert 'probe_references' not in fixed and 'cached_probe_references' not in fixed
    same(fixed['model_source_manifest'],plan['references']['model_source'])
    for current, old in [(refs['inference_plan'],terminal['plan']),(refs['launcher'],terminal['source']),
                         (refs['inference_plan'],summary['plan']),(refs['inference_source'],summary['source'])]: same(current,old)
    assert summary['plan_sha256'] == refs['inference_plan']['sha256'] and summary['source_sha256'] == refs['inference_source']['sha256']
    files = {}
    for name, expected in terminal['files'].items():
        relative = Path(name); assert not relative.is_absolute() and '..' not in relative.parts
        path = root/relative; assert path.is_file() and not path.is_symlink()
        actual = ref(path); same(actual,expected); files[name] = actual
    for name, expected in fixed['sources'].items(): same(ref(root/name),expected)
    for name in ('CPU_result','CPU_audit','full58_prompts'): same(fixed[name],plan['references'][name])
    use = read(refs['cache_use']['path'])
    assert use == dict(cached_rows=0,all58_fresh=True,native_responses_forbidden=True)
    assets, generation = read(refs['assets']['path']), read(refs['generation']['path'])
    model_source=read(plan['references']['model_source']['path'])
    expected_base={item['name']:{'bytes':item['size'],'sha256':item['sha256']} for item in model_source['files']}
    assert len(expected_base)==15 and assets['base_files']==expected_base
    assert {name:value['sha256'] for name,value in assets['adapter_files'].items()}==plan['adapter_files']
    assert assets['adapter_files']['adapter_model.safetensors']['bytes']==173181072
    assert assets['adapter_files']['adapter_config.json']['bytes']==1287
    assert assets['cached_original_indices']==[] and assets['input_tokens']==505090 and assets['max_input_tokens']==16245
    assert assets['no_truncation'] and assets['all58_actual_ids_equal_prior_native_CPU'] and not assets['gold_read']
    runtime=assets['runtime']
    for name,value in {'torch':'2.7.1+cu126','transformers':'5.20.2','peft':'0.21.2','tokenizers':'0.23.1',
                       'actual_tokenizer_class':'Qwen3_5Tokenizer','matmul_TF32':False,'cudnn_TF32':True,
                       'FLA_installed':True,'conv_backend':'native'}.items(): assert runtime[name]==value
    backend=read(refs['backend']['path']);same(refs['backend'],summary['backend_evidence'])
    assert backend['both_GDN_globals_fused'] and backend['conv_native'] and backend['no_native_comparison_calls']
    assert backend['independent_quality_variant_not_numerical_equivalence']
    assert backend['runtime']==dict(triton='3.3.1',fla_core='0.6.0',einops='0.8.1')
    assert runtime['FLA_GDN_bindings']==backend['fused']
    assert set(backend['fused'])=={'torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'}
    assert all(any(node['module'].startswith('fla.ops.gated_delta_rule') for node in graph) for graph in backend['fused'].values())
    assert not any(node['module'].startswith('causal_conv1d') for graph in backend['conv'].values() for node in graph)
    assert set(backend['installed_sources'])=={'fla-core','einops'} and all(backend['installed_sources'].values())
    for package,key in [('fla-core','FLA_wheel'),('einops','einops_wheel')]:same(backend['wheels'][package],fixed[key])
    assert summary['FLA_two_GDN_bindings_maintained'] and summary['backend_variant_not_transparent_replacement']
    assert generation['configured_eos_ids']==[248044,248046]
    assert generation['do_sample'] is False and generation['use_cache'] is True and generation['max_new_tokens']==1600
    assert generation['no_explicit_logits_override'] and generation['actual_attention_implementation']=='sdpa'
    assert generation['actual_model_class']=='Qwen3_5ForCausalLM'
    prompts = rows(plan['references']['full58_prompts']['path']); audit = read(plan['references']['CPU_audit']['path'])
    cpu = read(plan['references']['CPU_result']['path'])
    assert cpu['status'] == 'PASS_ACTUAL_58_APPEND_ONLY_NATIVE_TOKENIZER_INPUT_CENSUS_NO_MODEL_OR_QUALITY'
    assert cpu['expanded']['total'] == 505090 and cpu['expanded']['maximum'] == 16245
    assert len(prompts) == len(audit) == 58 and len({p['row_id'] for p in prompts}) == 58
    assert len({str(p['pmc_id']) for p in prompts}) == 16
    assert sum(len(p['code_map']) for p in prompts) == 415
    assert sum(len(p['code_map'])*len(p['patient_ids']) for p in prompts) == 1166
    ordered = sorted(range(58),key=lambda i:(audit[i]['expanded']['tokens'],prompts[i]['row_id']))
    assert cpu['expanded']['longest_indices']==[30] and audit[30]['expanded']['tokens']==16245
    ordered=[30]+[index for index in ordered if index!=30]
    assert summary['processing_original_indices'] == ordered
    actual_rows = {}
    for name in ('responses','canonical'):
        compressed = checked(refs[name]); plain = root/'actual'/('%s.jsonl'%name)
        assert gzip.decompress(compressed.read_bytes()) == plain.read_bytes()
        same(refs[name],summary[name]['gzip']); same(ref(plain),summary[name]['plain'])
        assert ref(plain)['sha256'] == summary[name+'_sha256']
        actual_rows[name] = rows(plain)
        assert len(actual_rows[name]) == 58
    longest=read(refs['longest_gate']['path']);same(refs['longest_gate'],summary['longest_engineering_gate'])
    assert summary['longest_engineering_passed'] is True and longest['passed'] is True
    assert longest['original_index']==30 and longest['row_id']==prompts[30]['row_id']
    assert (longest['input_tokens'],longest['input_cap'],longest['generation_limit'])==(16245,16384,1600)
    assert longest['fresh_NN'] and not longest['old_native_cache_used']
    assert longest['natural_eos'] and longest['strict_complete'] and longest['input_prefix_exact'] and longest['native_decode_exact']
    assert not longest['hit_token_limit'] and not longest['wall_stop'] and not longest['gold_read'] and longest['quality_calls']==0
    assert 0<longest['output_tokens']<=1600
    for field,directory in [('ids','generated-ids'),('raw','raw-responses'),('format','format-records')]:
        same(longest[field],files['actual/%s/030.json'%directory])
    records = read(refs['record_manifest']['path']); same(refs['record_manifest'],summary['record_manifest'])
    assert len(records) == 58 and [r['original_index'] for r in records] == ordered
    raw_by_id, canonical_by_id = {}, {}
    for index, record, raw, canonical in zip(ordered,records,actual_rows['responses'],actual_rows['canonical']):
        prompt, row_audit = prompts[index], audit[index]
        assert row_audit['index'] == index and jsha(prompt) == row_audit['expanded_row_sha256']
        assert record['row_id'] == prompt['row_id'] == row_audit['row_id'] == raw['row_id'] == canonical['row_id']
        assert raw['original_index'] == index and raw['pmc_id'] == canonical['pmc_id'] == prompt['pmc_id']
        expected_provenance = 'fresh'
        assert record['provenance'] == canonical['provenance'] == expected_provenance
        for name, directory in [('ids','generated-ids'),('raw','raw-responses'),('format','format-records'),('canonical','canonical-records')]:
            relative = 'actual/%s/%03d.json'%(directory,index)
            same(record[name],files[relative])
            assert Path(record[name]['path']) == Path(terminal['plan']['path']).parent/relative
        ids = read(files['actual/generated-ids/%03d.json'%index]['path'])
        form = read(files['actual/format-records/%03d.json'%index]['path'])
        assert raw == read(files['actual/raw-responses/%03d.json'%index]['path'])
        assert canonical == read(files['actual/canonical-records/%03d.json'%index]['path'])
        assert set(canonical) == {'row_id','pmc_id','candidate_patients','strict_complete','abstention','provenance'}
        assert ids['original_index'] == index and ids['row_id'] == prompt['row_id']
        assert ids['input_ids_sha256'] == row_audit['expanded']['input_ids_sha256']
        assert raw['input_tokens'] == row_audit['expanded']['tokens'] <= 16245
        if index==30:
            assert longest['output_tokens']==raw['output_tokens'] and canonical['strict_complete'] is True
        tokens = ids['generated_ids']; assert all(type(t) is int and t >= 0 for t in tokens)
        assert 0 < len(tokens) == raw['output_tokens'] <= 1600
        assert ids['configured_eos_ids'] == [248044,248046]
        assert raw['natural_eos'] == ids['natural_eos'] == (tokens[-1] in ids['configured_eos_ids'])
        assert raw['hit_token_limit'] == ids['hit_token_limit'] == (len(tokens) >= 1600 and not raw['natural_eos'])
        assert raw['wall_stop'] is False and ids['wall_stop'] is False
        assert raw['response'] == raw['native_response'] and raw['raw_with_special_tokens'] == raw['native_raw_with_special_tokens']
        assert form['strict_complete'] == canonical['strict_complete']
        assert form['error'] == canonical['abstention']
        assert form['whole_row_abstention'] is (not canonical['strict_complete'])
        assert not form['missing_CID_negative_fill'] and not form['fallback'] and not form['gold_read'] and form['quality_calls'] == 0
        assert canonical['candidate_patients'] == (form['parsed'] if form['strict_complete'] else {})
        if not form['strict_complete']: assert form['parsed'] is None
        assert ids['input_prefix_exact']
        raw_by_id[prompt['row_id']] = raw; canonical_by_id[prompt['row_id']] = canonical
    coverage = dict(rows=58,documents=16,patients=47,entities=947,fresh=58,cached=0,
        strict_complete_rows=sum(x['strict_complete'] for x in canonical_by_id.values()),
        whole_row_abstentions=sum(not x['strict_complete'] for x in canonical_by_id.values()),
        natural_EOS_rows=sum(x['natural_eos'] for x in raw_by_id.values()),
        token_cap_rows=sum(x['hit_token_limit'] for x in raw_by_id.values()),
        all_members_verified=len(files),all58_fresh_no_native_response_cache=True,FLA_two_GDN_bindings_maintained=True,
        longest_engineering_passed=True,longest_engineering_gate=refs['longest_gate'],
        raw_text_reparsed=False,native_decode_repeated=False)
    for key in ('strict_complete_rows','whole_row_abstentions','natural_EOS_rows','token_cap_rows'):
        assert coverage[key] == summary[key]
    return prompts,raw_by_id,canonical_by_id,coverage,files
