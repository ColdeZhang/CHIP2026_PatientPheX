"""Verify complete FLA234 evidence without interpreting any response text."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from common import FLOWS, checked, read, ref, rows, same


def relative_member(root, name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and str(p) == name
    full = root.joinpath(*p.parts)
    assert full.resolve().is_relative_to(root.resolve()) and full.is_file() and not full.is_symlink()
    return full


def verify(plan):
    actual = plan['actual']; root = Path(actual['root']).resolve()
    same(actual['plan'], plan['expected_gpu_plan'])
    gpu = read(checked(actual['plan'])); terminal = read(checked(actual['terminal']))
    assert gpu['schema'] == 'fixed-originaldev901-all-four-FLA-fresh234-v1' and gpu['execution_ready']
    assert gpu['flow_order'] == list(FLOWS)
    assert (gpu['cap_seconds'], gpu['soft_stop_seconds'], gpu['term_grace_seconds']) == (1800, 1770, 30)
    assert gpu['expected_rows'] == gpu['fresh_rows'] == 234 and gpu['cached_original_indices'] == []
    assert gpu['input_limit'] == 14000 and gpu['generation_limit'] == 1600
    assert terminal['status'] == 'COMPLETE_ORIGINAL901_FOUR_FRESH_FLA_FLOWS_NO_QUALITY'
    assert terminal['exit_code'] == 0 and terminal['timeout_triggered'] is False
    assert terminal['cached_rows'] == 0 and terminal['expected_rows'] == 234
    assert terminal['gold_read'] is False and terminal['quality_calls'] == 0
    assert terminal['flow_order'] == list(FLOWS) and set(terminal['flows']) == set(FLOWS)
    same(terminal['plan'], actual['plan'])
    assert terminal['source']['sha256'] == gpu['sources']['launch.py']['sha256']
    assert terminal['cap_seconds'] == 1800 and terminal['soft_stop_seconds'] == 1770 and terminal['term_grace_seconds'] == 30
    assert set(actual['files']) == set(terminal['files'])
    for name, identity in terminal['files'].items():
        path = relative_member(root, name); same(ref(path), identity)
        assert checked(actual['files'][name]) == path
    for name, expected in gpu['sources'].items(): same(ref(relative_member(root, name)), expected)
    same(ref(root/'plan.json'), actual['plan'])
    def load(name):
        assert name in actual['files']; return read(checked(actual['files'][name]))
    def local_identity(remote, name):
        assert name in actual['files']; same(remote, actual['files'][name])
        return checked(actual['files'][name])
    model = load('model-source-manifest.json')
    model_files = {x['name']: dict(bytes=x['size'], sha256=x['sha256']) for x in model['files']}
    same(actual['files']['model-source-manifest.json'], gpu['model_source_manifest'])
    coverage = {}; total = 0; record_files = set()
    for name in FLOWS:
        flow = gpu['flows'][name]; prefix = 'actual/' + name + '/'
        summary = load(prefix+'summary.json'); complete = load(prefix+'complete.json')
        assert all(summary[k] == v for k, v in complete.items())
        assert summary['status'] == 'COMPLETE_FRESH_ORIGINAL901_FLOW_NO_QUALITY' and summary['flow'] == name
        assert summary['rows'] == summary['fresh_rows'] == summary['completed_rows'] == flow['rows']
        assert summary['cached_rows'] == summary['parsed_rows'] == summary['quality_calls'] == 0
        assert summary['gold_read'] is False and summary['raw_before_parser'] and summary['original_decoder_and_order_preserved']
        assert summary['base_parameters'] == 8953803264 and summary['total_parameters'] == 8997081600
        same(summary['plan'], actual['plan']); same(summary['source'], gpu['sources']['infer_flow.py'])
        flow_terminal = terminal['flows'][name]
        assert flow_terminal['exit_code'] == 0 and flow_terminal['deadline_not_reset']
        local_identity(flow_terminal['summary'], prefix+'summary.json')
        raw_path = local_identity(summary['responses'], prefix+'responses.jsonl')
        records_path = local_identity(summary['record_manifest'], prefix+'record-manifest.json')
        local_identity(summary['backend'], prefix+'backend.json')
        local_identity(summary['input_and_assets'], prefix+'input-and-assets.json')
        stem = 'public_new' if name == 'public-new' else name
        prompt_ref = plan['fixed_references'][stem+'_prompts']
        same(prompt_ref, flow['prompts']); prompts = rows(checked(prompt_ref))
        native_name = name + '.native.json'; audit = load(native_name)
        same(actual['files'][native_name], flow['native_audit'])
        assert audit['status'] == 'PASS_NATIVE_INPUT_BUDGET' and audit['count'] == flow['rows']
        assert audit['source']['sha256'] == gpu['native_audit_source_sha256']
        same(audit['prompts'], flow['prompts'])
        by_audit = {x['row_id']: x for x in audit['rows']}
        by_prompt = {x['row_id']: x for x in prompts}
        assert len(by_prompt) == len(by_audit) == flow['rows']
        assert set(by_prompt) == set(by_audit)
        order = sorted(by_prompt, key=lambda rid:(by_audit[rid]['input_tokens'],rid))
        assert order == flow['ordered_row_ids']
        assets = load(prefix+'input-and-assets.json')
        assert assets['flow'] == name and assets['rows'] == flow['rows']
        assert assets['actual_native_IDs_verified'] and assets['original_order_exact']
        assert assets['decoder'] == flow['decoder'] and assets['hardware_disclosure'] == gpu['hardware_disclosure']
        same(assets['prompts'], flow['prompts']); same(assets['native_audit'], flow['native_audit'])
        assert assets['input_tokens'] == flow['input_tokens'] == audit['total_input_tokens']
        assert assets['max_input_tokens'] == flow['max_input_tokens'] == audit['maximum_input_tokens'] <= 14000
        assert set(assets['base_files']) == set(model_files)
        for filename, expected in model_files.items(): same(assets['base_files'][filename], expected)
        assert set(assets['adapter_files']) == set(flow['adapter_files'])
        for filename, expected in flow['adapter_files'].items(): same(assets['adapter_files'][filename], expected)
        runtime = assets['runtime']
        assert all(runtime[k] == v for k,v in gpu['runtime'].items())
        assert runtime['conv_native'] and runtime['FLA_GDN']
        generation = load(prefix+'generation.json')
        assert generation['do_sample'] is False and generation['use_cache'] and generation['thinking'] is False
        assert generation['max_new_tokens'] == 1600 and generation['eos_ids'] == [248044,248046]
        assert generation['pad_token_id'] == 248044 and generation['no_manual_seed']
        assert generation['finite_logits_only'] is (name == 'patient')
        assert generation['decoder'] == flow['decoder']
        loading = load(prefix+'loading.json')
        assert not any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
        assert all(k.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for k in loading.get('unexpected_keys',[]))
        backend = load(prefix+'backend.json')
        assert backend['both_GDN_globals_fused'] and backend['conv_native'] and backend['no_native_comparison_calls']
        assert backend['independent_quality_variant_not_numerical_equivalence']
        assert backend['runtime'] == dict(triton='3.3.1',fla_core='0.6.0',einops='0.8.1')
        same(backend['wheels']['fla-core'],gpu['FLA_wheel']); same(backend['wheels']['einops'],gpu['einops_wheel'])
        assert set(backend['fused']) == {'torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'}
        for graph in backend['fused'].values():
            assert any(x['module'].startswith('fla.ops.gated_delta_rule') for x in graph)
        assert not any(x['module'].startswith('causal_conv1d') for graph in backend['conv'].values() for x in graph)
        assert all(backend['installed_sources'][package] for package in ('fla-core','einops'))
        records = read(records_path); raw = rows(raw_path)
        assert len(records) == len(raw) == flow['rows']
        assert [x['row_id'] for x in records] == [x['row_id'] for x in raw] == order
        assert len({x['row_id'] for x in raw}) == flow['rows']
        indices = {p['row_id']:i for i,p in enumerate(prompts)}
        assert assets['processing_original_indices'] == [indices[rid] for rid in order]
        natural_count = 0; cap_count = 0
        for record, response in zip(records, raw):
            rid = record['row_id']; index = indices[rid]; expected = by_audit[rid]
            assert record['fresh'] and record['original_index'] == index
            ids_name = prefix+'generated-ids/'+('%03d.json'%index)
            raw_name = prefix+'raw-responses/'+('%03d.json'%index)
            ids = read(local_identity(record['ids'],ids_name)); saved = read(local_identity(record['raw'],raw_name))
            assert ids_name not in record_files and raw_name not in record_files
            record_files.update((ids_name,raw_name))
            assert saved == response and isinstance(response['response'],str)
            assert ids['row_id'] == response['row_id'] == rid
            assert str(ids['pmc_id']) == str(response['pmc_id']) == str(by_prompt[rid]['pmc_id'])
            assert ids['original_index'] == index
            assert ids['input_ids_sha256'] == expected['native_input_ids_sha256']
            assert ids['input_tokens'] == response['input_tokens'] == expected['input_tokens'] <= 14000
            values = ids['generated_token_ids']
            assert isinstance(values,list) and len(values) <= 1600 and all(type(v) is int and v >= 0 for v in values)
            assert len(values) == ids['output_tokens'] == response['output_tokens']
            natural = bool(values and values[-1] in (248044,248046)); cap = bool(len(values)>=1600 and not natural)
            assert ids['ended_with_eos'] is response['ended_with_eos'] is natural
            assert ids['hit_token_limit'] is response['hit_token_limit'] is cap
            assert ids['eos_ids'] == [248044,248046]
            assert ids['wall_stop'] is response['wall_stop'] is False
            assert ids['input_prefix_exact'] and ids['raw_ids_before_decode']
            assert response['removed_demonstrations'] == 0
            if name == 'patient': assert response['generated_token_ids'] == values
            # A valid completed capped generation is not rejected: the original
            # CID fallback / limited patient abstention remains in assembly.
            natural_count += natural; cap_count += cap
        assert summary['natural_EOS_rows'] == natural_count and summary['token_cap_rows'] == cap_count
        coverage[name] = dict(rows=len(raw),responses=ref(raw_path),prompts=prompt_ref,
            record_manifest=ref(records_path),ordered_row_ids=order,fresh_rows=len(raw),cached_rows=0,
            natural_EOS_rows=natural_count,token_cap_rows=cap_count,response_content_parsed=False)
        total += len(raw)
    assert total == 234
    return dict(status='PASS_COMPLETE_FRESH234_SOURCE_AND_MEMBER_EVIDENCE_NO_RESPONSE_PARSE',
        flows=coverage,total_rows=total,cached_rows=0,terminal=actual['terminal'],
        verified_member_count=len(actual['files']),verified_row_evidence_files=len(record_files),
        original_termination_parser_policies_unchanged=True,
        hardware_disclosure=gpu['hardware_disclosure'])
