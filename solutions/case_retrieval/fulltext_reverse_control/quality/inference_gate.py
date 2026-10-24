"""Complete fixed reverse evidence validation without decoding/parsing model text."""
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
    assert terminal['status'] == 'COMPLETE_FIXED80_FULLTEXT_REVERSE_PROCESS'
    assert (terminal['cap_seconds'],terminal['soft_stop_seconds'],terminal['term_grace_seconds']) == (1200,1170,30)
    assert (summary['fresh_rows'],summary['cached_rows'],summary['completed_rows'],summary['expected_rows']) == (79,1,80,80)
    assert summary['cached_original_indices'] == fixed['cached_original_indices'] == [19]
    assert summary['quality_calls'] == 0 and summary['gold_read'] is False
    assert summary['raw_ids_saved_before_decode'] and summary['raw_response_saved_before_parse'] and summary['all80_raw_native_decode_exact']
    assert summary['invalid_rows_are_whole_row_abstentions'] and not summary['missing_CIDs_are_negative'] and not summary['fallback']
    assert summary['base_parameters'] == 8953803264 and summary['total_parameters'] == 8997081600
    assert summary['input_tokens'] == 670117 and summary['max_input_tokens'] == 14390
    assert fixed['execution_ready'] and fixed['sources'] == plan['expected_inference_sources']
    assert (fixed['cap_seconds'],fixed['soft_stop_seconds'],fixed['term_grace_seconds']) == (1200,1170,30)
    assert (fixed['input_limit'],fixed['generation_limit'],fixed['expected_rows']) == (14848,1600,80)
    assert fixed['output_order'] == 'ascending(input_tokens,row_id)'
    assert fixed['adapter_files'] == plan['adapter_files']
    probe_plan=read(plan['cached_probe_references']['plan']['path'])
    for key in ('model','adapter','adapter_files','model_source_manifest','runtime_sha256','runtime','generation',
                'CPU_source_sha256','CPU_contract_sha256','input_limit','generation_limit'):
        assert fixed[key]==probe_plan[key]
    for current, old in [(refs['inference_plan'],terminal['plan']),(refs['launcher'],terminal['source']),
                         (refs['inference_plan'],summary['plan']),(refs['inference_source'],summary['source'])]: same(current,old)
    assert summary['plan_sha256'] == refs['inference_plan']['sha256'] and summary['source_sha256'] == refs['inference_source']['sha256']
    files = {}
    for name, expected in terminal['files'].items():
        relative = Path(name); assert not relative.is_absolute() and '..' not in relative.parts
        path = root/relative; assert path.is_file() and not path.is_symlink()
        actual = ref(path); same(actual,expected); files[name] = actual
    for name, expected in fixed['sources'].items(): same(ref(root/name),expected)
    for name in ('CPU_result','CPU_audit','full80_prompts'): same(fixed[name],plan['references'][name])
    for name, cached in plan['cached_probe_references'].items(): same(cached,fixed['probe_references'][name])
    use = read(refs['cache_use']['path'])
    assert use['original_index'] == 19 and not use['fresh_NN_in_this_run'] and use['exact_original_reverse_probe_bytes']
    assert use['old_normal_probe_forbidden'] and use['provenance'] == fixed['probe_references']
    assets, generation = read(refs['assets']['path']), read(refs['generation']['path'])
    assert assets['base_files'] == read(plan['cached_probe_references']['assets']['path'])['base_files']
    assert assets['adapter_files'] == read(plan['cached_probe_references']['assets']['path'])['adapter_files']
    assert assets['runtime'] == read(plan['cached_probe_references']['assets']['path'])['runtime']
    assert assets['no_truncation'] and assets['all80_actual_ids_equal_prior_native_CPU'] and not assets['gold_read']
    assert generation == read(plan['cached_probe_references']['generation']['path'])
    prompts = rows(plan['references']['full80_prompts']['path']); audit = read(plan['references']['CPU_audit']['path'])
    cpu = read(plan['references']['CPU_result']['path'])
    assert cpu['status'] == 'PASS_CPU_FIXED_FULLTEXT_REVERSE_NO_MODEL_NO_QUALITY'
    assert cpu['reverse_total_tokens'] == 670117 and cpu['maximum_input_tokens'] == 14390
    assert len(prompts) == len(audit) == 80 and len({p['row_id'] for p in prompts}) == 80
    assert len({str(p['pmc_id']) for p in prompts}) == 16
    assert sum(len(p['code_map']) for p in prompts) == 587
    assert sum(len(p['code_map'])*len(p['patient_ids']) for p in prompts) == 1557
    ordered = sorted(range(80),key=lambda i:(audit[i]['reverse']['tokens'],prompts[i]['row_id']))
    assert summary['processing_original_indices'] == ordered
    actual_rows = {}
    for name in ('responses','canonical'):
        compressed = checked(refs[name]); plain = root/'actual'/('%s.jsonl'%name)
        assert gzip.decompress(compressed.read_bytes()) == plain.read_bytes()
        same(refs[name],summary[name]['gzip']); same(ref(plain),summary[name]['plain'])
        assert ref(plain)['sha256'] == summary[name+'_sha256']
        actual_rows[name] = rows(plain)
        assert len(actual_rows[name]) == 80
    records = read(refs['record_manifest']['path']); same(refs['record_manifest'],summary['record_manifest'])
    assert len(records) == 80 and [r['original_index'] for r in records] == ordered
    raw_by_id, canonical_by_id = {}, {}
    for index, record, raw, canonical in zip(ordered,records,actual_rows['responses'],actual_rows['canonical']):
        prompt, row_audit = prompts[index], audit[index]
        assert row_audit['index'] == index and jsha(prompt) == row_audit['reversed_row_sha256']
        assert record['row_id'] == prompt['row_id'] == row_audit['row_id'] == raw['row_id'] == canonical['row_id']
        assert raw['original_index'] == index and raw['pmc_id'] == canonical['pmc_id'] == prompt['pmc_id']
        expected_provenance = 'cached_probe' if index == 19 else 'fresh'
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
        assert ids['input_ids_sha256'] == row_audit['reverse']['input_ids_sha256']
        assert raw['input_tokens'] == row_audit['reverse']['tokens'] <= 14390
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
        if index == 19:
            for name, directory in [('ids','generated-ids'),('raw','raw-responses'),('format','format-records')]:
                actual = root/'actual'/directory/'019.json'; old = checked(plan['cached_probe_references'][name])
                assert actual.read_bytes() == old.read_bytes()
        else: assert ids['input_prefix_exact']
        raw_by_id[prompt['row_id']] = raw; canonical_by_id[prompt['row_id']] = canonical
    coverage = dict(rows=80,documents=16,patients=39,entities=1189,fresh=79,cached=1,
        strict_complete_rows=sum(x['strict_complete'] for x in canonical_by_id.values()),
        whole_row_abstentions=sum(not x['strict_complete'] for x in canonical_by_id.values()),
        natural_EOS_rows=sum(x['natural_eos'] for x in raw_by_id.values()),
        token_cap_rows=sum(x['hit_token_limit'] for x in raw_by_id.values()),
        all_members_verified=len(files),cached_reverse19_three_files_byteexact=True,
        raw_text_reparsed=False,native_decode_repeated=False)
    for key in ('strict_complete_rows','whole_row_abstentions','natural_EOS_rows','token_cap_rows'):
        assert coverage[key] == summary[key]
    return prompts,raw_by_id,canonical_by_id,coverage,files
