"""Original complete39 verifier; only collection provenance scope changes."""
from pathlib import Path
from support import checked,jsonl,read,ref

def verify_actual(plan, inputs):
    a = plan['actual']; references = plan['references']
    collection = read(a['collection']['path']); files = collection['files']
    assert collection['schema'] == 'copy-grounded-complete39-collection-v1'
    assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
    assert collection['all39_raw_and_ID_bytes_verified'] and collection['files_and_ancestor_directories_fsynced']
    assert collection['collection_mode']=='closed_child_byteexact_archive' and collection['remote_archive_frozen']
    checked(collection['archive'])
    assert collection['child_exit_before_freeze']['no_running_process']
    local = Path(collection['local']).resolve(); remote = Path(collection['original_remote'])
    for name, item in files.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert Path(item['path']).resolve() == local / relative
        checked(item)
    wrapper = read(files['wrapper-complete.json']['path'])
    complete = read(files['output/complete.json']['path'])
    assert wrapper == collection['wrapper'] and wrapper['returncode'] == 0 and not wrapper['timed_out']
    assert 'endpoint_error' not in wrapper and wrapper['complete39_actual_members_verified']
    assert wrapper['result_sha256'] == files['output/complete.json']['sha256']
    assert complete['status'] == 'COMPLETE_FRESH39_CONSTRAINED_RAW_NO_QUALITY'
    assert complete['strict_rows']==complete['EOS_rows']==39 and complete['cap_rows']==0
    assert complete['rows'] == 39 and complete['documents'] == 16
    assert complete['no_regeneration_or_answer_repair']

    def mapped(item):
        relative = Path(item['path']).relative_to(remote)
        assert '..' not in relative.parts and relative.parts
        found = files[relative.as_posix()]
        assert found['bytes'] == item['bytes'] and found['sha256'] == item['sha256']
        return found

    assert mapped(complete['responses']) == files['output/responses.jsonl']
    responses = jsonl(files['output/responses.jsonl']['path'])
    records = {row['row_id']: row for row in responses}
    assert len(records) == len(responses) == 39
    generations = {}; raw = {}
    for name, target in [('generation_refs', generations), ('raw_refs', raw)]:
        assert len(complete[name]) == 39
        for item in complete[name]:
            value = read(mapped(item)['path']); rid = value['row_id']
            assert rid not in target; target[rid] = (item, value)
    assert set(records) == set(generations) == set(raw)
    constraint_audit = read(mapped(complete['constraint_audit'])['path'])
    assert len(constraint_audit)==39 and {x['row_id'] for x in constraint_audit}==set(records)
    assert all(x['pass_strict'] and x['complete_EOS_schema_pool_unique'] for x in constraint_audit)
    for rid, record in records.items():
        generation_ref, generation = generations[rid]
        assert raw[rid][1] == record and record['generation'] == generation_ref
        assert record['raw_decoded'] and record['raw_before_parse'] and record['HF_decode_equal']
        assert record['parsed'] is False and generation['decoded_or_parsed'] is False
        for key, value in generation.items():
            if key != 'decoded_or_parsed': assert record[key] == value
        assert len(record['generated_token_ids']) == record['output_tokens']
    expected_patients = {(str(d['pmc_id']), p['patient_id']) for d in inputs for p in d['patient']}
    assert {(str(r['pmc_id']), r['target_patient_id']) for r in responses} == expected_patients

    extraction = read(a['extraction_manifest']['path']); ep = read(a['extraction_plan']['path'])
    assert extraction['plan'] == a['extraction_plan'] and extraction['sources'] == ep['sources']
    assert ep['schema'] == 'grounded-extract-complete39-v1' and ep['gold_metadata_only'] == plan['gold']
    assert ep['sources']['extract_quotes.py'] == references['extract_quotes']
    assert ep['sources']['decode.py'] == references['decode'] and ep['sources']['schema.py'] == references['schema']
    assert complete['source']['sha256'] == ep['sources']['run.py']['sha256']
    for item in ep['sources'].values(): checked(item)
    assert ep['inputs']['complete'] == files['output/complete.json']
    assert ep['inputs']['wrapper'] == files['wrapper-complete.json']
    assert ep['inputs']['responses'] == files['output/responses.jsonl']
    assert ep['inputs']['F1_input'] == references['F1_input']
    assert Path(ep['local_output']).resolve() == local / 'output'
    assert Path(ep['remote_output']) == remote / 'output'
    prompts = jsonl(checked(ep['inputs']['prompts']))
    assert len(prompts) == 39 and len({r['row_id'] for r in prompts}) == 39
    assert {r['row_id'] for r in prompts} == set(records)
    for row in prompts:
        assert 'answer' not in row and 'labels' not in row
        record = records[row['row_id']]
        assert str(row['pmc_id']) == str(record['pmc_id']) and row['target_patient_id'] == record['target_patient_id']
    assert extraction['complete_original39'] and extraction['documents'] == 16 and extraction['patients'] == 39
    assert extraction['HPO_mapping_or_gold_or_score_performed'] is False
    assert extraction['original_raw_not_repaired'] and extraction['quotes'] == a['quotes'] and extraction['decisions'] == a['decisions']
    decisions = read(a['decisions']['path']); quotes = jsonl(a['quotes']['path'])
    assert len(decisions) == 39 and len({d['row_id'] for d in decisions}) == 39
    assert {d['row_id'] for d in decisions} == set(records)
    rebuilt = []
    for item in decisions:
        rid = item['row_id']; record = records[rid]; decision = item['decision']
        assert item['pmc_id'] == str(record['pmc_id']) and item['target_patient_id'] == record['target_patient_id']
        assert item['raw'] == raw[rid][0] and decision['raw_not_repaired']
        assert decision['status'] in ('complete', 'abstain')
        if decision['status'] == 'abstain': assert not decision['anchors']
        else: assert record['actual_last_token_eos'] and not record['hit_token_limit']
        for index, anchor in enumerate(decision['anchors']):
            rebuilt.append(dict(row_id=rid, pmc_id=item['pmc_id'], target_patient_id=item['target_patient_id'],
                                anchor_index=index, **anchor))
    assert quotes == rebuilt and len(quotes) == extraction['exact_anchors']
    assert extraction['abstaining_rows'] == sum(d['decision']['status'] == 'abstain' for d in decisions)
    assert extraction['rejected_items'] == sum(len(d['decision']['rejected']) for d in decisions)
    assert extraction['unique_quotes'] == len({q['quote'] for q in quotes})

    manifest = read(a['normalization_manifest']['path'])
    normalized = jsonl(a['normalizations']['path'])
    assert manifest['schema'] == 'grounded-held-normalization-complete-v1'
    assert manifest['source'] == references['held_finalize']
    assert manifest['extraction_manifest'] == a['extraction_manifest']
    assert manifest['normalizations'] == a['normalizations'] and manifest['semantic_evidence'] == a['semantic_evidence']
    assert manifest['documents'] == 16 and manifest['patients'] == 39
    assert manifest['no_quality'] and manifest['no_policy_or_threshold_changes']
    finalize_plan = read(checked(manifest['plan']))
    assert finalize_plan['schema'] == 'grounded-held-normalization-finalize-v1'
    assert manifest['sources'] == finalize_plan['sources']
    for item in manifest['sources'].values(): checked(item)
    assert manifest['sources']['held_finalize.py'] == references['held_finalize']
    for name in ['strict', 'semantic', 'public_bank']:
        assert manifest['sources'][name + '.py'] == references[name]
    preparation = read(checked(manifest['preparation']))
    assert preparation['schema'] == 'grounded-held-normalization-preparation-v1'
    assert preparation['source'] == references['held_prepare'] and preparation['extraction_manifest'] == a['extraction_manifest']
    assert preparation['input'] == references['F1_input'] and preparation['policy'] == references['normalization_policy']
    assert preparation['no_gold_fit_or_score'] and preparation['no_threshold_changes'] and preparation['no_unknown_to_NIL']
    assert preparation['train32']['sha256'] == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert preparation['documents'] == 16 and preparation['patients'] == 39 and preparation['anchors'] == len(quotes)
    # Only verify existing held-query output identities; never rebuild aliases or recompute a score.
    checked(preparation['strict_decisions']); queries = read(checked(preparation['queries']))
    assert len(queries) == preparation['necessary_public_query_count'] == manifest['necessary_public_queries']
    if queries:
        query_collection = read(checked(manifest['query_collection']))
        query_complete = read(checked(manifest['query_complete'])); checked(manifest['query_scores'])
        assert query_collection['all_actual_SHA_verified'] and query_collection['wrapper']['returncode'] == 0
        assert not query_collection['wrapper']['timed_out'] and 'endpoint_error' not in query_collection['wrapper']
        assert query_complete['status'] == 'COMPLETE_FIXED_PUBLIC_HELD_QUERIES'
        assert query_complete['query_count'] == len(queries) and query_complete['public_bank_rows'] == 41099
        assert query_complete['no_bank_reencoding'] and query_complete['no_fit']
        assert query_complete['scores']['sha256'] == manifest['query_scores']['sha256']
        assert query_complete['queries']['sha256'] == preparation['queries']['sha256']
        assert query_complete['policy']['sha256'] == references['normalization_policy']['sha256']
    else:
        assert manifest['query_collection'] is None and manifest['query_complete'] is None and manifest['query_scores'] is None
    assert manifest['input_sha256'] == references['F1_input']['sha256']
    assert manifest['quotes_sha256'] == a['quotes']['sha256']
    assert manifest['normalizations_sha256'] == a['normalizations']['sha256']
    assert manifest['semantic_evidence_sha256'] == a['semantic_evidence']['sha256']
    assert manifest['policy_sha256'] == references['normalization_policy']['sha256']
    assert manifest['no_fit'] is True and manifest['no_input_annotations'] is True
    assert len(normalized) == len(quotes) == manifest['records']
    assert manifest['status_counts'] == {s: sum(r['status'] == s for r in normalized) for s in ['mapped', 'nil', 'abstain']}
    for quote, row in zip(quotes, normalized):
        assert all(row[k] == value for k, value in quote.items()), 'No normalized row may relabel a patient or source anchor'
        assert row['policy_sha256'] == references['normalization_policy']['sha256']
        assert row['normalization_policy'] == 'grounded-public-sap-conservative-v1'
    return normalized, dict(rows=39, raw_members=39, generation_members=39,
        complete_decisions=39 - extraction['abstaining_rows'], abstaining_rows=extraction['abstaining_rows'],
        rejected_items=extraction['rejected_items'], anchors=len(quotes), unique_quotes=extraction['unique_quotes'],
        raw_IDs_decode_or_parser_reexecuted=False, normalization_or_encoder_reexecuted=False)
