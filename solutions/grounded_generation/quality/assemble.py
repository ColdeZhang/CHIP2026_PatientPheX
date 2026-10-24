"""Full F1 N-union-new and new-only assembly, preserving all nonphenotype bytes.

This entry does not tokenize, decode, normalize, fit, infer, read gold or evaluate.
It consumes the complete39 extraction and fixed normalization artifacts verbatim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback
from support import (ARMS, byte_helpers, canonical_values, checked, jsonl, load_module,
                     load_plan, now, publish, read, ref, relation_keys, replace_phenotypes,
                     rows, save)


def verify_actual(plan, inputs):
    a = plan['actual']; references = plan['references']
    collection = read(a['collection']['path']); files = collection['files']
    assert collection['schema'] == 'grounded-held39-complete-collection-v1'
    assert collection['inference_reported_success'] and collection['all_actual_SHA_verified']
    assert collection['all_original_output_files_included'] and collection['files_and_ancestor_directories_fsynced']
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
    assert wrapper['complete_sha256'] == files['output/complete.json']['sha256']
    assert complete['status'] == 'COMPLETE_FRESH39_RAW_GENERATION_NO_QUALITY'
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


def new_relations(normalized, inputs, strict, schema, policy_sha):
    by = {str(d['pmc_id']): d for d in inputs}; hpo = strict.HPO
    selected = {(did, p['patient_id']): set() for did, d in by.items() for p in d['patient']}
    records = []; counts = Counter()
    for row in normalized:
        did = str(row['pmc_id']); pid = row['target_patient_id']; doc = by[did]
        assert (did, pid) in selected and row['policy_sha256'] == policy_sha
        passage = strict.verify_span(doc, row['quote'], row['offset'])
        assert passage == row['passage_index'] and row['length'] == len(row['quote'])
        anchor = schema.grounded([row['unit_id'], row['quote'], row['occurrence_index']],
                                 {unit['unit_id']: unit for unit in schema.units(doc)})
        assert all(row[k] == value for k, value in anchor.items())
        status = row['status']; assert status in ('mapped', 'nil', 'abstain'); counts[status] += 1
        if status == 'abstain':
            assert row['ids'] == [] and row['identifier'] is None; values = []
        elif status == 'nil':
            assert row['ids'] == ['-1'] and row['identifier'] == '-1'
            assert row['match'] == 'literal' and row['reason'] == 'unanimous_whole_span_alias'
            assert hpo.relation_keys(row['quote']) == (('nil', row['quote']),)
            values = [row['quote']]
        else:
            ids = hpo.identifiers(row['identifier'])
            assert ids != ('-1',) and list(ids) == row['ids']
            assert all(hpo.canonical[hid] == hid for hid in ids)
            assert row['identifier'] == ';'.join(ids)
            assert row['reason'] in ('unanimous_whole_span_alias', 'public_sap_cosine_and_concept_margin')
            values = list(ids)
        for value in values: selected[(did, pid)].update(hpo.relation_keys(value))
        records.append(dict(row_id=row['row_id'], pmc_id=did, patient_id=pid, anchor_index=row['anchor_index'],
                            status=status, reason=row['reason'], values=values))
    return selected, records, dict(counts)


def old_candidate_pool(plan, inputs, hpo):
    references = plan['references']; original = read(references['old_plan']['path'])
    prompts = jsonl(references['old_prompts']['path']); actual = read(references['old_actual_input']['path'])
    expected = references['old_prompts']
    assert {k: expected[k] for k in ('bytes', 'sha256')} == original['files']['expanded80.jsonl']
    assert original['baseline_old_SFT32_reference']['input'] == references['F1_input']
    assert actual['rows'] == 80 and actual['documents'] == 16 and actual['candidate_codes'] == 587
    assert actual['all80_actual_ids_equal_prior_native_CPU'] and actual['no_truncation'] and actual['gold_read'] is False
    patients = {str(d['pmc_id']): {p['patient_id'] for p in d['patient']} for d in inputs}
    pair_pool = set(); document_pool = set(); row_ids = set(); codes = pairs = 0
    for row in prompts:
        assert row['row_id'] not in row_ids; row_ids.add(row['row_id']); did = str(row['pmc_id'])
        assert len(row['patient_ids']) == len(set(row['patient_ids']))
        assert set(row['patient_ids']) == patients[did]
        codes += len(row['code_map']); pairs += len(row['code_map']) * len(row['patient_ids'])
        for value in row['code_map'].values():
            for key in hpo.relation_keys(value):
                document_pool.add((did, *key))
                pair_pool.update((did, pid, *key) for pid in row['patient_ids'])
    assert len(prompts) == 80 and codes == 587 and pairs == 1557
    assert {str(r['pmc_id']) for r in prompts} == set(patients)
    return pair_pool, document_pool, dict(rows=80, sum_code_map=587, sum_code_map_times_patients=1557,
        unique_document_concepts=len(document_pool), unique_patient_concept_pairs=len(pair_pool))


def run(plan_path):
    plan, guard = load_plan(plan_path); assert plan['sources']['assemble.py'] == ref(__file__)
    output = Path(plan['output_directory']) / 'assembled'
    output.mkdir(parents=True, exist_ok=False)
    try:
        references = plan['references']; lines, baseline, ids = rows(references['baseline']['path'])
        _, inputs, input_ids = rows(references['F1_input']['path'])
        assert ids == input_ids and len(ids) == 16
        assert all(set(doc) == {'pmc_id', 'pmid', 'patient', 'full_text'} for doc in inputs)
        assert sum(len(doc['entities']) for doc in baseline) == 1189
        assert sum(len(doc['association']) for doc in baseline) == sum(len(doc['patient']) for doc in inputs) == 39
        for doc, source in zip(baseline, inputs):
            assert [a['patient_id'] for a in doc['association']] == [p['patient_id'] for p in source['patient']]
        strict = load_module('_grounded_quality_strict', references['strict'])
        hpo = strict.HPO(references['hpo']['path'])
        schema = load_module('_grounded_quality_schema', references['schema'])
        normalized, coverage = verify_actual(plan, inputs)
        # Pass the already loaded ontology; never instantiate StrictNormalizer or any encoder.
        class BoundStrict:
            HPO = hpo
            verify_span = staticmethod(strict.verify_span)
        selected, decisions, counts = new_relations(normalized, inputs, BoundStrict, schema,
                                                     references['normalization_policy']['sha256'])
        save(output / 'normalization-consumption.json', dict(counts=counts, coverage=coverage, records=decisions))
        candidate_pairs, candidate_docs, pool_counts = old_candidate_pool(plan, inputs, hpo)
        base_keys = relation_keys(baseline, hpo)
        assert all(canonical_values(a['phenotype'], hpo) == a['phenotype'] for d in baseline for a in d['association']), 'Actual strong N must already be canonical and unique; never rewrite its original values'
        assert base_keys <= candidate_pairs, 'Strong N must originate from its exact old prompt pool'
        new_keys = {(did, pid, *key) for (did, pid), values in selected.items() for key in values}
        novel = new_keys - base_keys
        outside_pair = new_keys - candidate_pairs
        outside_document = {key for key in new_keys if (key[0], *key[2:]) not in candidate_docs}
        contribution = dict(schema='grounded-new-relation-contribution-v1', no_gold_used=True,
            old_prompt=references['old_prompts'], old_actual_input=references['old_actual_input'], pool_counts=pool_counts,
            baseline_is_within_old_candidate_pool=True, baseline_relations=len(base_keys), new_relations=len(new_keys),
            new_and_N=len(new_keys & base_keys), new_not_N=len(novel),
            new_outside_old_patient_candidate_pool=len(outside_pair), new_outside_old_document_candidate_pool=len(outside_document),
            keys=dict(new=sorted(new_keys), new_not_N=sorted(novel), outside_patient_pool=sorted(outside_pair),
                      outside_document_pool=sorted(outside_document)),
            interpretation='Outside-pool presence measures candidate-ceiling escape, not correctness; TP/FP only after frozen two-endpoint scoring')
        save(output / 'contributions.json', contribution)
        spans, mask = byte_helpers(references['byte_helper'])
        validator = load_module('validate_predictions', references['validator']); official_hpo = validator.read_hpo(references['hpo']['path'])
        predictions = {}; validations = {}
        for arm in ARMS:
            rendered = []
            for line, doc in zip(lines, baseline):
                values = {}
                for association in doc['association']:
                    pid = association['patient_id']; did = str(doc['pmc_id'])
                    fresh = [key[1] for key in sorted(selected[(did, pid)])]
                    values[pid] = canonical_values((association['phenotype'] if arm == 'primary' else []) + fresh, hpo)
                    if arm == 'primary': assert values[pid][:len(association['phenotype'])] == association['phenotype']
                rendered.append(replace_phenotypes(line, values, spans, mask))
            path = output / (arm + '.jsonl'); publish(path, ''.join(rendered).encode())
            _, docs, actual_ids = rows(path); assert actual_ids == ids
            expected = base_keys | new_keys if arm == 'primary' else new_keys
            assert relation_keys(docs, hpo) == expected
            assert all(mask(a) == mask(b) for a, b in zip(lines, rendered))
            report = validator.validate_predictions(inputs, docs, official_hpo)
            validation_path = output / (arm + '.validation.json'); save(validation_path, report)
            assert report['valid'], report
            predictions[arm] = ref(path); validations[arm] = ref(validation_path)
        save(output / 'contract.json', dict(documents=16, patients=39, entities=1189,
            all39_raw_covered=True, inference_coverage=coverage, all_nonphenotype_bytes_exact=True,
            primary_is_N_union_new=True, primary_original_relation_prefix_exact=True, diagnostic_is_new_only=True, canonical_typed_key_dedup=True,
            patient_objects_and_Task1_unchanged=True, old_pool_counts=pool_counts,
            baseline_reference=references['baseline'], normalization_counts=counts,
            gold_read_or_hashed=False, evaluator_calls=0, normalization_or_NN_calls=0))
        assert guard['enabled'] is False
        save(output / 'common-freeze.json', dict(at_utc=now(), plan=ref(plan_path), sources=plan['sources'],
            actual=plan['actual'], predictions=predictions, validations=validations,
            contract=ref(output / 'contract.json'), contributions=ref(output / 'contributions.json'),
            normalization_consumption=ref(output / 'normalization-consumption.json'), both_complete_endpoints=True,
            gold_read_or_hashed=False, new_evaluator_calls_before_freeze=0))
        print(json.dumps(ref(output / 'common-freeze.json')))
    except BaseException:
        save(output / 'failure.json', dict(at_utc=now(), traceback=traceback.format_exc(),
            gold_read_or_hashed=False, automatic_retry=False, partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan.resolve())
