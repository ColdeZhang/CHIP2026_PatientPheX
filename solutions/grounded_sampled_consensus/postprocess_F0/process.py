"""Full94 raw -> exact anchors -> fixed train32/public normalization; no scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
from common import (SEEDS, POLICY_SHA, all_patient_keys, checked, guard, jsonl,
                    load_plan, modules, publish, read, ref, save)


def extract(plan):
    inputs = plan['inputs']; collection = read(checked(inputs['collection']))
    assert collection['all_actual_SHA_verified'] and collection['inference_reported_success']
    assert collection['all_original_output_files_included'] and collection['files_and_ancestor_directories_fsynced']
    assert collection['child_exit_before_freeze']['no_running_process']
    files = collection['files']; local = Path(collection['local']).resolve(); remote = Path(collection['original_remote'])
    for name, item in files.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert Path(item['path']).resolve() == local / name
        checked(item)
    wrapper = read(checked(files['wrapper-complete.json']))
    global_complete = read(checked(files['output/complete.json']))
    assert wrapper == collection['wrapper'] and wrapper['returncode'] == 0 and not wrapper['timed_out']
    assert 'endpoint_error' not in wrapper and wrapper['complete94_actual_members_verified']
    assert wrapper['complete_sha256'] == files['output/complete.json']['sha256']
    assert global_complete['status'] == 'COMPLETE_FRESH94_SAMPLED_QUOTE_NO_QUALITY'
    assert (global_complete['rows'], global_complete['documents']) == (94, 16)
    assert global_complete['no_regeneration_or_answer_repair']
    assert global_complete['source']['sha256'] == inputs['runner']['sha256']; checked(inputs['runner'])
    assert global_complete['plan']['sha256'] == inputs['inference_plan']['sha256']; checked(inputs['inference_plan'])

    def mapped(item):
        relative = Path(item['path']).relative_to(remote)
        assert '..' not in relative.parts and relative.parts
        found = files[relative.as_posix()]
        assert (found['bytes'], found['sha256']) == (item['bytes'], item['sha256'])
        return found

    _, _, decoder, schema, _, _ = modules()
    docs = jsonl(checked(inputs['F0_input'])); patients = all_patient_keys(docs)
    by = {str(d['pmc_id']): d for d in docs}
    assert all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs)
    prompts = jsonl(checked(inputs['prompts'])); prompt = {row['row_id']: row for row in prompts}
    assert len(prompt) == len(prompts) == 47
    assert {(str(r['pmc_id']), r['target_patient_id']) for r in prompts} == patients
    all_quotes = []; all_decisions = []; summaries = {}; all_gens = []; all_raws = []
    assert set(global_complete['seed_completes']) == set(map(str, SEEDS))
    for seed in SEEDS:
        c = read(checked(mapped(global_complete['seed_completes'][str(seed)])))
        assert mapped(global_complete['seed_completes'][str(seed)]) == files[f'output/{seed}/complete.json']
        assert (c['status'], c['rows'], c['documents']) == ('COMPLETE_FRESH47_RAW_GENERATION_NO_QUALITY', 47, 16)
        assert c['no_regeneration_or_answer_repair'] and c['source'] == global_complete['source']
        assert c['plan'] == global_complete['plan']
        assert mapped(c['responses']) == files[f'output/{seed}/responses.jsonl']
        records = jsonl(checked(mapped(c['responses'])))
        assert len(records) == 47 and {r['row_id'] for r in records} == set(prompt)
        generations = {}; raws = {}
        for field, target in [('generation_refs', generations), ('raw_refs', raws)]:
            assert len(c[field]) == 47
            for item in c[field]:
                value = read(checked(mapped(item))); rid = value['row_id']
                assert rid not in target and value['seed'] == seed
                target[rid] = (item, value)
        assert set(generations) == set(raws) == set(prompt)
        quotes = []; decisions = []
        for record in records:
            rid = record['row_id']; row = prompt[rid]; did = str(row['pmc_id']); pid = row['target_patient_id']
            assert record['seed'] == seed and (did, pid) in patients
            assert str(record['pmc_id']) == did and record['target_patient_id'] == pid
            assert 'answer' not in row and 'labels' not in row and row['units'] == schema.units(by[did])
            gref, generation = generations[rid]; rref, raw = raws[rid]
            assert raw == record and record['generation'] == gref
            assert record['raw_decoded'] and record['raw_before_parse'] and record['HF_decode_equal']
            assert record['parsed'] is False and generation['decoded_or_parsed'] is False
            for key, value in generation.items():
                if key != 'decoded_or_parsed': assert record[key] == value
            assert len(record['generated_token_ids']) == record['output_tokens']
            decision = decoder.interpret(record['response'], by[did], record['actual_last_token_eos'], record['hit_token_limit'])
            decisions.append(dict(seed=seed, row_id=rid, pmc_id=did, target_patient_id=pid, raw=rref, decision=decision))
            for index, anchor in enumerate(decision['anchors']):
                quotes.append(dict(seed=seed, row_id=rid, pmc_id=did, target_patient_id=pid, anchor_index=index, **anchor))
        assert c['EOS_rows'] == sum(r['actual_last_token_eos'] for r in records)
        assert c['cap_rows'] == sum(r['hit_token_limit'] for r in records)
        summaries[str(seed)] = dict(rows=47, quotes=len(quotes), EOS=c['EOS_rows'], cap=c['cap_rows'],
            abstaining_rows=sum(d['decision']['status'] == 'abstain' for d in decisions),
            rejected_items=sum(len(d['decision']['rejected']) for d in decisions),
            rows_without_valid_anchor=sum(not d['decision']['anchors'] for d in decisions),
            row_reasons=dict(Counter(d['decision']['reason'] for d in decisions)),
            rejected_item_reasons=dict(Counter(r['reason'] for d in decisions for r in d['decision']['rejected'])))
        all_quotes.extend(quotes); all_decisions.extend(decisions)
        all_gens.extend(c['generation_refs']); all_raws.extend(c['raw_refs'])
    assert all_gens == global_complete['generation_refs'] and all_raws == global_complete['raw_refs']
    out = Path(plan['output_directory']); out.mkdir(parents=True, exist_ok=False)
    publish(out/'quotes.jsonl', b''.join((json.dumps(q, ensure_ascii=False)+'\n').encode() for q in all_quotes))
    save(out/'decisions.json', all_decisions)
    save(out/'manifest.json', dict(schema='grounded-sampled-complete94-extraction-v1', plan=plan['_reference'],
        sources=plan['sources'], collection=inputs['collection'], F0_input=inputs['F0_input'], prompts=inputs['prompts'],
        complete_all94=True, seeds=list(SEEDS), documents=16, patients=47, rows=94,
        quotes=ref(out/'quotes.jsonl'), decisions=ref(out/'decisions.json'), exact_anchors=len(all_quotes),
        unique_quotes=len({q['quote'] for q in all_quotes}), per_seed=summaries,
        original_raw_not_repaired=True, HPO_mapping_or_gold_or_score_performed=False))
    return ref(out/'manifest.json')


def prepare(plan):
    refs = plan['inputs']; ex = read(checked(refs['extraction']))
    assert ex['complete_all94'] and ex['seeds'] == list(SEEDS) and ex['rows'] == 94
    quotes = jsonl(checked(ex['quotes'])); decisions = read(checked(ex['decisions']))
    assert len(quotes) == ex['exact_anchors'] and len(decisions) == 94
    docs = jsonl(checked(refs['F0_input'])); patients = all_patient_keys(docs)
    assert refs['F0_input'] == ex['F0_input']
    by = {str(d['pmc_id']): d for d in docs}
    for seed in SEEDS:
        assert {(d['pmc_id'], d['target_patient_id']) for d in decisions if d['seed'] == seed} == patients
    _, _, _, _, strict, semantic = modules()
    assert refs['train32']['sha256'] == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    normalizer = strict.StrictNormalizer(checked(refs['hpo']), checked(refs['train32']))
    normalizer.assert_disjoint_input(checked(refs['F0_input']))
    policy = semantic.load_policy(checked(refs['policy'])); assert refs['policy']['sha256'] == POLICY_SHA
    rows = []; pending = set()
    for q in quotes:
        assert q['seed'] in SEEDS and (q['pmc_id'], q['target_patient_id']) in patients
        passage = strict.verify_span(by[q['pmc_id']], q['quote'], q['offset'])
        decision = normalizer.normalize(q['quote'])
        rows.append(dict(anchor=q, passage_index=passage, strict=decision))
        if decision['status'] == 'abstain' and decision['reason'] == policy['semantic_fallback_reason'] and not semantic.precheck(q['quote'], policy):
            pending.add(q['quote'])
    queries = [dict(quote=q) for q in sorted(pending)]
    out = Path(plan['output_directory']); out.mkdir(parents=True, exist_ok=False)
    save(out/'strict-decisions.json', rows); save(out/'queries.json', queries)
    save(out/'manifest.json', dict(schema='grounded-sampled-normalization-preparation-v1', plan=plan['_reference'],
        sources=plan['sources'], extraction=refs['extraction'], complete_all94=True, rows=94,
        seeds=list(SEEDS), documents=16, patients=47, anchors=len(rows),
        necessary_public_query_count=len(queries), strict_decisions=ref(out/'strict-decisions.json'),
        queries=ref(out/'queries.json'), input=refs['F0_input'], train32=refs['train32'], hpo=refs['hpo'],
        policy=refs['policy'], shared_queries_only_exact_identical_strings=True,
        no_gold_fit_or_score=True, no_unknown_to_NIL=True, no_threshold_changes=True))
    return ref(out/'manifest.json')


def finalize(plan):
    refs = plan['inputs']; prep = read(checked(refs['preparation']))
    assert prep['schema'] == 'grounded-sampled-normalization-preparation-v1' and prep['complete_all94']
    assert prep['seeds'] == list(SEEDS) and prep['rows'] == 94
    rows = read(checked(prep['strict_decisions'])); queries = read(checked(prep['queries']))
    assert len(rows) == prep['anchors'] and len(queries) == prep['necessary_public_query_count']
    _, _, _, _, _, semantic = modules()
    policy = semantic.load_policy(checked(prep['policy'])); assert prep['policy']['sha256'] == POLICY_SHA
    scores = {}
    if queries:
        collection = read(checked(refs['query_collection']))
        assert collection['all_actual_SHA_verified'] and collection['query_reported_success']
        assert collection['wrapper']['returncode'] == 0 and not collection['wrapper']['timed_out']
        assert 'endpoint_error' not in collection['wrapper']
        raw = read(checked(refs['query_scores'])); complete = read(checked(refs['query_complete']))
        assert complete['status'] == 'COMPLETE_FIXED_PUBLIC_HELD_QUERIES'
        assert complete['query_count'] == len(queries) and complete['public_bank_rows'] == 41099
        assert complete['no_bank_reencoding'] and complete['no_fit'] and complete['no_quality']
        assert complete['queries']['sha256'] == prep['queries']['sha256']
        assert complete['scores']['sha256'] == refs['query_scores']['sha256']
        assert complete['policy']['sha256'] == POLICY_SHA
        assert refs['query_complete'] in collection['files'].values() and refs['query_scores'] in collection['files'].values()
        scores = {r['quote']: r for r in raw}
        assert len(scores) == len(raw) == len(queries) and set(scores) == {q['quote'] for q in queries}
    else:
        assert all(refs.get(k) is None for k in ['query_collection', 'query_scores', 'query_complete'])
    normalized = []
    for row in rows:
        q = row['anchor']; decision = semantic.decide(row['strict'], scores.get(q['quote']), policy)
        normalized.append({**q, **decision, 'policy_sha256': POLICY_SHA,
                           'length': len(q['quote']), 'passage_index': row['passage_index']})
    out = Path(plan['output_directory']); out.mkdir(parents=True, exist_ok=False)
    publish(out/'normalizations.jsonl', b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in normalized))
    save(out/'semantic-evidence.json', scores)
    save(out/'manifest.json', dict(schema='grounded-sampled-complete94-normalization-v1', plan=plan['_reference'],
        sources=plan['sources'], preparation=refs['preparation'], extraction=prep['extraction'],
        complete_all94=True, seeds=list(SEEDS), rows=94, documents=16, patients=47,
        records=len(normalized), normalizations=ref(out/'normalizations.jsonl'),
        semantic_evidence=ref(out/'semantic-evidence.json'),
        query_collection=refs.get('query_collection'), query_complete=refs.get('query_complete'), query_scores=refs.get('query_scores'),
        necessary_public_queries=len(queries), policy=prep['policy'], input=prep['input'], train32=prep['train32'],
        per_seed={str(s): dict(records=sum(r['seed']==s for r in normalized),
            status_counts=dict(Counter(r['status'] for r in normalized if r['seed']==s)),
            reasons=dict(Counter(r['reason'] for r in normalized if r['seed']==s)),
            accepted_sources=dict(Counter(r['match'] for r in normalized if r['seed']==s and r['status'] in ('mapped','nil')))) for s in SEEDS},
        exact_anchor_fields_preserved=True, no_policy_or_threshold_changes=True, no_gold_fit_or_quality=True))
    return ref(out/'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['extract','prepare','finalize'])
    parser.add_argument('--plan', required=True); args = parser.parse_args(); state = guard()
    plan = load_plan(args.plan, args.stage); plan['_reference'] = ref(args.plan)
    result = {'extract':extract, 'prepare':prepare, 'finalize':finalize}[args.stage](plan)
    assert not state['F0_quality_enabled']; print(json.dumps(result))
