"""Original G normalization verification body; T5 raw validation lives separately."""
from support import read,checked,jsonl


def verify_normalization(plan, extraction):
    a=plan['actual'];references=plan['references'];quotes=jsonl(checked(a['quotes']))
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
    return normalized
