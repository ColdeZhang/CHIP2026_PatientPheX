#!/usr/bin/env python3
"""Pure F0 construction and provenance checks; no tokenizer, model, gold or quality."""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True


def sha(b): return hashlib.sha256(b).hexdigest()
def jsha(v): return sha(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
def ref(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=sha(b))
def check(r):
    p = Path(r['path']); b = p.read_bytes()
    assert len(b) == r['bytes'] and sha(b) == r['sha256'], str(p)
    return b
def js(b): return json.loads(b)
def lines(b): return [json.loads(line) for line in b.decode().splitlines() if line.strip()]
def write(p, b):
    with Path(p).open('xb') as out:
        out.write(b); out.flush(); os.fsync(out.fileno())
def save(p, v): write(p, (json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    item = importlib.util.module_from_spec(spec); sys.modules[name] = item; spec.loader.exec_module(item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', required=True); parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(); started = time.monotonic()
    root = Path(args.output_dir).resolve(); plan = js(Path(args.source_plan).read_bytes())
    assert plan['status'] == 'FROZEN_F0_ORIGINAL58_SFT48_APPEND_ONLY_SOURCE_PREPARATION'
    references = plan['references']; blobs = {k: check(v) for k, v in references.items()}
    original, documents = lines(blobs['original_prompts']), lines(blobs['input16'])
    assert len(original) == 58 and len(documents) == 16
    assert all('answer' not in row and set(row) == {'row_id', 'pmc_id', 'code_map', 'patient_ids', 'messages'} for row in original)
    assert all(not {'entities', 'association'} & doc.keys() for doc in documents)
    assert sum(len(doc['patient']) for doc in documents) == 47
    assert sum(len(row['code_map']) for row in original) == 415
    assert sum(len(row['code_map']) * len(row['patient_ids']) for row in original) == 1166
    assert len({row['row_id'] for row in original}) == 58
    document_ids = {str(doc['pmc_id']) for doc in documents}
    checkpoint = js(blobs['training_checkpoint']); training = js(blobs['training_manifest']); fp = checkpoint['fingerprint']
    assert fp == training['fingerprint']
    assert len(set(fp['training_document_ids'])) == 48 and not document_ids & set(fp['training_document_ids'])
    assert document_ids == set(fp['excluded_document_ids']) and fp['excluded_input_sha256'] == references['input16']['sha256']
    assert training['examples'] == 204 and training['epochs'] == 2 and training['validation_labels_used'] is False
    assert checkpoint['completed_epochs'] == 2 and checkpoint['global_step'] == 52 and checkpoint['smoke_only'] is False
    adapter = plan['original_SFT48_adapter']['files']
    hashes = js(blobs['adapter_hash_receipt'])['epoch-002']['files']
    manifest = js(blobs['old_response_manifest']); old_plan = js(blobs['old_inference_plan'])
    for name, expected in adapter.items():
        assert checkpoint['files']['adapter/' + name] == hashes['adapter/' + name] == expected['sha256']
        assert manifest['fingerprint']['adapter_files'][name] == expected['sha256']
    assert old_plan['source_prompts_sha256'] == manifest['fingerprint']['prompts_sha256'] == references['original_prompts']['sha256']
    assert manifest['expected_rows'] == 58 and manifest['output_sha256'] == references['old_responses']['sha256']
    baseline = lines(blobs['old_baseline_pred']); metrics = js(blobs['old_baseline_metrics'])
    assert metrics['prediction_sha256'] == references['old_baseline_pred']['sha256'] and metrics['metrics']['Score'] == .6954439475814587
    assert len(baseline) == 16 and sum(len(doc['entities']) for doc in baseline) == 947
    assert [str(d['pmc_id']) for d in baseline] == [str(d['pmc_id']) for d in documents]
    for prediction, doc in zip(baseline, documents):
        assert {a['patient_id'] for a in prediction['association']} == {p['patient_id'] for p in doc['patient']}
    # H50: register the frozen sibling module before loading the constructor by path.
    legacy = module('legacy_full_context', references['legacy_full_context']['path'])
    construct = module('f0_frozen_full_report_construct', references['construct']['path'])
    expanded, doc_audit, row_audit = construct.append_reports(original, documents, training=False)
    report_by_doc = {d['pmc_id']: d for d in doc_audit}
    per_row = []
    for index, (old, new, audit) in enumerate(zip(original, expanded, row_audit)):
        assert list(old) == list(new) and {k:v for k,v in old.items() if k != 'messages'} == {k:v for k,v in new.items() if k != 'messages'}
        assert old['messages'][:-1] == new['messages'][:-1]
        suffix = new['messages'][-1]['content'][len(old['messages'][-1]['content']):]
        restored = copy.deepcopy(new); restored['messages'][-1]['content'] = restored['messages'][-1]['content'][:-len(suffix)]
        assert restored == old and suffix.startswith(legacy.APPEND_PREFIX) and sha(suffix.encode()) == audit['suffix_sha256']
        assert old['messages'][-1]['content'].encode() == new['messages'][-1]['content'][:-len(suffix)].encode()
        assert jsha(old['code_map']) == jsha(new['code_map']) and jsha(old['patient_ids']) == jsha(new['patient_ids'])
        assert list(old['code_map']) == ['C%03d' % j for j in range(len(old['code_map']))]
        per_row.append(dict(index=index, **audit, code_map_sha256=jsha(old['code_map']), patient_ids_sha256=jsha(old['patient_ids']),
                            original_user_utf8_sha256=sha(old['messages'][-1]['content'].encode()), suffix_bytes=len(suffix.encode()),
                            appended_passages=len(report_by_doc[str(old['pmc_id'])]['passages'])))
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in expanded).encode()
    write(root / 'expanded58.jsonl', payload)
    save(root / 'construction-document-audit.json', doc_audit); save(root / 'construction-row-audit.json', per_row)
    cpu = root / 'input-cpu'
    for filename, data in [('original58.jsonl.gz', blobs['original_prompts']), ('input16.jsonl.gz', blobs['input16']), ('expanded58.jsonl.gz', payload)]:
        write(cpu / filename, gzip.compress(data, mtime=0))
    summary = {'status':'PASS_LOCAL_F0_58_APPEND_ONLY_CONSTRUCTION_AND_OLD48_PROVENANCE_NO_TOKENIZER',
        'at_utc':datetime.now(timezone.utc).isoformat(), 'source':ref(__file__), 'source_plan':ref(args.source_plan),
        'rows':58, 'documents':16, 'patients':47, 'candidate_codes':415, 'pairs':1166, 'preserved_baseline_entities':947,
        'training_documents':48, 'training_rows_metadata':204, 'training_holdout_document_overlap':[],
        'training_and_excluded_document_ids_checked_from_original_checkpoint_metadata':True,
        'passages':sum(len(d['full_text']) for d in documents),
        'given_mentions':sum(d['given_mentions'] for d in doc_audit),
        'inserted_markers':sum(len(p['inserted_markers']) for d in doc_audit for p in d['passages']),
        'overlap_skipped_markers':sum(len(p['overlap_skipped_markers']) for d in doc_audit for p in d['passages']),
        'all_given_patients_marked':all(d['all_patients_marked'] for d in doc_audit),
        'all_passage_text_reconstructs_exact':all(d['original_text_reconstruction_exact'] for d in doc_audit),
        'all_original_rows_restored_exact_after_removing_suffix':all(r['original_row_restored_exact'] for r in per_row),
        'all_original_metadata_schema_CID_PID_and_local_user_UTF8_unchanged':True,
        'old_prompts_line_serialization_equals_canonical_output_style':blobs['original_prompts'] == ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in original).encode(),
        'source_construct_identical_to_F1':references['construct']['sha256'], 'legacy_marker_source_identical_to_F1':references['legacy_full_context']['sha256'],
        'baseline_existing_metrics_reused':metrics['metrics'], 'original_SFT48_adapter_metadata':plan['original_SFT48_adapter'],
        'artifacts':{n:ref(root/n) for n in ['expanded58.jsonl','construction-document-audit.json','construction-row-audit.json']},
        'native_tokenizer_execution_pending':True, 'input_token_lengths_not_yet_measured':True,
        'CPU_rule':'Actual maximum rounded upward to 512; if maximum >32768 report and do not start probe; no truncation.',
        'gold_read':False,'training_contents_read':False,'model_weights_loaded':False,'adapter_weights_read':False,
        'tokenizer_loaded':False,'new_quality_calls':0,'remote_operations':0,'elapsed_seconds':time.monotonic()-started}
    save(root / 'construction-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__': main()
