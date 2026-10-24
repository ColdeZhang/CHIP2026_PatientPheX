"""Build only the fixed allowed 124 training rows; no model/gold/remote access."""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
sys.dont_write_bytecode = True

def sha(data): return hashlib.sha256(data).hexdigest()
def ref(path):
    p = Path(path); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=sha(b))
def save(path, data):
    with Path(path).open('xb') as f: f.write(data)
def jsave(path, value): save(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
def rows(data): return [json.loads(x) for x in data.decode().splitlines() if x.strip()]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs-root', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    r = Path(args.runs_root); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'preparation.json').exists()
    q = r/'case_retrieval/qwen35-preparation/sft-meta1-fusion'
    manifest = json.loads((q/'preparation-manifest.json').read_text())
    assert ref(q/'preparation-manifest.json')['sha256'] == '356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
    raw = (q/'train.jsonl').read_bytes()
    assert sha(raw) == manifest['training_sha256'] == 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    original = rows(raw)
    assert len(original) == 124 and all([m['role'] for m in row['messages']] == ['system','user'] for row in original)
    assert all(isinstance(row['answer'], str) for row in original)
    documents = []
    for pair in manifest['provenance']:
        assert (pair['generator_fold'], pair['target_fold']) in [(2,3),(3,2)]
        record = pair['target_input']; data = Path(record['path']).read_bytes()
        assert sha(data) == record['sha256'] and len(data) == record['bytes']
        part = rows(data); assert sorted(str(d['pmc_id']) for d in part) == pair['target_ids']
        assert not (set(pair['generator_ids']) & set(pair['target_ids']))
        assert not (set(pair['target_ids']) & set(manifest['excluded_document_ids']))
        documents.extend(part)
    assert len(documents) == 32 and len({str(d['pmc_id']) for d in documents}) == 32
    assert sorted(str(d['pmc_id']) for d in documents) == manifest['training_document_ids']
    assert all(not {'entities','association'} & d.keys() for d in documents)
    src = r/'research/qwen35-sft32-full-report-feasibility/input-cpu'
    sources = {}
    for name, digest in [('legacy_full_context.py','ae7f5853fc55ab430b080631496b73686cd3c6dd1a80ce181553d695ea3e01bb'),
                         ('construct.py','b201fb7f4b1208a6bdad6a4178aea04045cdb9d3f59e64fb0d666fdc891b5c44')]:
        data = (src/name).read_bytes(); assert sha(data) == digest
        save(out/name, data); sources[name] = ref(src/name)
    helper = q/'qwen35_smoke.py'; assert ref(helper)['sha256'] == '9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
    save(out/'qwen35_smoke.py', helper.read_bytes()); sources['qwen35_smoke.py'] = ref(helper)
    load('legacy_full_context',out/'legacy_full_context.py')
    construct = load('construct',out/'construct.py')
    expanded, document_audit, row_audit = construct.append_reports(original, documents, training=True)
    assert all(old['answer'].encode() == new['answer'].encode() for old,new in zip(original,expanded))
    expanded_raw = ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in expanded).encode()
    input_raw = ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in documents).encode()
    for name,data in [('original124.jsonl',raw),('expanded124.jsonl',expanded_raw),('input32.jsonl',input_raw)]:
        save(out/name, data); save(out/(name+'.gz'),gzip.compress(data,mtime=0))
    jsave(out/'document-audit.json',document_audit); jsave(out/'row-construction-audit.json',row_audit)
    def charstats(data):
        values=[len(row['messages'][-1]['content']) for row in data]
        return dict(total_user_characters=sum(values), max_user_characters=max(values),
                    longest_character_indices=[i for i,v in enumerate(values) if v==max(values)])
    result = dict(status='LOCAL_CONSTRUCTION_COMPLETE_NATIVE_TOKEN_COST_PENDING_NO_TRAINING',
        source=ref(__file__), original_preparation=ref(q/'preparation-manifest.json'),
        training_rows=124, training_documents=32, training_folds=[2,3], patients=sum(len(d['patient']) for d in documents),
        passages=sum(len(d['full_text']) for d in documents), given_mentions=sum(a['given_mentions'] for a in document_audit),
        inserted_markers=sum(len(p['inserted_markers']) for a in document_audit for p in a['passages']),
        skipped_markers=sum(len(p['overlap_skipped_markers']) for a in document_audit for p in a['passages']),
        candidates=sum(len(x['code_map']) for x in original), pairs=sum(len(x['code_map'])*len(x['patient_ids']) for x in original),
        original_shape='messages=[system,user]; answer is separate; original encoder appends assistant',
        answers_utf8_preserved_all124=True, all_metadata_CID_PID_local_evidence_preserved=True,
        suffix_removal_restores_all124=True, all_passage_text_reconstruction_exact=True,
        original_characters=charstats(original),expanded_characters=charstats(expanded),
        original_train=ref(out/'original124.jsonl'),expanded_train=ref(out/'expanded124.jsonl'),
        input32=ref(out/'input32.jsonl'),source_files=sources,
        training_document_ids=manifest['training_document_ids'],excluded_document_ids_metadata_only=manifest['excluded_document_ids'],
        validation_gold_opened=False, model_loaded=False, remote_or_GPU_or_training_performed=False,
        native_token_census_performed=False)
    jsave(out/'preparation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','training_document_ids','excluded_document_ids_metadata_only']}))

if __name__ == '__main__': main()
