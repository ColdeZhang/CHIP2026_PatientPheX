"""Actual frozen tokenizer/input census only; never load weights, labels or adapters."""
from pathlib import Path
import gzip, hashlib, json, os, statistics, time

F = Path(__file__).parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

def sha(b): return hashlib.sha256(b).hexdigest()
def jsha(v): return sha(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
def save(name, obj):
    with (F/name).open('x') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2); file.write('\n'); file.flush(); os.fsync(file.fileno())

def main():
    t0 = time.monotonic()
    assert not (F/'result.json').exists()
    plan = json.loads((F/'plan.json').read_text())
    for name, r in plan['files'].items():
        b = (F/name).read_bytes(); assert len(b) == r['bytes'] and sha(b) == r['sha256']
    b = gzip.decompress((F/'original80.jsonl.gz').read_bytes())
    assert len(b) == plan['original_prompts']['bytes'] and sha(b) == plan['original_prompts']['sha256']
    original = [json.loads(x) for x in b.decode().splitlines()]
    b = gzip.decompress((F/'input16.jsonl.gz').read_bytes())
    assert len(b) == plan['input16']['bytes'] and sha(b) == plan['input16']['sha256']
    documents = [json.loads(x) for x in b.decode().splitlines()]
    assert len(original) == 80 and len(documents) == 16
    assert all('answer' not in x for x in original)
    assert all(not {'entities', 'association'} & x.keys() for x in documents)
    from construct import append_reports
    expanded, doc_audit, row_audit = append_reports(original, documents, training=False)
    assert all(x['original_row_restored_exact'] for x in row_audit)
    base = Path(plan['base']); actual_assets = {}
    for name, r in plan['tokenizer_files'].items():
        b = (base/name).read_bytes(); assert len(b) == r['bytes'] and sha(b) == r['sha256']
        actual_assets[name] = dict(bytes=len(b), sha256=sha(b))
    import torch, transformers, tokenizers
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    assert transformers.__version__ == '5.20.2' and not torch.cuda.is_initialized()
    auto = AutoTokenizer.from_pretrained(base, local_files_only=True)
    raw = tokenizers.Tokenizer.from_file(str(base/'tokenizer.json'))
    assert auto.is_fast
    audits = []
    for index, (old, new, ra) in enumerate(zip(original, expanded, row_audit)):
        item = {'index': index, 'row_id': old['row_id'], 'pmc_id': str(old['pmc_id']),
                'original_row_sha256': ra['original_row_sha256'],
                'expanded_row_sha256': ra['extended_row_sha256'],
                'suffix_sha256': ra['suffix_sha256']}
        for name, row in [('original', old), ('expanded', new)]:
            rendered = auto.apply_chat_template(row['messages'], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            ids = auto.encode(rendered, add_special_tokens=False)
            templated = auto.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                add_generation_prompt=True, enable_thinking=False)
            native = raw.encode(rendered, add_special_tokens=False).ids
            assert ids == templated == native, (index, name, 'input ID mismatch')
            decoded = auto.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            raw_decoded = raw.decode(ids, skip_special_tokens=False)
            assert decoded == raw_decoded == rendered, (index, name, 'roundtrip mismatch')
            item[name] = dict(tokens=len(ids), input_ids_sha256=jsha(ids), rendered_sha256=sha(rendered.encode()),
                              raw_Auto_template_IDs_exact=True, raw_Auto_roundtrip_exact=True)
        audits.append(item)
    def lengths(name):
        x = [r[name]['tokens'] for r in audits]; maximum = max(x)
        return dict(total=sum(x), minimum=min(x), maximum=maximum, median=statistics.median(x),
                    over14000=sum(v > 14000 for v in x), longest_indices=[i for i,v in enumerate(x) if v == maximum],
                    max_input_plus_original1600_generation=maximum+1600)
    rows = [{'row_id':x['row_id'], 'messages':x['messages'], 'code_map':x['code_map'], 'patient_ids':x['patient_ids']} for x in expanded]
    result = dict(status='PASS_ACTUAL_80_APPEND_ONLY_NATIVE_TOKENIZER_INPUT_CENSUS_NO_MODEL_OR_QUALITY',
                  plan_sha256=sha((F/'plan.json').read_bytes()), source_sha256=sha(Path(__file__).read_bytes()),
                  runtime=dict(torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                               actual_tokenizer_class=type(auto).__name__, CUDA_initialized=torch.cuda.is_initialized()),
                  actual_tokenizer_assets=actual_assets, rows=80, documents=16,
                  patients=sum(len(x['patient']) for x in documents),
                  candidates=sum(len(x['code_map']) for x in original),
                  pairs=sum(len(x['code_map'])*len(x['patient_ids']) for x in original),
                  passages=sum(len(x['full_text']) for x in documents),
                  given_mentions=sum(x['given_mentions'] for x in doc_audit),
                  inserted_markers=sum(len(p['inserted_markers']) for x in doc_audit for p in x['passages']),
                  skipped_markers=sum(len(p['overlap_skipped_markers']) for x in doc_audit for p in x['passages']),
                  all_passage_text_reconstructs_exact=all(x['original_text_reconstruction_exact'] for x in doc_audit),
                  all_original_rows_restored_exact=all(x['original_row_restored_exact'] for x in row_audit),
                  original=lengths('original'), expanded=lengths('expanded'),
                  expanded_messages_CID_PID_projection_sha256=jsha(rows),
                  model_weights_loaded=False, adapter_loaded=False, gold_read=False, new_quality_calls=0,
                  elapsed_seconds=time.monotonic()-t0)
    result['input_token_ratio'] = result['expanded']['total']/result['original']['total']
    save('row-audit.json',audits)
    result['row_audit'] = dict(bytes=(F/'row-audit.json').stat().st_size, sha256=sha((F/'row-audit.json').read_bytes()))
    save('result.json', result); print(json.dumps(result))

if __name__ == '__main__': main()
