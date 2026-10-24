"""F0 all58 actual tokenizer census, CPU only; never load model weights/adapter/gold."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
sys.dont_write_bytecode = True
F = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def sha(b): return hashlib.sha256(b).hexdigest()
def jsha(v): return sha(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
def save(name, value):
    with (F / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    started = time.monotonic()
    assert not (F / 'result.json').exists() and not (F / 'row-audit.json').exists()
    plan = json.loads((F / 'plan.json').read_text())
    assert plan['status'] == 'F0_ORIGINAL58_SOURCE_ONLY_NATIVE_TOKENIZER_CPU_NO_MODEL_NO_QUALITY'
    assert plan['cap_rule'] == {'round_up_multiple': 512, 'maximum_observed_limit': 32768, 'generation_limit': 1600, 'above_limit_action': 'REPORT_AND_BLOCK_PROBE_NO_TRUNCATION'}
    for name, record in plan['files'].items():
        data = (F / name).read_bytes()
        assert len(data) == record['bytes'] and sha(data) == record['sha256']
    original_bytes = gzip.decompress((F / 'original58.jsonl.gz').read_bytes())
    input_bytes = gzip.decompress((F / 'input16.jsonl.gz').read_bytes())
    expanded_bytes = gzip.decompress((F / 'expanded58.jsonl.gz').read_bytes())
    for data, name in [(original_bytes, 'original_prompts'), (input_bytes, 'input16'), (expanded_bytes, 'expanded_prompts')]:
        assert len(data) == plan[name]['bytes'] and sha(data) == plan[name]['sha256']
    original = [json.loads(line) for line in original_bytes.decode().splitlines()]
    documents = [json.loads(line) for line in input_bytes.decode().splitlines()]
    assert len(original) == 58 and len(documents) == 16
    assert all('answer' not in row for row in original) and all(not {'entities', 'association'} & doc.keys() for doc in documents)
    assert sum(len(doc['patient']) for doc in documents) == 47
    assert sum(len(row['code_map']) for row in original) == 415
    assert sum(len(row['code_map']) * len(row['patient_ids']) for row in original) == 1166
    # Normal CLI loads siblings from this script directory; path-loader callers must register them (H50).
    from construct import append_reports
    expanded, doc_audit, row_audit = append_reports(original, documents, training=False)
    reconstructed = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in expanded).encode()
    assert reconstructed == expanded_bytes and all(r['original_row_restored_exact'] for r in row_audit)
    base = Path(plan['base']); assets = {}
    for name, record in plan['tokenizer_files'].items():
        data = (base / name).read_bytes()
        assert len(data) == record['bytes'] and sha(data) == record['sha256']
        assets[name] = dict(bytes=len(data), sha256=sha(data))
    config = json.loads((base / 'config.json').read_text())
    context = config.get('text_config', config)['max_position_embeddings']
    import torch
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    assert torch.__version__ == plan['runtime_expected']['torch']
    assert transformers.__version__ == plan['runtime_expected']['transformers']
    assert tokenizers.__version__ == plan['runtime_expected']['tokenizers']
    assert not torch.cuda.is_initialized()
    auto = AutoTokenizer.from_pretrained(base, local_files_only=True)
    native = tokenizers.Tokenizer.from_file(str(base / 'tokenizer.json'))
    assert auto.is_fast
    audits = []
    for index, (old, new, source) in enumerate(zip(original, expanded, row_audit)):
        item = dict(index=index, row_id=old['row_id'], pmc_id=str(old['pmc_id']),
                    original_row_sha256=source['original_row_sha256'], expanded_row_sha256=source['extended_row_sha256'],
                    suffix_sha256=source['suffix_sha256'])
        for name, row in [('original', old), ('expanded', new)]:
            rendered = auto.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = auto.encode(rendered, add_special_tokens=False)
            templated = auto.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                add_generation_prompt=True, enable_thinking=False)
            native_ids = native.encode(rendered, add_special_tokens=False).ids
            assert ids == templated == native_ids, (index, name, 'actual input IDs differ')
            decoded = auto.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            native_decoded = native.decode(ids, skip_special_tokens=False)
            assert decoded == native_decoded == rendered, (index, name, 'actual decode roundtrip differs')
            item[name] = dict(tokens=len(ids), input_ids_sha256=jsha(ids), rendered_sha256=sha(rendered.encode()),
                              raw_Auto_template_IDs_exact=True, raw_Auto_roundtrip_exact=True)
        audits.append(item)
    def lengths(name):
        values = [row[name]['tokens'] for row in audits]; largest = max(values)
        return dict(total=sum(values), minimum=min(values), maximum=largest, median=statistics.median(values),
                    over_old14000=sum(v > 14000 for v in values), over32768=sum(v > 32768 for v in values),
                    longest_indices=[i for i, v in enumerate(values) if v == largest],
                    max_input_plus_original1600_generation=largest + 1600)
    expanded_lengths = lengths('expanded')
    recommended_cap = ((expanded_lengths['maximum'] + 511) // 512) * 512
    within_declared_limit = expanded_lengths['maximum'] <= 32768
    context_fits = recommended_cap + 1600 <= context
    projection = [{'row_id':r['row_id'],'messages':r['messages'],'code_map':r['code_map'],'patient_ids':r['patient_ids']} for r in expanded]
    result = {'status':'PASS_ACTUAL_58_APPEND_ONLY_NATIVE_TOKENIZER_INPUT_CENSUS_NO_MODEL_OR_QUALITY',
        'plan_sha256':sha((F / 'plan.json').read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),
        'runtime':dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
                       actual_tokenizer_class=type(auto).__name__,CUDA_initialized=torch.cuda.is_initialized()),
        'actual_tokenizer_assets':assets,'rows':58,'documents':16,'patients':47,'candidates':415,'pairs':1166,
        'passages':sum(len(doc['full_text']) for doc in documents),'given_mentions':sum(d['given_mentions'] for d in doc_audit),
        'inserted_markers':sum(len(p['inserted_markers']) for d in doc_audit for p in d['passages']),
        'skipped_markers':sum(len(p['overlap_skipped_markers']) for d in doc_audit for p in d['passages']),
        'all_passage_text_reconstructs_exact':all(d['original_text_reconstruction_exact'] for d in doc_audit),
        'all_original_rows_restored_exact':all(r['original_row_restored_exact'] for r in row_audit),
        'original':lengths('original'),'expanded':expanded_lengths,'expanded_messages_CID_PID_projection_sha256':jsha(projection),
        'recommended_input_limit':recommended_cap,'input_limit_rounding':512,'generation_limit':1600,'model_context_tokens':context,
        'observed_maximum_within32768':within_declared_limit,'rounded_cap_plus_generation_fits_context':context_fits,
        'probe_input_gate_passed':within_declared_limit and context_fits,'gpu_execution_performed':False,
        'model_weights_loaded':False,'adapter_loaded':False,'gold_read':False,'new_quality_calls':0,
        'truncation':False,'elapsed_seconds':time.monotonic()-started}
    result['input_token_ratio'] = result['expanded']['total'] / result['original']['total']
    if not within_declared_limit or not context_fits:
        result['status'] = 'INPUT_CENSUS_COMPLETE_BUT_GPU_PROBE_BLOCKED_BY_DECLARED_LENGTH_OR_CONTEXT'
    save('row-audit.json', audits)
    result['row_audit'] = dict(bytes=(F / 'row-audit.json').stat().st_size, sha256=sha((F / 'row-audit.json').read_bytes()))
    save('result.json', result)
    assert not torch.cuda.is_initialized()
    print(json.dumps(result))


if __name__ == '__main__': main()
