"""Actual native/Auto token census only; this source does not start itself."""
from pathlib import Path
import argparse, gzip, hashlib, importlib.metadata, json, os, platform, sys, time
os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
import cyclic1_codec as cyclic


def sha(data): return hashlib.sha256(data).hexdigest()
def jsha(value): return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
def save(path, value):
    with Path(path).open('x') as h:
        json.dump(value, h, ensure_ascii=False, indent=2); h.write('\n'); h.flush(); os.fsync(h.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY); os.fsync(fd); os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True, type=Path)
    ap.add_argument('--base', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args(); B = a.plan.parent; out = a.output
    started = time.monotonic(); p = json.loads(a.plan.read_text())
    assert p['CPU_only'] and not p['GPU_authorized'] and not p['training_authorized']
    assert p['cap_seconds'] == 120 and p['grace_seconds'] == 5
    assert p['view'] == cyclic.VIEW and p['new_NN_rows'] == 53 and p['cache_rows'] == 27
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.path.realpath(sys.executable) == os.path.realpath(p['python'])
    assert a.base.resolve() == Path(p['base']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, item in p['files'].items():
        b = (B/name).read_bytes(); assert len(b) == item['bytes'] and sha(b) == item['sha256']
    for name, item in p['tokenizer_files'].items():
        b = (a.base/name).read_bytes(); assert len(b) == item['bytes'] and sha(b) == item['sha256']
    save(out/'native-intent.json', {'started_unix': time.time(), 'pid': os.getpid(),
        'source_sha256': sha(Path(__file__).read_bytes()), 'plan_sha256': sha(a.plan.read_bytes()),
        'model_loaded': False, 'unique_native_census': True})
    import torch, transformers, tokenizers
    runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
        'tokenizers': tokenizers.__version__, 'peft': importlib.metadata.version('peft')}
    assert runtime == p['expected_runtime']
    torch.set_num_threads(1); assert not torch.cuda.is_initialized()
    package = Path(transformers.__file__).parent
    runtime_files = {}
    for name, relative in [('qwen_modeling', 'models/qwen3_5/modeling_qwen3_5.py'),
                           ('generation_utils', 'generation/utils.py')]:
        runtime_files[name] = sha((package/relative).read_bytes())
    assert runtime_files == p['runtime_sha256']
    # Only tokenizer assets are loaded. No AutoModel/PEFT/model class or adapter.
    auto = transformers.AutoTokenizer.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
    native = tokenizers.Tokenizer.from_file(str(a.base/'tokenizer.json'))
    native.no_truncation(); native.no_padding()
    assert type(auto).__name__ == p['expected_tokenizer_class']
    old_raw = gzip.decompress((B/'original-coded80.jsonl.gz').read_bytes())
    new_raw = gzip.decompress((B/'cyclic1-coded80.jsonl.gz').read_bytes())
    assert sha(old_raw) == p['original_coded_sha256'] and sha(new_raw) == p['cyclic1_coded_sha256']
    old_lines, new_lines = old_raw.splitlines(keepends=True), new_raw.splitlines(keepends=True)
    old, new = [json.loads(x) for x in old_lines], [json.loads(x) for x in new_lines]
    contract = json.loads((B/'row-contract.json').read_text())
    prior = json.loads((B/'original-held-native-audit.json').read_text())
    assert len(old) == len(new) == len(contract) == len(prior) == 80
    for i, (before, after, r) in enumerate(zip(old, new, contract)):
        assert r['index'] == i and before['row_id'] == after['row_id'] == r['row_id']
        assert after == cyclic.transform(before)
        assert sha(old_lines[i]) == r['original_line_sha256'] and sha(new_lines[i]) == r['cyclic1_line_sha256']
        if r['cache_original_raw']: assert old_lines[i] == new_lines[i]
    all_encoded, audit, groups = {}, [], {}
    for arm, rows in [('original', old), ('cyclic1', new)]:
        encoded, lengths = [], []
        for i, row in enumerate(rows):
            assert 'answer' not in row
            rendered = auto.apply_chat_template(row['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = auto.encode(rendered, add_special_tokens=False)
            assert ids == native.encode(rendered, add_special_tokens=False).ids
            assert ids == auto.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False)
            assert auto.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == rendered
            assert native.decode(ids, skip_special_tokens=False) == rendered
            prior_row = prior[i]
            assert prior_row['index'] == i and prior_row['row_id'] == row['row_id']
            if arm == 'original':
                assert prior_row['prefix_tokens'] == len(ids) and prior_row['prefix_ids_sha256'] == jsha(ids)
            elif contract[i]['cache_original_raw']:
                assert all_encoded['original'][i]['input_ids'] == ids
            else:
                assert all_encoded['original'][i]['input_ids'] != ids
            encoded.append({'row_id': row['row_id'], 'input_ids': ids}); lengths.append(len(ids))
            audit.append({'arm': arm, 'index': i, 'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']),
                'prefix_tokens': len(ids), 'prefix_ids_sha256': jsha(ids), 'rendered_utf8_sha256': sha(rendered.encode()),
                'native_Auto_template_decode_exact': True, 'cache_original_raw': contract[i]['cache_original_raw']})
        all_encoded[arm] = encoded
        groups[arm] = {'rows': len(rows), 'total_tokens': sum(lengths), 'max_tokens': max(lengths),
            'longest_indices': [i for i,n in enumerate(lengths) if n == max(lengths)],
            'encoded_sha256': jsha(encoded), 'over_prior_cap14848': sum(n > 14848 for n in lengths),
            'proposed_512_rejection_cap': ((max(lengths)+511)//512)*512,
            'NN_tokens': sum(n for n,r in zip(lengths,contract) if r['needs_new_model_response']),
            'cached_tokens': sum(n for n,r in zip(lengths,contract) if r['cache_original_raw'])}
    assert groups['original']['total_tokens'] == 685372 and groups['original']['max_tokens'] == 14570
    assert groups['original']['encoded_sha256'] == '71cfac564f1c18b68e91050953f48ca9e84b95dd740093ddd0b0381bd1e09dd9'
    context = json.loads((a.base/'config.json').read_text())['text_config']['max_position_embeddings']
    assert max(v['max_tokens'] for v in groups.values()) + 1600 <= context
    save(out/'native-row-audit.json', audit)
    result = {'status': 'PASS_ACTUAL_NATIVE_CPU_CYCLIC1_NO_MODEL', 'view': cyclic.VIEW, 'groups': groups,
        'rows': 80, 'new_NN_rows': 53, 'cached_original_rows': 27,
        'singleton_prompt_IDs_equal_original27': True, 'all53_changed_prompts_have_changed_IDs': True,
        'context_tokens': context, 'later_generation_limit': 1600, 'truncation': False,
        'runtime': runtime, 'runtime_source_SHA': runtime_files, 'python_version': platform.python_version(),
        'python_executable': sys.executable, 'actual_tokenizer_class': type(auto).__name__,
        'tokenizer_assets': p['tokenizer_files'], 'model_or_adapter_loaded': False,
        'CUDA_initialized': torch.cuda.is_initialized(), 'gold_read': False, 'quality_calls': 0,
        'source_sha256': sha(Path(__file__).read_bytes()), 'plan_sha256': sha(a.plan.read_bytes()),
        'audit_sha256': sha((out/'native-row-audit.json').read_bytes()), 'seconds': time.monotonic()-started,
        'scope': 'Actual inputs/decoder only; no claim that regenerating cached rows on another host/runtime is bitwise equivalent.'}
    assert not result['CUDA_initialized']; save(out/'native-result.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
