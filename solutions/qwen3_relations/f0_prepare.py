"""Pure CPU native Qwen3 encoding of the existing complete F0 fulltext rows."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time


def ref(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    assert ref(__file__)['sha256'] == plan['source_sha256']
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    assert transformers.__version__ == '4.51.3' and tokenizers.__version__ == '0.21.2'
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    for name, expected in plan['tokenizer_files'].items():
        current = ref(Path(plan['model_dir']) / name)
        assert current['sha256'] == expected['sha256'] and current['bytes'] == expected['size']
    data = Path(plan['prompts_gzip']['path'])
    assert ref(data) == plan['prompts_gzip']
    body = gzip.decompress(data.read_bytes())
    assert hashlib.sha256(body).hexdigest() == plan['prompts_sha256']
    rows = [json.loads(line) for line in body.splitlines()]
    assert len(rows) == 58 and len({row['row_id'] for row in rows}) == 58
    assert {str(row['pmc_id']) for row in rows} == set(plan['document_ids'])
    tok = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    raw = Tokenizer.from_file(str(Path(plan['model_dir']) / 'tokenizer.json'))
    left, right = json.loads(raw.to_str()), json.loads(tok.backend_tokenizer.to_str())
    assert all(left.get(k) == right.get(k) for k in ['model', 'normalizer', 'pre_tokenizer',
                                                   'post_processor', 'decoder', 'added_tokens'])
    encoded, audits, concepts = [], [], set()
    for row in rows:
        assert 'answer' not in row and 'labels' not in row
        assert [x['role'] for x in row['messages']] == ['system', 'user']
        assert row['patient_ids'] == plan['patient_ids'][str(row['pmc_id'])]
        assert list(row['code_map']) == [f'C{i:03d}' for i in range(len(row['code_map']))]
        text = tok.apply_chat_template(row['messages'], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        ids = tok.apply_chat_template(row['messages'], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False)
        assert ids == raw.encode(text, add_special_tokens=False).ids
        assert tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
        assert raw.decode(ids, skip_special_tokens=False) == text
        # Reject only a genuine model-context overflow; no token/row truncation.
        assert len(ids) + 1600 <= 32768
        for concept in row['code_map'].values():
            key = (str(row['pmc_id']), concept)
            assert key not in concepts
            concepts.add(key)
        item = {'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']), 'input_ids': ids,
                'prompt_length': len(ids), 'code_map': row['code_map'],
                'patient_ids': row['patient_ids']}
        encoded.append(item)
        audits.append({'row_id': row['row_id'], 'tokens': len(ids),
                       'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    assert len(concepts) == 415 and sum(len(x) for x in plan['patient_ids'].values()) == 47
    dest = out / 'held58.jsonl.gz'
    packed = ''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':')) + '\n' for x in encoded).encode()
    with dest.open('xb') as stream:
        stream.write(gzip.compress(packed, mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    save(out / 'row-audit.json', audits)
    maximum = max(x['prompt_length'] for x in encoded)
    result = {'status': 'PASS_F0_FULL58_NATIVE_QWEN3_CPU_NO_TRUNCATION',
              'runtime': {'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                          'tokenizer': type(tok).__name__},
              'rows': 58, 'documents': 16, 'patients': 47, 'concepts': len(concepts),
              'pairs': sum(len(x['code_map']) * len(x['patient_ids']) for x in rows),
              'total_tokens': sum(x['prompt_length'] for x in encoded),
              'maximum_tokens': maximum,
              'longest_indices': [i for i, x in enumerate(encoded) if x['prompt_length'] == maximum],
              'minimum_reject_cap_rounded512': ((maximum + 511) // 512) * 512,
              'generation_cap_unchanged': 1600, 'model_context_guard': 32768,
              'all_native_raw_ids_and_roundtrip_exact': True, 'all_full_rows_preserved': True,
              'file': ref(dest), 'audit': ref(out / 'row-audit.json'),
              'seconds': time.monotonic() - started, 'GPU': False, 'model_loaded': False,
              'new_gold_read': False, 'quality_calls': 0, 'source': ref(__file__)}
    save(out / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
