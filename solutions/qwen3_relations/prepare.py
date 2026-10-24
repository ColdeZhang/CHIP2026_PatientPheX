"""Persist the same complete native tokenization already checked read-only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
from common import checked, ref, save, sha, source_gate, stamp


def encode(plan):
    start = time.monotonic()
    source_gate(plan)
    output = Path(plan['cpu_dir'])
    output.mkdir(exist_ok=False)
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    raw = Tokenizer.from_file(str(Path(plan['model_dir']) / 'tokenizer.json'))
    a, b = json.loads(raw.to_str()), json.loads(tokenizer.backend_tokenizer.to_str())
    assert all(a.get(k) == b.get(k) for k in ['model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens'])
    bodies = {'train': gzip.decompress(checked(plan['train_gzip']).read_bytes()),
              'held': checked(plan['held_prompts']).read_bytes()}
    encoded, counts = {}, {}
    for split, body in bodies.items():
        assert hashlib.sha256(body).hexdigest() == plan['data_sha256'][split]
        rows = [json.loads(line) for line in body.splitlines()]
        assert len(rows) == (124 if split == 'train' else 80)
        output_rows = []
        for row in rows:
            prefix = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = raw.encode(prefix, add_special_tokens=False).ids
            assert ids == tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            assert raw.decode(ids, skip_special_tokens=False) == tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prefix
            item = {'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']), 'input_ids': ids,
                    'prompt_length': len(ids), 'code_map': row['code_map'], 'patient_ids': row['patient_ids']}
            if split == 'train':
                full = tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': row['answer']}],
                                                    tokenize=False, add_generation_prompt=False, enable_thinking=False)
                full_ids = raw.encode(full, add_special_tokens=False).ids
                assert full_ids == tokenizer.apply_chat_template(row['messages'] + [{'role': 'assistant', 'content': row['answer']}],
                                                    tokenize=True, add_generation_prompt=False, enable_thinking=False)
                assert full_ids[:len(ids)] == ids
                assert raw.decode(full_ids[len(ids):], skip_special_tokens=False) == row['answer'] + tokenizer.eos_token + '\n'
                item['input_ids'] = full_ids
                item['labels'] = [-100] * len(ids) + full_ids[len(ids):]
                assert len(full_ids) <= 8192 and len(full_ids) > len(ids)
            else:
                assert len(ids) <= 14400
            output_rows.append(item)
        path = output / (split + '.jsonl.gz')
        data = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in output_rows).encode()
        with path.open('xb') as stream:
            stream.write(gzip.compress(data, mtime=0))
            stream.flush()
            import os
            os.fsync(stream.fileno())
        encoded[split] = output_rows
        counts[split] = {'rows': len(output_rows), 'documents': len({row['pmc_id'] for row in output_rows}),
                        'tokens': sum(len(row['input_ids']) for row in output_rows),
                        'max_tokens': max(len(row['input_ids']) for row in output_rows), 'file': ref(path)}
    assert counts['train']['tokens'] == 403151 and counts['train']['max_tokens'] == 5534
    assert counts['held']['tokens'] == 666591 and counts['held']['max_tokens'] == 14322
    train_ids = {row['pmc_id'] for row in encoded['train']}
    held_ids = {row['pmc_id'] for row in encoded['held']}
    assert train_ids == set(plan['train32_ids']) and held_ids == set(plan['F1_ids']) and not train_ids & held_ids
    supervised = [sum(value != -100 for value in row['labels'][1:]) for row in encoded['train']]
    assert sum(supervised) == 7170
    shapes = [(len(row['input_ids']), supervised[i]) for i, row in enumerate(encoded['train'])]
    pareto = [i for i, shape in enumerate(shapes) if not any(
        other[0] >= shape[0] and other[1] >= shape[1] and other != shape for other in shapes)]
    # One representative per exact duplicate shape; no task row is removed from formal training.
    representative = {}
    for i in pareto:
        representative.setdefault(shapes[i], i)
    pareto = sorted(representative.values(), key=lambda i: shapes[i])
    order = list(range(124))
    random.Random(plan['seed']).shuffle(order)
    result = {'status': 'PASS_NATIVE_COMPLETE_124_AND_FULL80_PERSISTED', 'at_utc': stamp(), 'source': ref(__file__),
              'counts': counts, 'supervised_tokens': sum(supervised), 'shortest_index': min(range(124), key=lambda i: shapes[i]),
              'pareto_indices': pareto, 'pareto_shapes': [shapes[i] for i in pareto],
              'stress_initial_eight': order[:8], 'stress_accumulated_three': order[8:11],
              'no_new_gold': True, 'no_model_weights': True, 'seconds': time.monotonic() - start}
    save(output / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    encode(json.loads(Path(args.plan).read_text()))
