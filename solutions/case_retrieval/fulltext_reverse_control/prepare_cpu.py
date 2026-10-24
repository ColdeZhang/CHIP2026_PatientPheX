"""Input-only exact fulltext reversal and native token census; no weights/gold/score."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from reverse_fulltext import fulltext_reverse


def sha(data): return hashlib.sha256(data).hexdigest()
def jsha(value): return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=sha(data))
def checked(record, directory=None):
    path = Path(record['path']) if directory is None else Path(directory) / Path(record['path']).name
    actual = ref(path)
    assert actual['bytes'] == record['bytes'] and actual['sha256'] == record['sha256']
    return path
def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
def save(path, value):
    with path.open('x') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n'); file.flush(); os.fsync(file.fileno())
    fsync_dir(path.parent)
def rows(path): return [json.loads(line) for line in path.read_bytes().splitlines()]
def write_rows(path, values):
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values).encode()
    with path.open('xb') as file:
        file.write(data); file.flush(); os.fsync(file.fileno())
    destination = path.with_suffix(path.suffix + '.gz')
    with destination.open('xb') as file:
        with gzip.GzipFile(filename='', mode='wb', fileobj=file, mtime=0) as compressed:
            compressed.write(data)
        file.flush(); os.fsync(file.fileno())
    fsync_dir(path.parent)
    assert gzip.decompress(destination.read_bytes()) == data
    return {'plain': ref(path), 'gzip': ref(destination)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, help='Optional staged copies by original basename; exact SHA still required')
    args = parser.parse_args(); started = time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    plan = json.loads(args.plan.read_text()); output = args.output_dir.resolve()
    assert plan['cpu_seconds'] == 60
    output.mkdir(exist_ok=False); fsync_dir(output.parent)
    report = dict(status='RUNNING_INPUT_ONLY_CENSUS', source=ref(__file__), plan=ref(args.plan),
                  model_loaded=False, gold_read=False, new_quality_calls=0, GPU_calls=0)
    def timeout(*_): raise TimeoutError('Fixed60second CPU preparation budget')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(60)
    try:
        for name, expected in plan['sources'].items():
            actual = ref(Path(__file__).parent / name)
            assert all(actual[key] == expected[key] for key in ('bytes', 'sha256'))
        sources = plan['input_references']
        original = rows(checked(sources['short_original80'], args.input_dir))
        expanded = rows(checked(sources['fulltext_normal80'], args.input_dir))
        old_audit = json.loads(checked(sources['normal_row_audit'], args.input_dir).read_text())
        assert len(original) == len(expanded) == len(old_audit) == 80
        assert len({row['row_id'] for row in original}) == 80
        assert len({str(row['pmc_id']) for row in original}) == 16
        assert sum(len(row['code_map']) for row in original) == 587
        assert sum(len(row['code_map'])*len(row['patient_ids']) for row in original) == 1557
        reversed_rows, audits = [], []
        for index, (short, full, prior) in enumerate(zip(original, expanded, old_audit)):
            assert short['row_id'] == full['row_id'] == prior['row_id'] and prior['index'] == index
            assert jsha(short) == prior['original_row_sha256'] and jsha(full) == prior['expanded_row_sha256']
            new, reversed_short, mapping, suffix = fulltext_reverse(full, short)
            restored, restored_short, inverse, restored_suffix = fulltext_reverse(new, reversed_short)
            assert restored == full and restored_short == short and restored_suffix.encode() == suffix.encode()
            assert all(inverse[new_code] == old_code for old_code, new_code in mapping.items())
            assert sha(suffix.encode()) == prior['suffix_sha256']
            fake = {cid:list(short['patient_ids']) for cid in short['code_map']}
            assert {inverse[mapping[cid]]: value for cid, value in fake.items()} == fake
            reversed_rows.append(new)
            audits.append(dict(index=index, row_id=full['row_id'], pmc_id=str(full['pmc_id']),
                original_row_sha256=jsha(full), reversed_row_sha256=jsha(new), suffix_sha256=sha(suffix.encode()),
                old_to_new=mapping, new_to_old=inverse, inverse_all_fields_exact=True,
                suffix_UTF8_bytes_exact=True, concept_PID_sets_exact=True))
        actual_assets = {}
        for name, expected in plan['tokenizer_files'].items():
            actual = ref(args.base / name)
            assert all(actual[key] == expected[key] for key in ('bytes', 'sha256'))
            actual_assets[name] = actual
        import torch, transformers, tokenizers
        from transformers import AutoTokenizer
        assert torch.__version__ == '2.7.1+cu126' and transformers.__version__ == '5.20.2'
        assert tokenizers.__version__ == '0.23.1' and not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        auto = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        raw = tokenizers.Tokenizer.from_file(str(args.base / 'tokenizer.json'))
        assert auto.is_fast
        encoded = []
        for index, (normal, reverse, audit, prior) in enumerate(zip(expanded, reversed_rows, audits, old_audit)):
            for name, row in [('normal', normal), ('reverse', reverse)]:
                rendered = auto.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = auto.encode(rendered, add_special_tokens=False)
                templated = auto.apply_chat_template(row['messages'], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                assert ids == templated == raw.encode(rendered, add_special_tokens=False).ids
                assert auto.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == raw.decode(ids, skip_special_tokens=False) == rendered
                audit[name] = dict(tokens=len(ids), input_ids_sha256=jsha(ids), rendered_sha256=sha(rendered.encode()),
                                   raw_Auto_template_IDs_exact=True, raw_Auto_roundtrip_exact=True)
                if name == 'normal':
                    assert audit[name] == prior['expanded']
                else:
                    encoded.append(dict(index=index, row_id=row['row_id'], input_ids=ids))
        lengths = [row['reverse']['tokens'] for row in audits]
        selected = min(range(80), key=lambda i:(-lengths[i], reversed_rows[i]['row_id']))
        input_cap = ((max(lengths) + 511) // 512) * 512
        config = json.loads((args.base / 'config.json').read_text())
        context = config.get('text_config', config)['max_position_embeddings']
        assert input_cap + 1600 <= context and not torch.cuda.is_initialized()
        artifacts = dict(prompts=write_rows(output / 'prompts.jsonl', reversed_rows),
                         encoded=write_rows(output / 'encoded.jsonl', encoded))
        save(output / 'row-audit.json', audits)
        write_rows(output / 'longest-probe.jsonl', [reversed_rows[selected]])
        report.update(status='PASS_CPU_FIXED_FULLTEXT_REVERSE_NO_MODEL_NO_QUALITY', rows=80, documents=16,
            candidates=587, pairs=1557, patients=39, changed_rows=sum(a != b for a,b in zip(expanded,reversed_rows)),
            normal_total_tokens=sum(a['normal']['tokens'] for a in audits), reverse_total_tokens=sum(lengths),
            maximum_input_tokens=max(lengths), selected_longest_index=selected, selected_row_id=reversed_rows[selected]['row_id'],
            input_cap_ceil512=input_cap, original_generation_limit=1600, model_context=context, truncated_rows=0,
            raw_Auto_template_IDs_and_roundtrip_all80=True, all_inverse_and_suffix_bytes_exact=True,
            runtime=dict(torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                         actual_tokenizer_class=type(auto).__name__, CUDA_initialized=False),
            artifacts=artifacts, row_audit=ref(output / 'row-audit.json'), actual_tokenizer_assets=actual_assets,
            CPU_scope='No weights/adapter/gold/model generation; native tokenizer input census only.')
    except BaseException as error:
        report.update(status='FAILED_CPU_PREPARATION_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        save(output / 'result.json', report)
        print(json.dumps({key:report.get(key) for key in ('status','maximum_input_tokens','input_cap_ceil512','elapsed_seconds')}))


if __name__ == '__main__': main()
