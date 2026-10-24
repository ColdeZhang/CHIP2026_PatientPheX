"""Complete original G strings under the existing Qwen3 native runtime; no NN."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from common import ref, write, load_jsonl, validate_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    args = ap.parse_args()
    started = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    validate_refs(plan['files'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['OMP_NUM_THREADS'] == '1'
    import torch
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from token_census import census
    from reuse.native_tokenizer import encode
    torch.set_num_threads(1)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    assert transformers.__version__ == '4.51.3'
    assert tokenizers.__version__ == '0.21.2'
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    model = Path(plan['model_dir'])
    for name, expected in plan['model_assets'].items():
        actual = ref(model / name)
        assert actual['bytes'] == expected['size'] and actual['sha256'] == expected['sha256'], name
    write(out / 'runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        tokenizers=tokenizers.__version__, transformers_path=transformers.__file__,
        cuda_initialized=torch.cuda.is_initialized(), cuda_available=torch.cuda.is_available(),
        no_model_weights=True))
    inputs = out / 'inputs'
    inputs.mkdir()
    for name in ['train', 'held_input']:
        item = plan['compressed_inputs'][name]
        raw = gzip.decompress(Path(item['path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == item['raw_sha256']
        with (inputs / (name + '.jsonl')).open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    native_plan = dict(model='Qwen/Qwen3-8B', template_kind='config', eos_token_id=151645,
        tokenizers_version='0.21.2', inputs=dict(train=ref(inputs / 'train.jsonl'),
        held_input=ref(inputs / 'held_input.jsonl'), tokenizer=ref(model / 'tokenizer.json'),
        template=ref(model / 'tokenizer_config.json')))
    write(out / 'native-plan.json', native_plan)
    native = census(native_plan, out / 'native')
    encoded = json.loads(gzip.decompress((out / 'native/encoded.json.gz').read_bytes()))
    tk = AutoTokenizer.from_pretrained(model, local_files_only=True)
    raw_tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    observations = []
    counts = {}
    for phase, inputname in [('train', 'train'), ('held', 'held_input')]:
        rows = load_jsonl(inputs / (inputname + '.jsonl'))
        assert len(rows) == len(encoded[phase]) == (87 if phase == 'train' else 39)
        expanded = []
        for row, expected in zip(rows, encoded[phase]):
            prompt = tk.apply_chat_template(row['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = tk.encode(prompt, add_special_tokens=False)
            assert ids == raw_tokenizer.encode(prompt, add_special_tokens=False).ids
            assert tk.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prompt
            assert raw_tokenizer.decode(ids, skip_special_tokens=False) == prompt
            if phase == 'train':
                actual = encode(row, tk)
                assert actual == expected
                assert tk.decode(actual['input_ids'], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) == prompt + row['answer'] + '<|im_end|>\n'
            else:
                assert ids == expected['input_ids'] and 'answer' not in row
            expanded.append(dict(expected, pmc_id=row['pmc_id'], target_patient_id=row['target_patient_id']))
            observations.append(dict(phase=phase, row_id=row['row_id'], original_messages_answer_unchanged=True,
                native_HF_IDs_decode_labels_exact=True))
        payload = b''.join((json.dumps(row, separators=(',', ':')) + '\n').encode() for row in expanded)
        path = out / (phase + '.encoded.jsonl.gz')
        with path.open('xb') as handle:
            handle.write(gzip.compress(payload, mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        counts[phase] = dict(rows=len(expanded), documents=len({row['pmc_id'] for row in expanded}),
            tokens=sum(len(row['input_ids']) for row in expanded),
            max_tokens=max(len(row['input_ids']) for row in expanded),
            supervised_tokens=sum(row.get('supervised_tokens', 0) for row in expanded), encoded=ref(path))
    write(out / 'observations.json', observations)
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS_QWEN3_GROUNDED_126_NATIVE_CPU_NO_MODEL', route_id=plan['route_id'],
        plan=ref(args.plan), source=ref(__file__), runtime=ref(out / 'runtime.json'),
        native_manifest=ref(out / 'native/manifest.json'), original_encoded=ref(out / 'native/encoded.json.gz'),
        counts=counts, stress_initial8=native['probe_first_real_group'],
        stress_accumulated3=native['probe_accumulate_next_three'],
        pareto_indices=[row['index'] for row in native['joint_length_supervision_frontier']],
        pareto_shapes=[[row['full_tokens'], row['supervised_tokens']] for row in native['joint_length_supervision_frontier']],
        all126_native_HF_IDs_decode_exact=True, all87_original_answers_EOS_labels_exact=True,
        complete_no_truncation=True, no_model_GPU_fit_forward_or_quality=True,
        seconds=time.monotonic() - started)
    print(json.dumps(write(out / 'result.json', result)), flush=True)


if __name__ == '__main__':
    main()
