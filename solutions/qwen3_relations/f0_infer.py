"""Same frozen Qwen3 epoch2, complete 58-row F0 inference, no training path."""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
from common import checked, memory, ref, save, sha, stamp, update, verify_assets
from native import parse_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan['run_root']) / 'native'
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    save(root / 'started.json', {'at_utc': stamp(), 'pid': os.getpid(), 'plan': ref(args.plan)})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original900 deadline')))
    torch = None
    stage = 'source_asset_gate'
    try:
        for name, expected in plan['source_files'].items():
            assert sha(Path(__file__).parent / name) == expected, name
        assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
        g = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
        a = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                     '--format=csv,noheader,nounits'], text=True)
        row = [line.split(', ') for line in g.splitlines() if plan['gpu_uuid'] in line]
        gate = {'at_utc': stamp(), 'all_gpu': g, 'all_compute_apps': a,
                'target': row, 'target_apps': [x for x in a.splitlines() if plan['gpu_uuid'] in x],
                'free_disk': shutil.disk_usage(root).free}
        save(root / 'fresh-gate.json', gate)
        assert len(row) == 1 and row[0][0] == '0' and row[0][2] == 'NVIDIA A30'
        assert int(row[0][3]) >= 23500 and not gate['target_apps'] and gate['free_disk'] >= 1024**3
        compatibility = json.loads(checked(plan['cpu_compatibility']).read_text())
        assert compatibility['status'] == 'PASS_TARGET_PRIVATE_RUNTIME_ALL58_IDENTICAL_NATIVE_IDS'
        assert compatibility['versions'] == plan['runtime']
        assert compatibility['all58_native_raw_ids_roundtrip_same_as_source']
        assets = verify_assets(plan, full=True)
        save(root / 'verified-base-assets.json', assets)
        for name, item in plan['adapter_files'].items():
            p = Path(plan['adapter_dir']) / name
            assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
        encoded = checked(plan['held_encoded'])
        rows = [json.loads(x) for x in gzip.decompress(encoded.read_bytes()).splitlines()]
        assert len(rows) == len({x['row_id'] for x in rows}) == 58
        assert len({x['pmc_id'] for x in rows}) == 16
        assert sum(len(x['input_ids']) for x in rows) == 503003
        assert max(len(x['input_ids']) for x in rows) == 16142
        assert all(len(x['input_ids']) <= 16384 and 'labels' not in x for x in rows)
        stage = 'load_frozen_model'
        import torch
        import numpy as np
        import transformers
        import peft
        import tokenizers
        import accelerate
        import safetensors
        import huggingface_hub
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        versions = {n: m.__version__ for n, m in [('torch', torch), ('transformers', transformers),
                    ('peft', peft), ('tokenizers', tokenizers), ('accelerate', accelerate),
                    ('safetensors', safetensors), ('huggingface_hub', huggingface_hub)]}
        assert versions == plan['runtime'], versions
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(20261023)
        np.random.seed(20261023)
        torch.manual_seed(20261023)
        torch.cuda.manual_seed_all(20261023)
        assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        base, loading = AutoModelForCausalLM.from_pretrained(plan['model_dir'], torch_dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='sdpa', local_files_only=True,
            trust_remote_code=False, output_loading_info=True)
        save(root / 'loading.json', loading)
        assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        assert type(base).__name__ == 'Qwen3ForCausalLM' and sum(p.numel() for p in base.parameters()) == 8190735360
        assert hashlib.sha256(inspect.getsource(type(base).forward).encode()).hexdigest() == plan['native_forward_sha256']
        model = PeftModel.from_pretrained(base, plan['adapter_dir'], is_trainable=False, local_files_only=True)
        assert sum(p.numel() for p in model.parameters()) == 8234382336 < 10000000000
        assert not any(p.requires_grad for p in model.parameters())
        lora = {n: p for n, p in model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
        assert len(lora) == 504 and sum(p.numel() for p in lora.values()) == 43646976
        assert all(p.dtype == torch.float32 for p in lora.values())
        saved = load_file(str(Path(plan['adapter_dir']) / 'adapter_model.safetensors'), device='cpu')
        loaded = get_peft_model_state_dict(model)
        assert set(saved) == set(loaded) and len(saved) == 504
        assert all(torch.equal(value, loaded[name].detach().cpu()) for name, value in saved.items())
        del saved, loaded
        model.eval()
        model.config.use_cache = True
        tokenizer = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
        save(root / 'model-manifest.json', {'runtime': versions, 'base_parameters': 8190735360,
             'adapter_parameters': 43646976, 'total_parameters': 8234382336, 'trainable_parameters': 0,
             'adapter_files': plan['adapter_files'], 'model_dir': plan['model_dir'],
             'loaded_adapter_tensors_exact': 504, 'TF32': False, 'training_steps': 0,
             'cross_host_difference': plan['cross_host_difference']})
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
        eos_ids.discard(None)
        for name in ['generated-ids', 'raw-responses', 'canonical-records']:
            (root / name).mkdir()
        raw, canonical = {}, {}
        stage = 'all58_fresh_inference'
        for _, _, index in sorted((len(x['input_ids']), x['row_id'], i) for i, x in enumerate(rows)):
            row = rows[index]
            update(root / 'progress.json', {'at_utc': stamp(), 'completed': len(raw), 'original_index': index,
                   'row_id': row['row_id'], 'tokens': len(row['input_ids'])})
            ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
            tick = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    use_cache=True, max_new_tokens=1600, eos_token_id=sorted(eos_ids), pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            tokens = generated[0, ids.shape[1]:].tolist()
            natural = bool(tokens and tokens[-1] in eos_ids)
            cap = len(tokens) >= 1600 and not natural
            identity = {'row_id': row['row_id'], 'pmc_id': row['pmc_id'], 'original_index': index,
                        'natural_eos': natural, 'hit_token_limit': cap, 'generation_seconds': time.monotonic() - tick}
            save(root / 'generated-ids' / f'{index:03d}.json', {**identity, 'generated_ids': tokens, 'configured_eos_ids': sorted(eos_ids)})
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            item = {**identity, 'response': response, 'input_tokens': ids.shape[1], 'output_tokens': len(tokens), 'provenance': 'fresh'}
            save(root / 'raw-responses' / f'{index:03d}.json', item)
            parsed, error = None, None
            try:
                if not natural or cap:
                    raise ValueError('No natural EOS or generation reached cap')
                parsed = parse_complete(response, row)
            except (ValueError, TypeError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            record = {**identity, 'candidate_patients': parsed if parsed is not None else {},
                      'strict_complete': parsed is not None, 'abstention': error, 'provenance': 'fresh'}
            save(root / 'canonical-records' / f'{index:03d}.json', record)
            raw[index], canonical[index] = item, record
            del ids, generated
            print(json.dumps({'completed': len(raw), 'strict_complete': parsed is not None}), flush=True)
        assert len(raw) == len(canonical) == 58
        for name, records in [('responses.jsonl', raw), ('canonical.jsonl', canonical)]:
            with (root / name).open('x') as stream:
                for i in range(58):
                    stream.write(json.dumps(records[i], ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        save(root / 'complete.json', {'status': 'COMPLETE_FIXED_F0_ALL58_FRESH', 'at_utc': stamp(),
             'rows': 58, 'cached': 0, 'strict_complete': sum(x['strict_complete'] for x in canonical.values()),
             'responses': ref(root / 'responses.jsonl'), 'canonical': ref(root / 'canonical.jsonl'),
             'seconds': time.monotonic() - started, 'memory': memory(torch), 'quality_calls': 0,
             'training_steps': 0, 'gold_read': False})
    except BaseException as exc:
        save(root / 'failure.json', {'at_utc': stamp(), 'stage': stage, 'error': type(exc).__name__,
             'message': str(exc)[:3000], 'seconds': time.monotonic() - started,
             'memory': memory(torch) if torch is not None else None})
        raise


if __name__ == '__main__':
    main()
