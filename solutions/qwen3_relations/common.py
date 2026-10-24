"""Shared immutable identity, native Qwen3 loading and durable state helpers."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save(path, value):
    def convert(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item, key=repr)
        raise TypeError(type(item).__name__)
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=convert)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def update(path, value):
    temporary = Path(str(path) + '.next')
    save(temporary, value)
    os.replace(temporary, path)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked(item):
    assert ref(item['path']) == item, item['path']
    return Path(item['path'])


def source_gate(plan):
    for name, digest in plan['source_files'].items():
        assert sha(Path(__file__).parent / name) == digest, name


def gpu_gate(plan, output):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    rows = [line.split(', ') for line in gpu.splitlines() if plan['gpu_uuid'] in line]
    observation = {'at_utc': stamp(), 'gpu': gpu, 'compute_apps': apps,
                   'disk_free_bytes': shutil.disk_usage(output).free}
    save(Path(output) / 'fresh-gate.json', observation)
    assert len(rows) == 1 and rows[0][0] == '1' and rows[0][2] == 'NVIDIA A30'
    assert int(rows[0][-1]) >= 23500
    assert not any(line.split(',')[0].strip() == plan['gpu_uuid'] for line in apps.splitlines())
    assert observation['disk_free_bytes'] >= 3 * 1024**3


def verify_assets(plan, *, full, previous=None):
    result = {}
    for name, expected in plan['model_files'].items():
        path = Path(plan['model_dir']) / name
        state = path.stat()
        assert state.st_size == expected['size'], name
        item = {'bytes': state.st_size, 'mtime_ns': state.st_mtime_ns, 'inode': state.st_ino}
        if full:
            item['sha256'] = sha(path)
            assert item['sha256'] == expected['sha256'], name
        else:
            assert previous is not None
            assert all(item[key] == previous[name][key] for key in item), name
            item['sha256'] = previous[name]['sha256']
            assert item['sha256'] == expected['sha256']
        result[name] = item
    return result


def load_model(plan, output, adapter=None):
    """Frozen BF16 public base, FP32 rank16 adapters; no dtype fallback."""
    import numpy as np
    import torch
    import transformers
    import peft
    import tokenizers
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '2.7.1+cu126' and transformers.__version__ == '4.51.3'
    assert peft.__version__ == plan['runtime']['peft']
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(plan['seed'])
    np.random.seed(plan['seed'])
    torch.manual_seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    started = time.monotonic()
    base, loading = AutoModelForCausalLM.from_pretrained(
        plan['model_dir'], torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False,
        output_loading_info=True)
    save(Path(output) / 'loading.json', loading)
    assert not any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
    assert type(base).__name__ == 'Qwen3ForCausalLM'
    assert sum(p.numel() for p in base.parameters()) == 8190735360
    assert base.config.rope_scaling is None and base.config.num_hidden_layers == 36
    forward_sha = hashlib.sha256(inspect.getsource(type(base).forward).encode()).hexdigest()
    assert forward_sha == plan['native_forward_sha256']
    if adapter is None:
        model = get_peft_model(base, LoraConfig(**plan['lora']))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        named = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert len(named) == 504 and sum(p.numel() for p in named.values()) == 43646976
        assert all('.lora_A.' in name or '.lora_B.' in name for name in named)
        assert all(p.dtype == torch.float32 for p in named.values())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
    else:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        named = {}
        model.eval()
        model.config.use_cache = True
    assert sum(p.numel() for p in model.parameters()) == 8234382336 < 10000000000
    assert not base.lm_head.weight.requires_grad and not base.get_input_embeddings().weight.requires_grad
    manifest = {'at_utc': stamp(), 'base_parameters': 8190735360, 'lora_parameters': 43646976,
                'total_parameters': 8234382336, 'trainable_tensors': len(named), 'adapter': adapter,
                'native_forward_sha256': forward_sha,
                'precision': 'BF16 frozen base and FP32 LoRA; full unquantized output head',
                'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'peft': peft.__version__, 'tokenizers': tokenizers.__version__},
                'load_seconds': time.monotonic() - started, 'TF32': False}
    save(Path(output) / 'model-manifest.json', manifest)
    return torch, np, model, named, manifest


def memory(torch):
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
            'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
            'free_bytes': torch.cuda.mem_get_info()[0]}


def batch(torch, row):
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
            'labels': torch.tensor([row['labels']], dtype=torch.long, device='cuda')}


def checkpoint(torch, np, model, optimizer, scheduler, output, epoch, updates, counts, plan):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    start = time.monotonic()
    out = Path(output)
    temporary = out / f'.epoch-{epoch:03d}.incomplete'
    target = out / f'epoch-{epoch:03d}'
    temporary.mkdir(exist_ok=False)
    assert not target.exists()
    model.save_pretrained(temporary / 'adapter', safe_serialization=True)
    state = {'completed_epochs': epoch, 'global_step': updates, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
    with (temporary / 'training_state.pt').open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    for path in temporary.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    files = {str(path.relative_to(temporary)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(temporary.rglob('*')) if path.is_file()}
    save(temporary / 'checkpoint.json', {'epoch': epoch, 'global_step': updates, 'counts': counts,
         'files': files, 'recipe': plan['recipe'], 'model': plan['model_id'], 'at_utc': stamp()})
    sync_dir(temporary / 'adapter')
    sync_dir(temporary)
    temporary.rename(target)
    sync_dir(out)
    for name, expected in files.items():
        assert sha(target / name) == expected['sha256'] and (target / name).stat().st_size == expected['bytes']
    saved_adapter = load_file(str(target / 'adapter/adapter_model.safetensors'), device='cpu')
    actual = get_peft_model_state_dict(model)
    assert set(saved_adapter) == set(actual) and len(saved_adapter) == 504
    assert all(torch.equal(value, actual[name].detach().cpu()) for name, value in saved_adapter.items())
    reopened = torch.load(target / 'training_state.pt', map_location='cpu', weights_only=False)
    assert reopened['completed_epochs'] == epoch and reopened['global_step'] == updates == epoch * 16
    assert reopened['scheduler'] == scheduler.state_dict()
    assert reopened['optimizer']['param_groups'] == optimizer.state_dict()['param_groups']
    current = optimizer.state_dict()['state']
    assert set(reopened['optimizer']['state']) == set(current) and len(current) == 504
    tensor_count = 0
    for key, values in current.items():
        for name, value in values.items():
            other = reopened['optimizer']['state'][key][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(other, value.detach().cpu()) and bool(torch.isfinite(other).all())
                tensor_count += 1
            else:
                assert other == value
    assert reopened['python_rng'] == state['python_rng']
    assert reopened['numpy_rng'][0] == state['numpy_rng'][0]
    assert np.array_equal(reopened['numpy_rng'][1], state['numpy_rng'][1])
    assert reopened['numpy_rng'][2:] == state['numpy_rng'][2:]
    assert torch.equal(reopened['torch_rng'], state['torch_rng'])
    assert all(torch.equal(a, b) for a, b in zip(reopened['cuda_rng'], state['cuda_rng']))
    receipt = {'status': 'DURABLE_COMPLETE_EPOCH_ALL_ADAPTER_ADAM_RNG_REOPEN_EXACT', 'at_utc': stamp(),
               'epoch': epoch, 'global_step': updates, 'adapter_tensors': 504, 'optimizer_states': 504,
               'optimizer_tensor_entries': tensor_count, 'all_rng_exact': True,
               'checkpoint': ref(target / 'checkpoint.json'), 'seconds': time.monotonic() - start,
               'files': {name: {'path': str(target / name), **entry} for name, entry in files.items()}}
    save(out / f'epoch-{epoch:03d}-persistent-ack.json', receipt)
    return receipt
