"""Persist actual A100 native runtime identity without model or CUDA loading."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import torch
import transformers
import peft
import transformers.models.qwen3_5.modeling_qwen3_5 as q

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
assert transformers.__version__ == '5.20.2' and peft.__version__ == '0.21.2'
assert importlib.util.find_spec('fla') is None
bindings = {n:dict(module=getattr(q,n).__module__, unwrapped=inspect.unwrap(getattr(q,n)).__module__)
            for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']}
assert all(v['module'] == v['unwrapped'] == q.__name__ for v in bindings.values())
digest = hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()
assert digest == 'b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
report = dict(status='PASS_actual_A100_native_CPU_runtime', torch=torch.__version__,
              transformers=transformers.__version__, peft=peft.__version__, FLA=False,
              cuda_initialized=False, allocator_environment={k:os.environ.get(k) for k in ['PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF']}, bindings=bindings, modeling_path=q.__file__, modeling_sha256=digest,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), no_weights=True)
with Path('runtime-cpu.json').open('x') as f:
    json.dump(report, f, indent=2); f.write('\n')
print(json.dumps(report), flush=True)
