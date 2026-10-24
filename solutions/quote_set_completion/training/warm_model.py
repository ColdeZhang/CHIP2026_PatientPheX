"""Exact G32 d52 LoRA warm weights on the public base; never restore old Adam/RNG."""
from pathlib import Path
import hashlib,json,os,random

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_warm(model_path,manifest_path,seed,warm):
    import numpy as np
    import torch,transformers,peft
    from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
    import qwen35_runner as runner
    import qwen35_relation_cost as protocol
    assert sha(runner.__file__)=='33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669'
    assert sha(protocol.__file__)=='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
    assert transformers.__version__=='5.20.2' and peft.__version__=='0.21.2' and torch.__version__=='2.7.1+cu126'
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    directory=Path(warm['directory'])
    assert directory.is_absolute() and not directory.is_symlink()
    for name,item in warm['files'].items():
        path=directory/name
        assert not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    assert warm['files']['adapter_model.safetensors']['sha256']=='d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba'
    assert warm['files']['adapter_config.json']['sha256']=='a9e753c405c1a0b0f3811d54224437455629c9cd96d647cfe5f6cf8222dfd307'
    cfg=json.loads((directory/'adapter_config.json').read_text())
    assert cfg['r']==16 and cfg['lora_alpha']==32 and cfg['lora_dropout']==.05
    assert not cfg.get('use_dora') and not cfg.get('use_rslora') and cfg['bias']=='none'
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    files=runner.verify_model(model_path,manifest_path)
    config=Qwen3_5TextConfig.from_pretrained(model_path,local_files_only=True);config.use_cache=False
    base,loading=Qwen3_5ForCausalLM.from_pretrained(model_path,config=config,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=peft.PeftModel.from_pretrained(base,directory,is_trainable=True)
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(named)==496 and sum(p.numel() for p in named.values())==43278336
    assert sum(p.numel() for p in model.parameters())==8997081600<10_000_000_000
    head=model.get_base_model().lm_head
    assert type(head) is torch.nn.Linear and list(head.weight.shape)==[248320,4096]
    assert not any(p.requires_grad for p in head.parameters())
    assert all(('.lora_A.' in n or '.lora_B.' in n) and 'lm_head' not in n for n in named)
    assert {protocol.family(n) for n in named}=={'DeltaNet','full_attention','MLP'}
    assert all(bool(torch.isfinite(p).all()) for p in named.values())
    assert any(bool(torch.count_nonzero(p)) for n,p in named.items() if '.lora_B.' in n)
    return model,named,dict(model_files=files,loading=loading,warm_adapter=warm,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
            cuda=torch.version.cuda,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(0)),
        fresh_public_base=True,no_saved_adapter_loaded=False,existing_lm_head_frozen=True,
        warm_adapter_weights_only=True,old_optimizer_scheduler_RNG_loaded=False,fresh_optimizer_required=True)
