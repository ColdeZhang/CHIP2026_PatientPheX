"""Answer-free fixed-slot classification using the existing frozen Qwen LM head."""
from pathlib import Path
import gzip, hashlib, json, random, os

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    import qwen35_runner as runner
    with Path(path).open('x') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=runner.json_default)
        file.write('\n')

def rows(path, expected, training):
    assert sha(path) == expected
    values = json.loads(gzip.decompress(Path(path).read_bytes()))
    assert len(values) == (124 if training else 80)
    assert len({v['pmc_id'] for v in values}) == (32 if training else 16)
    assert len({v['row_id'] for v in values}) == len(values)
    for value in values:
        positions = value['slot_positions']
        assert 0 < len(value['input_ids']) <= 7168
        assert positions == sorted(set(positions)) and positions[-1] == len(value['input_ids'])-1
        assert len(positions) == len(value['keys']) and positions[0] >= value['prefix_tokens']
        assert 'answer' not in value and 'response' not in value and 'labels' not in value
        if training:
            assert len(value['targets']) == len(positions) and set(value['targets']) <= {0,1}
        else:
            assert 'targets' not in value
    assert sum(len(v['keys']) for v in values) == (2506 if training else 1557)
    if training:
        assert sum(map(lambda v:sum(v['targets']),values)) == 594
        assert sum(len(v['input_ids']) for v in values) == 424087
        assert max(len(v['input_ids']) for v in values) == 6777
    return values

def load_fresh(model_path, manifest_path, seed):
    import numpy as np
    import torch, transformers, peft
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    import qwen35_runner as runner
    import qwen35_relation_cost as protocol
    assert sha(runner.__file__) == '33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669'
    assert sha(protocol.__file__) == 'a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
    assert transformers.__version__ == '5.20.2' and peft.__version__ == '0.21.2'
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    base_files = runner.verify_model(model_path, manifest_path)
    config = Qwen3_5TextConfig.from_pretrained(model_path, local_files_only=True)
    config.use_cache = False
    base, loading = Qwen3_5ForCausalLM.from_pretrained(model_path, config=config,
        dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='sdpa',
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model, named, coverage = protocol.attach_adapter(base, torch, peft)
    assert len(named) > 0 and sum(p.numel() for p in named.values()) == 43278336
    head = model.get_base_model().lm_head
    assert type(head) is torch.nn.Linear and list(head.weight.shape) == [248320,4096]
    assert not any(p.requires_grad for p in head.parameters())
    assert all('lora_' in name and 'lm_head' not in name for name in named)
    return model, named, dict(model_files=base_files, loading=loading, coverage=coverage,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
            cuda=torch.version.cuda, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(0)),
        fresh_public_base=True, no_saved_adapter_loaded=True, existing_lm_head_frozen=True)

def tensor_input(row, torch):
    ids = torch.tensor([row['input_ids']], device='cuda', dtype=torch.long)
    positions = torch.tensor(row['slot_positions'], device='cuda', dtype=torch.long)
    return ids, positions

def decision_logits(model, ids, positions, torch):
    # Position s is the hidden state AFTER reading this question's delimiter.
    # There is deliberately no s-1 shift and no language-model labels argument.
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions, return_dict=True)
    assert output.past_key_values is None and output.logits.shape == (1,len(positions),248320)
    classes = torch.tensor([15,16],device=ids.device,dtype=torch.long)
    return output.logits[0].index_select(-1,classes).float()

def row_loss(model, row, torch):
    ids, positions = tensor_input(row,torch)
    logits = decision_logits(model,ids,positions,torch)
    targets = torch.tensor(row['targets'],device='cuda',dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits,targets,reduction='mean')
    assert torch.isfinite(loss)
    return loss, logits, targets
