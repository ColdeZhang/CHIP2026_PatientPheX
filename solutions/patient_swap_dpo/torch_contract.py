"""Actual CPU tiny native Qwen3 + PEFT DPO gradient and update check."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch,transformers,peft
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    from dpo_ops import sequence_logp,dpo_loss,serial_dpo_backward,disable_dropout
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261024)
    assert not torch.cuda.is_available()
    assert transformers.__version__=='4.51.3' and peft.__version__=='0.15.2'
    events=[]
    def observe(x):
        events.append(x)
        with (out/'observations.jsonl').open('a') as f:
            f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    c=Qwen3Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                  num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=32,
                  attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2,
                  tie_word_embeddings=False,use_sliding_window=False,sliding_window=None)
    c._attn_implementation='sdpa'
    model=get_peft_model(Qwen3ForCausalLM(c).float(),LoraConfig(task_type='CAUSAL_LM',r=16,
                  lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none')).train()
    disabled=disable_dropout(model)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.normal_(0,.01)
    reference=copy.deepcopy(model).eval()
    chosen=torch.tensor([[1,3,4,5,7,8,2,9]])
    rejected=torch.tensor([[1,3,4,5,10,8,11,2,9]])
    prefix=4
    with torch.no_grad():
        rc=sequence_logp(reference,chosen,prefix);rr=sequence_logp(reference,rejected,prefix)
        for n,p in model.named_parameters():
            if 'lora_B' in n:p.add_(.0002)
    joint,serial=copy.deepcopy(model),copy.deepcopy(model)
    pc=sequence_logp(joint,chosen,prefix);pr=sequence_logp(joint,rejected,prefix)
    # Independently compare actual whole-vocab native logits and shifted sum.
    full=joint(input_ids=chosen,use_cache=False,return_dict=True).logits.float()
    expected=-torch.nn.functional.cross_entropy(full[0,prefix-1:-1],chosen[0,prefix:],reduction='sum')
    err=float((pc-expected).abs().detach());ok=bool(torch.allclose(pc,expected,rtol=1e-6,atol=1e-6))
    observe(dict(kind='indexed_vs_full_shifted_sum',max_abs=err,passed=ok));assert ok
    loss=dpo_loss(pc,pr,rc,rr);loss.backward()
    actual=serial_dpo_backward(serial,chosen,rejected,prefix,rc,rr,1,observe)
    comparisons=0;max_abs=0.;nonzero=0
    for (n,p),(ns,ps) in zip(joint.named_parameters(),serial.named_parameters()):
        assert n==ns
        if not p.requires_grad:
            assert p.grad is None and ps.grad is None;continue
        assert p.grad is not None and ps.grad is not None
        difference=float((p.grad-ps.grad).abs().max());finite=bool(torch.isfinite(p.grad).all() and torch.isfinite(ps.grad).all())
        passed=finite and bool(torch.allclose(p.grad,ps.grad,rtol=1e-5,atol=1e-7))
        observe(dict(kind='joint_vs_serial_parameter_gradient',name=n,max_abs=difference,finite=finite,passed=passed));assert passed
        comparisons+=1;max_abs=max(max_abs,difference);nonzero+=bool(p.grad.abs().sum()>0)
    before={n:p.detach().clone() for n,p in serial.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW([p for p in serial.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    norm=float(torch.nn.utils.clip_grad_norm_(serial.parameters(),1.));opt.step()
    changed=sum(not torch.equal(before[n],p) for n,p in serial.named_parameters() if p.requires_grad)
    observe(dict(kind='actual_tiny_Adam',grad_norm=norm,adapter_parameters_changed=changed,states=len(opt.state),passed=changed>0));assert changed>0
    result=dict(status='PASS_ACTUAL_CPU_QWEN3_PEFT_DPO_SERIAL_GRADIENT',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seconds=time.monotonic()-started,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ops_sha256=hashlib.sha256((Path(__file__).parent/'dpo_ops.py').read_bytes()).hexdigest(),
                disabled_dropout_modules=disabled,parameter_gradient_comparisons=comparisons,nonzero_gradient_parameters=nonzero,
                max_gradient_abs_error=max_abs,indexed_vs_full_sum_abs_error=err,joint_loss=float(loss.detach()),serial_observation=actual,
                one_actual_tiny_Adam_update=True,unequal_completion_lengths=True,base_has_no_grad=True,
                GPU_calls=0,pretrained_weights_loaded=False,competition_data_read=False,quality_calls=0,
                scope='Tiny FP32 CPU numerical/interface proof only. Full BF16 native model repeated-forward/memory/cost gate still required.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
