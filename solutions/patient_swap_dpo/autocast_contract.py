"""CPU-only diagnosis of mixed no-grad/grad autocast checkpoint cache."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch, transformers, peft
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from dpo_ops import sequence_logp, serial_dpo_backward, disable_dropout
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261024)
    assert not torch.cuda.is_available()
    out=Path(output); out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    c=Qwen3Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                 num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=32,
                 attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2,
                 tie_word_embeddings=False,use_sliding_window=False,sliding_window=None)
    c._attn_implementation='sdpa'
    base=get_peft_model(Qwen3ForCausalLM(c).to(torch.bfloat16),LoraConfig(task_type='CAUSAL_LM',r=16,
                       lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none'))
    disable_dropout(base)
    for n,p in base.named_parameters():
        if p.requires_grad:
            p.data=p.data.float()
            if 'lora_B' in n:
                with torch.no_grad():p.normal_(0,.01)
    base.train(); base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    base.enable_input_require_grads(); base.config.use_cache=False
    chosen=torch.tensor([[1,3,4,5,7,8,2,9]]); rejected=torch.tensor([[1,3,4,5,10,8,11,2,9]])
    results=[]
    for cache in [True,False]:
        model=copy.deepcopy(base); events=[]; tick=time.monotonic()
        def observe(x):
            events.append(x)
            with (out/('observations-cache-'+str(cache)+'.jsonl')).open('a') as f:
                f.write(json.dumps(x,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        result={'cache_enabled':cache}
        try:
            with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=cache):
                values=serial_dpo_backward(model,chosen,rejected,4,-14.,-17.,1,observe)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            grads={n:p.grad for n,p in named.items()}
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            before={n:p.detach().clone() for n,p in named.items()}
            opt=torch.optim.AdamW(list(named.values()),lr=1e-5,weight_decay=.01)
            norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.); opt.step()
            result.update(status='PASS_SERIAL_BF16_CHECKPOINT_AND_ADAM',values=values,
                          gradients=len(grads),nonzero=sum(bool(g.abs().sum()>0) for g in grads.values()),
                          changed=sum(not torch.equal(before[n],p) for n,p in named.items()),
                          grad_norm=float(norm),Adam_states=len(opt.state))
        except Exception as e:
            trace=traceback.format_exc(); (out/('failure-cache-'+str(cache)+'.txt')).write_text(trace)
            result.update(status='FAILED_ORIGINAL_EVIDENCE',error_type=type(e).__name__,error_head=str(e)[:250])
        result.update(seconds=time.monotonic()-tick,replay=events); results.append(result)
    value={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,
           'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'ops_sha256':hashlib.sha256((Path(__file__).parent/'dpo_ops.py').read_bytes()).hexdigest(),
           'results':results,'GPU_calls':0,'pretrained_weights_loaded':False,'competition_data_read':False,'quality_calls':0,
           'scope':'One cache flag CPU BF16 checkpoint diagnosis; not proof of full CUDA success.'}
    with (out/'result.json').open('x') as f:json.dump(value,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
