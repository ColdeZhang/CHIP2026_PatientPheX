"""Tiny new-loss contract, no pretrained weights/data/CUDA/generation claim."""
import argparse,copy,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import atomic,ref
from scst_ops import token_logps,replay_observation,disable_dropout

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from types import SimpleNamespace
    torch.set_num_threads(1);torch.manual_seed(19)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(17,8);self.proj=torch.nn.Linear(8,17);self.drop=torch.nn.Dropout(.05);self.calls=0
        def forward(self,input_ids,attention_mask,labels,use_cache,logits_to_keep,return_dict):
            assert not use_cache and labels is None and attention_mask.shape==input_ids.shape
            hidden=self.emb(input_ids)
            def core(h):self.calls+=1;return self.proj(self.drop(h))
            y=checkpoint(core,hidden,use_reentrant=False)
            return SimpleNamespace(logits=y[:,logits_to_keep,:])
    base=Tiny().train();disable_dropout(base);model=copy.deepcopy(base)
    ids=torch.tensor([[3,4,5,6,2]]);labels=ids.clone();labels[:,:2]=-100
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):ng=token_logps(model,ids,labels)
    with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):lp=token_logps(model,ids,labels)
    obs=replay_observation(ng,lp);assert obs['allclose_rtol_atol_1e5'] and lp.numel()==3
    # CE anchor and positive/negative/zero advantages use full suffix including2.
    actual=[]
    for advantage in (-.3,0.,.4):
        m=copy.deepcopy(base);serial=copy.deepcopy(base)
        with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
            jointlp=token_logps(m,ids,labels);joint=-jointlp.mean()/4-advantage/(11884/87)/4*jointlp.sum()
        joint.backward()
        with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):s=token_logps(serial,ids,labels);pg=-advantage/(11884/87)/4*s.sum()
        pg.backward()
        with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):s=token_logps(serial,ids,labels);ce=-s.mean()/4
        ce.backward()
        # Separate BF16 matmuls round independently: observe, do not claim byte identity.
        diffs={n:float((p.grad-dict(serial.named_parameters())[n].grad).abs().max()) for n,p in m.named_parameters()}
        assert all(bool(torch.isfinite(p.grad).all()) for p in serial.parameters())
        actual.append(dict(advantage=advantage,joint_vs_serial_gradient_max=max(diffs.values()),all_finite=True))
    model.zero_grad(set_to_none=True)
    loss=-lp.mean()-.4/(11884/87)*lp.sum();loss.backward();before=model.proj.weight.detach().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    assert torch.isfinite(norm) and bool((model.proj.weight-before).abs().max()>0)
    atomic(a.output,dict(status='PASS_TINY_NEW_SUM_SUFFIX_AND_CE_SERIAL_CONTRACT',source=ref(__file__),ops=ref(Path(__file__).parent/'scst_ops.py'),torch=torch.__version__,BF16_CPU=True,
         same_full_no_grad_grad=obs,advantage_signs=actual,checkpoint_actual_calls=model.calls,one_Adam_changed=True,model_weights_downloaded=False,CUDA_calls=0,
         limitation='Tiny CPU contract only; not actual9B generation/replay, CUDA numerics, memory or semantic signal'))
if __name__=='__main__':main()
