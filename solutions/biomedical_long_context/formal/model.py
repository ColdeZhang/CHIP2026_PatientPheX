"""Full-context encoder, original GP equations, local original-passage outputs."""
import math
import torch
from torch import nn


def rotate(x, positions):
    half=x.shape[-1]//2
    inv=torch.pow(10000.,-torch.arange(half,device=x.device).float()/half)
    phase=positions.float()[:,None]*inv[None]
    c,s=phase.cos(),phase.sin();even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*c-odd*s,even*s+odd*c),-1).flatten(-2)


def pair_loss(score, positive):
    n=score.shape[0];valid=torch.ones((n,n),dtype=torch.bool,device=score.device).triu()
    pos=torch.zeros_like(valid)
    if positive:
        ij=torch.tensor(positive,device=score.device,dtype=torch.long);pos[ij[:,0],ij[:,1]]=True
    assert not (pos&~valid).any()
    zero=score.new_zeros(1)
    return torch.logsumexp(torch.cat(((-score).masked_fill(~pos,-torch.inf).flatten(),zero)),0)+torch.logsumexp(torch.cat((score.masked_fill(~(valid&~pos),-torch.inf).flatten(),zero)),0)


class FullContextGP(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.head=nn.Linear(encoder.config.hidden_size,128)
        nn.init.normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)

    def scores(self,row):
        device=self.head.weight.device;ids=torch.tensor([row['ids']],device=device)
        hidden=self.encoder(input_ids=ids,attention_mask=torch.ones_like(ids),
                            output_hidden_states=False,output_attentions=False).last_hidden_state[0]
        result=[]
        with torch.autocast(device_type=device.type,enabled=False):
            q,k=self.head(hidden.float()).chunk(2,-1)
            for b in row['blocks']:
                ind=torch.tensor(b['indices'],device=device,dtype=torch.long)
                pos=torch.tensor(b['positions'],device=device,dtype=torch.float32)
                qi=rotate(q.index_select(0,ind),pos);ki=rotate(k.index_select(0,ind),pos)
                result.append((qi@ki.T)/math.sqrt(64))
        return result

    def forward(self,row):
        scores=self.scores(row)
        losses=[pair_loss(x,b['positive']) for x,b in zip(scores,row['blocks'])]
        return torch.stack(losses).mean()


def load_model(path,loading_path):
    from transformers import AutoModel
    from common import save
    enc,info=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,
             attn_implementation='sdpa',reference_compile=False,output_loading_info=True)
    save(loading_path,info)
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
    assert all(k.startswith('head.') or k.startswith('decoder.') for k in info.get('unexpected_keys',[])),info
    assert sum(p.numel() for p in enc.parameters())==149014272
    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=FullContextGP(enc)
    assert sum(p.numel() for p in model.parameters())==149112704
    return model.cuda()
