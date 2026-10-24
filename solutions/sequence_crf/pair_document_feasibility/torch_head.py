"""Independent torch implementation matching the frozen NumPy head contract."""
import math
import torch
from torch import nn


class GlobalPairHead(nn.Module):
    def __init__(self, encoder_dim=768, width=128, hidden=128):
        super().__init__()
        self.U=nn.Parameter(torch.empty(2*encoder_dim,width));self.bq=nn.Parameter(torch.zeros(width))
        self.W=nn.Parameter(torch.empty(encoder_dim,width));self.bz=nn.Parameter(torch.zeros(width))
        self.A=nn.Parameter(torch.empty(3*width,hidden));self.ba=nn.Parameter(torch.zeros(hidden))
        self.o=nn.Parameter(torch.empty(hidden));self.bo=nn.Parameter(torch.zeros(()))
        for x in [self.U,self.W,self.A]:nn.init.xavier_uniform_(x)
        nn.init.normal_(self.o,std=hidden**-.5)

    def project(self,H,mask):
        assert mask.ndim==1 and H.ndim==2 and len(mask)==len(H) and bool(mask.any())
        return torch.tanh(H[mask]@self.W+self.bz)

    def from_projected(self,z,patient,concept,mode):
        assert mode in ('uniform','pair') and len(z)>0
        q=torch.tanh(torch.cat((patient,concept),-1)@self.U+self.bq)
        if mode=='pair':a=torch.softmax(q@z.T/math.sqrt(q.shape[-1]),-1)
        else:a=z.new_full((len(q),len(z)),1/len(z))
        c=a@z;f=torch.cat((q,c,q*c),-1);t=torch.tanh(f@self.A+self.ba)
        return t@self.o+self.bo,c,a

    def forward(self,H,patient,concept,mask,mode):
        return self.from_projected(self.project(H,mask),patient,concept,mode)


def endpoint(H,mentions,valid):
    assert mentions,'missing occurrence'
    vals=[]
    for positions in mentions:
        assert positions and all(isinstance(i,int) and 0<=i<len(H) and bool(valid[i]) for i in positions)
        vals.append(H[positions].mean(0))
    return torch.stack(vals).mean(0)


def endpoints(H,pairs,patients,concepts,valid):
    for pid,cid in pairs:
        if pid not in patients or cid not in concepts:raise ValueError('unknown candidate or patient')
    pp={pid:endpoint(H,mentions,valid) for pid,mentions in patients.items()}
    cc={cid:endpoint(H,mentions,valid) for cid,mentions in concepts.items()}
    return torch.stack([pp[pid] for pid,cid in pairs]),torch.stack([cc[cid] for pid,cid in pairs])


def class_loss(logits,targets,weights,denominator=None):
    assert logits.shape==targets.shape and bool(((targets==0)|(targets==1)).all())
    sample_weight=weights[targets.long()]
    if denominator is None:denominator=sample_weight.sum()
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')*sample_weight).sum()/denominator
