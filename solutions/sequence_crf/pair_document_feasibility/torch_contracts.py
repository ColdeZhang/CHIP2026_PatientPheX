#!/usr/bin/env python3
import argparse,json,time,datetime,hashlib,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from global_head import initialize,loss_and_grad
from torch_head import GlobalPairHead,endpoints,class_loss


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();tick=time.monotonic();torch.set_num_threads(1)
    rng=np.random.default_rng(7);raw=rng.normal(size=(5,4));mask=np.array([1,1,0,1,0],dtype=bool);pn=initialize(4,3,5);out=[]
    for mode in ['uniform','pair']:
        for target in [0,1]:
            model=GlobalPairHead(4,3,5).double()
            with torch.no_grad():
                for k,p in model.named_parameters():p.copy_(torch.from_numpy(np.asarray(pn[k])))
            H=torch.tensor(raw,dtype=torch.double,requires_grad=True);p=torch.tensor(raw[0:1],dtype=torch.double,requires_grad=True);c=torch.tensor(raw[1:2],dtype=torch.double,requires_grad=True);m=torch.tensor(mask)
            logits,context,attn=model(H,p,c,m,mode);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.full_like(logits,target));loss.backward()
            nl,ng,ni=loss_and_grad(pn,raw,raw[0],raw[1],mask,mode,target)
            maxerr=max(abs(float(loss)-nl),max(float(np.max(abs(v.grad.numpy()-ng[k]))) for k,v in model.named_parameters()),float(np.max(abs(H.grad.numpy()-ni['H']))),float(np.max(abs(p.grad.numpy()[0]-ni['patient']))),float(np.max(abs(c.grad.numpy()[0]-ni['concept']))))
            assert maxerr<1e-12 and all(float(v.grad.norm())>0 for v in model.parameters())
            poisoned=H.detach().clone();poisoned[~m]=float('nan');lp,cp,ap=model(poisoned,p.detach(),c.detach(),m,mode)
            assert torch.equal(lp,logits.detach()) and torch.equal(cp,context.detach()) and torch.equal(ap,attn.detach())
            assert torch.count_nonzero(H.grad[~m])==0
            out.append(dict(mode=mode,target=target,numpy_autograd_max_error=maxerr,all_parameter_gradients_nonzero=True,padding_nan_exact=True))
    H=torch.arange(12,dtype=torch.float64).reshape(3,4).requires_grad_();valid=torch.ones(3,dtype=torch.bool)
    pp,cc=endpoints(H,[('P','C')],{'P':[[0,1],[2]]},{'C':[[1,2]]},valid)
    (pp.sum()+cc.sum()).backward();assert torch.allclose(H.grad[:,0],torch.tensor([.25,.75,1.],dtype=torch.double))
    try:endpoints(H,[('P','UNKNOWN')],{'P':[[0]]},{'C':[[1]]},valid)
    except ValueError:pass
    else:raise AssertionError('unknown accepted')
    # Exact old CE weighted-mean equivalent on arbitrary document batch, plus chunked denominator.
    logits=torch.tensor([-2.,.1,3.,-.8],requires_grad=True);y=torch.tensor([0.,1.,1.,0.]);w=torch.tensor([2506/(2*1912),2506/(2*594)])
    new=class_loss(logits,y,w);old=torch.nn.functional.cross_entropy(torch.stack((torch.zeros_like(logits),logits),-1),y.long(),weight=w)
    gnew=torch.autograd.grad(new,logits,retain_graph=True)[0];gold=torch.autograd.grad(old,logits,retain_graph=True)[0]
    chunks=sum(class_loss(logits[i:i+2],y[i:i+2],w,w[y.long()].sum()) for i in range(0,4,2))
    assert torch.allclose(new,old,atol=1e-7) and torch.allclose(gnew,gold,atol=1e-7) and torch.allclose(new,chunks,atol=1e-7)
    # Differentiable cross-window duplicate average, independent of padded state poison.
    a1=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True);a2=torch.tensor([[5.,6.],[7.,8.]],requires_grad=True)
    summed=torch.zeros(3,2).index_add(0,torch.tensor([0,1]),a1).index_add(0,torch.tensor([1,2]),a2);avg=summed/torch.tensor([1.,2.,1.])[:,None]
    assert torch.equal(avg,torch.tensor([[1.,2.],[4.,5.],[7.,8.]]));avg.sum().backward();assert torch.equal(a1.grad,torch.tensor([[1.,1.],[.5,.5]]))
    # Cross-window no shared local support does not constrain learned global context.
    model=GlobalPairHead(2,2,2).double()
    with torch.no_grad():model.W.copy_(torch.eye(2));model.bz.zero_();model.U.zero_();model.U[:2].copy_(torch.eye(2));model.bq.zero_()
    H=torch.tensor([[2.,0.],[0.,3.],[1.,1.]],dtype=torch.double);m=torch.ones(3,dtype=torch.bool)
    _,c1,_=model(H,H[:1],H[1:2],m,'pair');_,c2,_=model(H,H[1:2],H[1:2],m,'pair')
    assert c1.norm()>0 and not torch.allclose(c1,c2)
    full=GlobalPairHead();assert sum(p.numel() for p in full.parameters())==344577
    result=dict(status='ACTUAL_CPU_AUTOGRAD_PASS',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-tick,torch=torch.__version__,numpy=np.__version__,CUDA_initialized=torch.cuda.is_initialized(),sources={p.name:sha(p) for p in Path('.').glob('*.py')},contracts=out,overlap_gradients_exact=True,unknown_rejected=True,weighted_CE_loss_and_gradient_equivalent=True,chunked_document_denominator_equivalent=True,duplicate_window_mean_gradient_exact=True,cross_window_global_nonzero_patient_sensitive=True,head_parameters=344577,no_model_weights_loaded=True)
    assert not result['CUDA_initialized'];Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
