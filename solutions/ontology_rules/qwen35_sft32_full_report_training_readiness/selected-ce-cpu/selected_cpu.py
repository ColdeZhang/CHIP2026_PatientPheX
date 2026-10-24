"""Target-environment hidden-CUDA CPU fixture only; no model or task data.

Owner executes one bounded CPU contract under the exact A800 runtime.
Both native CE and selected CE come from verified actual imported source files.
Synthetic logits only; no model construction, task inputs or gold.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import inspect
import importlib.util
import sys
import os
from pathlib import Path
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
LOSS_SHA = '83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'


def run(args):
    global torch, nn, selected_causal_loss
    began = time.monotonic(); signal.alarm(60)
    with (args.output.parent/'selected-CPU-intent.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),CPU_only=True),f);f.write('\n');f.flush();os.fsync(f.fileno())
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert torch.__version__ == '2.7.1+cu126'
    import transformers
    assert transformers.__version__ == '5.20.2'
    import transformers.loss.loss_utils as actual_native
    raw = args.loss_source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LOSS_SHA
    actual_path = Path(inspect.getsourcefile(actual_native))
    assert actual_path.read_bytes() == raw
    native = actual_native.ForCausalLMLoss
    native_fixed_cross_entropy = actual_native.fixed_cross_entropy
    selected_path = Path(__file__).parent / 'selected_loss.py'
    assert hashlib.sha256(selected_path.read_bytes()).hexdigest() == 'dfd0759fd43a181f7fbc4faa339a61a5495b3b1e6cadfae462f59a3d752d3cf9'
    spec = importlib.util.spec_from_file_location('selected_loss', selected_path)
    selected_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = selected_module; spec.loader.exec_module(selected_module)
    assert selected_module.native_fixed_cross_entropy is native_fixed_cross_entropy
    selected_causal_loss = selected_module.selected_causal_loss
    function_shas = {name:hashlib.sha256(inspect.getsource(getattr(actual_native,name)).strip().encode()).hexdigest()
                     for name in ['fixed_cross_entropy','ForCausalLMLoss']}
    g = torch.Generator().manual_seed(20261024)
    labels = {
        'right_padding': [[-100,-100,2,1,7,-100,-100],[-100,4,3,-100,-100,-100,-100]],
        'left_padding_interior_mask': [[-100,-100,-100,2,-100,5,7],[-100,-100,4,-100,0,-100,3]],
        'one_supervised_last_token': [[-100,-100,-100,-100,-100,-100,7],[-100]*7],
        'all_tokens_labeled_t0_not_predicted': [[2,3,4,5,6,7,0],[7,6,5,4,3,2,1]],
        'all_ignore': [[-100]*7,[-100]*7],
        'only_t0_ignored_after_shift': [[4,-100,-100,-100,-100,-100,-100],[-100]*7],
    }
    results=[]
    for dtype in [torch.float32,torch.bfloat16]:
        for scale in [1e-4,1.,50.,1e4]:
            for name, lab in labels.items():
                lab=torch.tensor(lab,dtype=torch.long)
                z=(torch.randn((2,7,8),generator=g)*scale).to(dtype)
                for denominator_name, denominator in [('mean',None),('num_items_int',23),('num_items_tensor',torch.tensor(23))]:
                    for supplied_shift in [False,True]:
                        shift_labels=nn.functional.pad(lab,(0,1),value=-100)[...,1:].contiguous() if supplied_shift else None
                        a=z.clone().requires_grad_(); b=z.clone().requires_grad_()
                        la=native(a,lab,8,num_items_in_batch=denominator,shift_labels=shift_labels)
                        lb=selected_causal_loss(b,lab,8,num_items_in_batch=denominator,shift_labels=shift_labels)
                        ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
                        torch.testing.assert_close(la,lb,rtol=2e-6,atol=2e-6,equal_nan=True)
                        torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7,equal_nan=True)
                        target=nn.functional.pad(lab,(0,1),value=-100)[...,1:]
                        assert torch.count_nonzero(ga[target==-100])==0
                        n=int((target!=-100).sum())
                        assert bool(torch.isnan(la)) == (n==0 and denominator is None)
                        results.append(dict(case=name,dtype=str(dtype),scale=scale,denominator=denominator_name,shift_labels_supplied=supplied_shift,supervised=n,native_nan=bool(torch.isnan(la)),loss_abs_error=None if torch.isnan(la) else float(abs(la-lb)),grad_max_abs_error=float((ga.float()-gb.float()).abs().max())))
    # Same complete head shape for both paths, then actual autograd through it.
    heads=[]
    for dtype in [torch.float32,torch.bfloat16]:
        h=torch.randn((2,7,4),generator=g).to(dtype);w=torch.randn((4,8),generator=g).to(dtype)
        lab=torch.tensor(labels['right_padding'],dtype=torch.long)
        h1=h.clone().requires_grad_();w1=w.clone().requires_grad_()
        h2=h.clone().requires_grad_();w2=w.clone().requires_grad_()
        za=h1@w1;zb=h2@w2;assert torch.equal(za,zb)
        a=native(za,lab,8);b=selected_causal_loss(zb,lab,8)
        ga=torch.autograd.grad(a,(h1,w1));gb=torch.autograd.grad(b,(h2,w2))
        torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-7)
        heads.append(dict(dtype=str(dtype),hidden_grad_max_abs_error=float((ga[0].float()-gb[0].float()).abs().max()),head_grad_max_abs_error=float((ga[1].float()-gb[1].float()).abs().max()),complete_same_shape_head_logits_exact=True))
    # Explicit accumulation: unchanged per-example CE mean, average over actual
    # group length (including the one-row final group), never pooled token mean.
    accum=[]
    for count in [1,2,4,8]:
        z=torch.randn((count,7,8),generator=g,dtype=torch.float32)
        lab=torch.tensor([labels['right_padding'][i%2] for i in range(count)],dtype=torch.long)
        a=z.clone().requires_grad_();b=z.clone().requires_grad_()
        la=sum(native(a[i:i+1],lab[i:i+1],8)/count for i in range(count))
        lb=sum(selected_causal_loss(b[i:i+1],lab[i:i+1],8)/count for i in range(count))
        ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
        torch.testing.assert_close(la,lb,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7)
        accum.append(dict(group_size=count,loss_abs_error=float(abs(la-lb)),grad_max_abs_error=float((ga-gb).abs().max())))
    # Perturb only causally ignored positions; both valid gradients and loss
    # must remain fixed. This distinguishes masking from a coincidental loss match.
    perturb=[]
    for dtype in [torch.float32,torch.bfloat16]:
        lab=torch.tensor(labels['left_padding_interior_mask'],dtype=torch.long)
        target=nn.functional.pad(lab,(0,1),value=-100)[...,1:]
        z=torch.randn((2,7,8),generator=g).to(dtype)
        poisoned=z.clone();poisoned[target==-100]=torch.arange(8).to(dtype)*1000
        for name,fn in [('native',native),('selected',selected_causal_loss)]:
            a=z.clone().requires_grad_();b=poisoned.clone().requires_grad_()
            la=fn(a,lab,8);lb=fn(b,lab,8)
            ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
            torch.testing.assert_close(la,lb,rtol=0,atol=0)
            assert torch.equal(ga,gb) and torch.count_nonzero(gb[target==-100])==0
            perturb.append(dict(dtype=str(dtype),implementation=name,ignored_logits_perturbed=True,loss_exact=True,gradients_exact=True))
    custom=[]
    for dtype in [torch.float32,torch.bfloat16]:
        lab=torch.tensor(labels['right_padding'],dtype=torch.long);lab[lab==-100]=-7
        z=torch.randn((2,7,8),generator=g).to(dtype)
        for supplied_shift in [False,True]:
            shifted=nn.functional.pad(lab,(0,1),value=-7)[...,1:].contiguous() if supplied_shift else None
            a=z.clone().requires_grad_();b=z.clone().requires_grad_()
            la=native(a,lab,8,ignore_index=-7,shift_labels=shifted)
            lb=selected_causal_loss(b,lab,8,ignore_index=-7,shift_labels=shifted)
            ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
            torch.testing.assert_close(la,lb,rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7)
            custom.append(dict(dtype=str(dtype),ignore_index=-7,shift_labels_supplied=supplied_shift,loss_abs_error=float(abs(la-lb)),grad_max_abs_error=float((ga.float()-gb.float()).abs().max())))
    elapsed=time.monotonic()-began;assert elapsed<60 and not torch.cuda.is_initialized()
    result=dict(status='PASS_ACTUAL_A800_TORCH271_CPU_SELECTED_CE_SYNTHETIC_CONTRACT_NO_MODEL',at_utc=datetime.now(timezone.utc).isoformat(),seconds=elapsed,torch=torch.__version__,transformers=transformers.__version__,actual_native_source_path=str(actual_path),selected_source_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),actual_loss_source_sha256=LOSS_SHA,exact_native_function_SHA=function_shas,loss_rtol=2e-6,loss_atol=2e-6,gradient_rtol=2e-5,gradient_atol=2e-7,cases=len(results),results=results,full_head_gradient=heads,accumulation=accum,ignored_position_perturbations=perturb,custom_ignore=custom,model_constructed=False,training_input_read=False,gold_read=False,GPU_operation=False,limits='CPU tiny fixtures only; no model weights, task input/gold, GPU, optimizer or formal run. No claim of GPU bitwise equivalence or peak memory success.')
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),seconds=elapsed,cases=len(results))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--loss-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
