"""Target-environment hidden-CUDA CPU fixture only; no model or task data.

Owner executes this after copying the exact actual loss_utils.py alongside it.
Only the two verified causal-CE functions are AST-extracted, avoiding unrelated
model imports. This file is not a formal trainer or a GPU launcher.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
LOSS_SHA = '83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'


def selected_causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                         ignore_index=-100, shift_labels=None, **kwargs):
    # Keep the complete original BF16 head output. No selected hidden states,
    # logits_to_keep, model forward or head GEMM changes occur in this function.
    if shift_labels is None:
        padded = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = padded[..., 1:].contiguous()
    target = shift_labels.view(-1).to(logits.device)
    keep = target != ignore_index
    source = logits.view(-1, vocab_size)[keep].float()
    # Empty selected rows retain native CE mean NaN or sum/explicit-denominator
    # behavior. No invented zero-loss success or group/token reweighting.
    return native_fixed_cross_entropy(source, target[keep], num_items_in_batch,
                                      ignore_index, **kwargs)


def run(args):
    global torch, nn, native_fixed_cross_entropy
    began = time.monotonic(); signal.alarm(120)
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert torch.__version__ == '2.6.0+cu124'
    raw = args.loss_source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LOSS_SHA
    text = raw.decode(); tree = ast.parse(text)
    names = ['fixed_cross_entropy', 'ForCausalLMLoss']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert [n.name for n in nodes] == names
    env = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(args.loss_source),'exec'),env)
    native = env['ForCausalLMLoss']; native_fixed_cross_entropy = env['fixed_cross_entropy']
    function_shas = {n.name: hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest() for n in nodes}
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
    for count in [1,2,8]:
        z=torch.randn((count,7,8),generator=g,dtype=torch.float32)
        lab=torch.tensor([labels['right_padding'][i%2] for i in range(count)],dtype=torch.long)
        a=z.clone().requires_grad_();b=z.clone().requires_grad_()
        la=sum(native(a[i:i+1],lab[i:i+1],8)/count for i in range(count))
        lb=sum(selected_causal_loss(b[i:i+1],lab[i:i+1],8)/count for i in range(count))
        ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
        torch.testing.assert_close(la,lb,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-7)
        accum.append(dict(group_size=count,loss_abs_error=float(abs(la-lb)),grad_max_abs_error=float((ga-gb).abs().max())))
    elapsed=time.monotonic()-began;assert elapsed<120 and not torch.cuda.is_initialized()
    result=dict(status='PASS_actual_Torch_CPU_fixed_full_logits_selected_CE_contract',at_utc=datetime.now(timezone.utc).isoformat(),seconds=elapsed,torch=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),actual_loss_source_sha256=LOSS_SHA,exact_native_function_SHA=function_shas,loss_rtol=2e-6,loss_atol=2e-6,gradient_rtol=2e-5,gradient_atol=2e-7,cases=len(results),results=results,full_head_gradient=heads,accumulation=accum,limits='CPU tiny fixtures only; no model weights, task input/gold, GPU, optimizer or formal run. No claim of GPU bitwise equivalence or peak memory success.')
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),seconds=elapsed,cases=len(results))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--loss-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
