#!/usr/bin/env python3
"""Bounded cause check and disposable BF16 verification; never evaluates gold."""
from pathlib import Path
import argparse, gc, hashlib, json, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_encoded_ce as t


def cpu_contract():
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.SGD([p], lr=.1)
    scale = torch.amp.GradScaler('cpu')
    loss = p.square().sum()
    scale.scale(loss).backward()
    p.grad.fill_(float('inf'))
    before = p.detach().clone()
    old_scale = scale.get_scale()
    scale.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_([p], 1.)
    scale.step(opt)
    scale.update()
    t.require(not bool(torch.isfinite(norm)) and torch.equal(p, before) and scale.get_scale() == old_scale / 2,
              'CPU AMP overflow did not skip/backoff')
    linear = torch.nn.Linear(2, 2)
    adam = torch.optim.AdamW(linear.parameters(), lr=.01)
    before = linear.weight.detach().clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = linear(torch.tensor([[1.,2.],[-1.,-2.]]))
        loss = t.weighted_ce(logits, torch.tensor([0,1]), torch.ones(2))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(linear.parameters(), 1.)
    t.require(bool(torch.isfinite(loss)) and bool(torch.isfinite(norm)), 'BF16 CPU finite check failed')
    adam.step()
    t.require(not torch.equal(linear.weight, before), 'BF16 CPU did not update')
    t.require(all(x.dtype == torch.float32 for x in linear.parameters()), 'Parameter dtype drift')
    return dict(status='pass_cpu_numeric_contract', torch_version=torch.__version__,
                injected_inf_native_scaler_skips=True, scale_before=old_scale, scale_after=scale.get_scale(),
                bf16_loss=float(loss.detach()), bf16_gradient_norm=float(norm), bf16_parameters_updated=True,
                bf16_parameters_fp32=True, cuda_initialized=torch.cuda.is_initialized())


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',action='store_true');a=p.parse_args()
    start=time.monotonic();root=a.plan.resolve().parent;plan=json.loads(a.plan.read_text())
    t.require(t.sha(__file__)==plan['source_sha256'] and t.sha(t.__file__)==plan['trainer_sha256'],'Source mismatch')
    t.require(not a.output.exists(),'Existing diagnostic output must not be overwritten');a.output.mkdir()
    train_plan=json.loads((root/plan['train_plan']).read_text());t.require(t.sha(root/plan['train_plan'])==plan['train_plan_sha256'],'Training plan drift')
    _,rows,counts=t.load_inputs(train_plan,root);torch.set_num_threads(1)
    cpu=dict(numeric=cpu_contract(),encoded=t.cpu_checks(train_plan,rows,counts))
    if a.cpu:
        t.save(a.output/'cpu-contract.json',dict(**cpu,source_sha256=t.sha(__file__),trainer_sha256=t.sha(t.__file__),elapsed_seconds=time.monotonic()-start));return
    t.require(plan['cap_seconds']==300 and plan['max_fp16_batches']==64 and plan['bf16_formal_batches']==3,'Probe scope drift')
    model_path,actual_files=t.verify_model(root,train_plan);gpu=t.fresh_gpu(train_plan)
    t.require(torch.cuda.is_bf16_supported(),'BF16 unsupported')
    def fresh():
        random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
        model,loading=t.AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=2,output_loading_info=True)
        t.require(not loading['mismatched_keys'] and not loading['error_msgs'],'Public base load failed')
        t.require(set(loading['missing_keys']) <= {'classifier.weight','classifier.bias','bert.pooler.dense.weight','bert.pooler.dense.bias'},'Unexpected base missing')
        model.cuda().train();t.require(all(p.dtype==torch.float32 for p in model.parameters()),'Non-FP32 parameters')
        opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        sched=t.get_linear_schedule_with_warmup(opt,103,1032)
        # Exactly the formal shuffled Dataset length/order; attach indices only for traceability.
        loader=DataLoader(list(enumerate(rows)),batch_size=16,shuffle=True,num_workers=0,collate_fn=lambda batch:([i for i,r in batch],t.collate([r for i,r in batch],'baseline')))
        return model,opt,sched,iter(loader)
    weights=torch.tensor(len(rows)/(2*counts),dtype=torch.float32,device='cuda')
    def identity(indices):
        keys=[rows[i]['key'] for i in indices]
        return dict(indices=indices,keys_sha256=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),positive_labels=sum(rows[i]['label'] for i in indices))
    model,opt,sched,loader=fresh();scaler=torch.amp.GradScaler('cuda');fp16=[];overflow=None
    torch.cuda.reset_peak_memory_stats()
    for bi in range(64):
        indices,batch=next(loader);batch={k:v.cuda() for k,v in batch.items()};labels=batch.pop('labels');opt.zero_grad(set_to_none=True)
        before=model.classifier.weight.detach().clone();step_before=float(opt.state.get(model.classifier.weight,{}).get('step',0));old_scale=float(scaler.get_scale())
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(**batch).logits;loss=t.weighted_ce(logits,labels,weights)
        t.require(bool(torch.isfinite(loss)) and bool(torch.isfinite(logits).all()),'Nonfinite forward; not mere scaled-gradient overflow')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        bad=sum(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        found_inf=sum(float(v) for v in scaler._per_optimizer_states[id(opt)]['found_inf_per_device'].values())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update();sched.step()
        event=dict(batch_index=bi,**identity(indices),loss=float(loss.detach()),unscaled_norm=float(norm) if bool(torch.isfinite(norm)) else str(float(norm)),nonfinite_gradient_tensors=bad,found_inf=found_inf,scale_before=old_scale,scale_after=float(scaler.get_scale()),classifier_unchanged=torch.equal(before,model.classifier.weight),optimizer_step_before=step_before,optimizer_step_after=float(opt.state.get(model.classifier.weight,{}).get('step',0)))
        fp16.append(event)
        if bad:
            t.require(found_inf>0 and event['scale_after']==old_scale/2 and event['classifier_unchanged'] and event['optimizer_step_after']==step_before,'Not recoverable AMP overflow')
            overflow=event;break
    t.save(a.output/'fp16-diagnosis.json',dict(events=fp16,overflow_reproduced=overflow is not None,source_sha256=t.sha(__file__),original_failure_source_sha256=plan['original_failed_source_sha256']))
    t.require(overflow is not None,'Failed to reproduce FP16 overflow within bounded diagnostic; cause remains unconfirmed')
    del model,opt,sched,loader,scaler,batch,logits,loss,before;gc.collect();torch.cuda.empty_cache()
    model,opt,sched,loader=fresh();bf16=[];torch.cuda.reset_peak_memory_stats()
    for bi in range(4):
        if bi<3: indices,batch=next(loader);arm='baseline'
        else: indices=train_plan['probe_row_indices'];batch=t.collate([rows[i] for i in indices],'bridge');arm='bridge_longest'
        before=model.classifier.weight.detach().clone();torch.cuda.synchronize();step_start=time.monotonic()
        result=t.one_update(model,batch,opt,sched,None,weights,diagnostics=True)
        torch.cuda.synchronize();seconds=time.monotonic()-step_start
        changed=not torch.equal(before,model.classifier.weight)
        t.require(changed==(bi>0),'Wrong warmup/actual parameter mutation')
        state=opt.state[model.classifier.weight]
        t.require(all(p.dtype==torch.float32 for p in model.parameters()),'Parameter dtype drift')
        t.require(state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32,'Adam state dtype drift')
        event=dict(batch_index=bi,arm=arm,**identity(indices),**result,seconds=seconds,classifier_changed=changed,adam_step=float(state['step']),adam_moments_fp32=True,parameters_fp32=True)
        bf16.append(event)
    result=dict(status='complete_disposable_fp16_cause_and_bf16_probe',source_sha256=t.sha(__file__),trainer_sha256=t.sha(t.__file__),plan_sha256=t.sha(a.plan),cpu_contract=cpu,model_files=actual_files,gpu_before=gpu,fp16_overflow=overflow,bf16_steps=bf16,elapsed_seconds=time.monotonic()-start,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),no_checkpoint_or_optimizer_saved=True,no_gold_access=True,formal_state_reused=False)
    t.save(a.output/'diagnostic-complete.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
