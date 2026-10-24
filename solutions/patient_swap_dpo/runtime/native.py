"""One fixed DPO reference, pressure probe, or one-epoch formal stage."""
import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from dpo_ops import sequence_logp,serial_dpo_backward
from common import checked,checkpoint,gpu_gate,load_model,memory,ref,save,source_gate,stamp,update,verify_assets


def encoded(plan):
    rows=[json.loads(x) for x in gzip.decompress(checked(plan['encoded']).read_bytes()).splitlines()]
    assert len(rows)==74
    return rows


def inputs(torch,row):
    return [torch.tensor([row[k+'_input_ids']],dtype=torch.long,device='cuda') for k in ['chosen','rejected']]


def reference(plan,out,torch,model,rows):
    results=[];started=time.monotonic()
    for i,row in enumerate(rows):
        update(out/'progress.json',dict(at_utc=stamp(),completed=i,total=74,row_id=row['row_id']))
        cc,rr=inputs(torch,row);tick=time.monotonic()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pc=sequence_logp(model,cc,row['prompt_length']);pr=sequence_logp(model,rr,row['prompt_length'])
        torch.cuda.synchronize();finite=bool(torch.isfinite(pc)&torch.isfinite(pr))
        item=dict(index=i,row_id=row['row_id'],chosen=float(pc) if finite else None,rejected=float(pr) if finite else None,
                  finite=finite,seconds=time.monotonic()-tick,
                  chosen_tokens=cc.shape[1],rejected_tokens=rr.shape[1])
        save(out/f'pair-{i:03d}.json',item);assert finite;results.append(item)
    save(out/'reference-logps.json',results)
    return dict(status='COMPLETE_ALL148_FROZEN_REFERENCE_LOGPS',at_utc=stamp(),pairs=74,sequences=148,
                tokens=519954,seconds=time.monotonic()-started,reference=ref(out/'reference-logps.json'),
                parent_adapter_files=plan['parent_adapter_files'],encoded=plan['encoded'],memory=memory(torch))


def inventory(torch,model,named,opt):
    gradients=[x.grad for x in named.values() if x.grad is not None]
    states=[v for item in opt.state.values() for v in item.values() if isinstance(v,torch.Tensor)]
    return dict(gradient_tensors=len(gradients),all_grad_finite=all(bool(torch.isfinite(x).all()) for x in gradients),
                optimizer_states=len(opt.state),optimizer_tensors=len(states),all_optimizer_finite=all(bool(torch.isfinite(x).all()) for x in states),
                frozen_base_no_gradient=all(x.grad is None for x in model.parameters() if not x.requires_grad),memory=memory(torch))


def apply_pair(plan,out,torch,model,row,reference,group_size,identifier):
    cc,rr=inputs(torch,row);tick=time.monotonic();observations=[]
    def observe(x):
        observations.append(x)
        with (out/'replay-observations.jsonl').open('a') as f:
            f.write(json.dumps(dict(row_id=row['row_id'],event=identifier,**x),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    # A no-grad coefficient pass must not populate casts reused by the grad
    # replay/checkpoint pass. Keep BF16 math, disable only the weight-cast cache.
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        result=serial_dpo_backward(model,cc,rr,row['prompt_length'],reference['chosen'],reference['rejected'],group_size,observe)
    torch.cuda.synchronize()
    return dict(**result,seconds=time.monotonic()-tick,tokens=cc.shape[1]+rr.shape[1],row_id=row['row_id'],replay_observations=len(observations))


def probe(plan,out,torch,model,named,rows,refs,manifest):
    opt=torch.optim.AdamW(list(named.values()),lr=1e-5,weight_decay=.01)
    order=plan['probe_pair_indices'];durations=[]
    before={family:next(p for n,p in named.items() if family in n).detach().clone() for family in ['.q_proj.lora_B.','.gate_proj.lora_B.']}
    for step,index in enumerate(order):
        row=rows[index];update(out/'progress.json',dict(stage='DPO_serial_pressure',at_utc=stamp(),step=step,index=index,row_id=row['row_id']))
        result=apply_pair(plan,out,torch,model,row,refs[index],8,'probe-'+str(step))
        observed=inventory(torch,model,named,opt);save(out/f'pair-{step:03d}.json',dict(index=index,**result,inventory=observed))
        assert observed['all_grad_finite'] and observed['frozen_base_no_gradient'];durations.append(dict(index=index,**result))
        if step==7:
            norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
            assert bool(torch.isfinite(norm)) and float(norm)>0
            opt.step();changed={family:not torch.equal(value,next(p for n,p in named.items() if family in n).detach()) for family,value in before.items()}
            observed=inventory(torch,model,named,opt);save(out/'Adam-update.json',dict(changed=changed,grad_norm=float(norm),**observed))
            assert all(changed.values()) and observed['optimizer_states']==504 and observed['all_optimizer_finite']
            opt.zero_grad(set_to_none=True)
    observed=inventory(torch,model,named,opt)
    assert observed['gradient_tensors']==504 and observed['optimizer_states']==504 and observed['all_optimizer_finite']
    worst=max(x['seconds']/x['tokens'] for x in durations)
    estimated=1.25*worst*519954+manifest['load_seconds']+120
    return dict(status='PASS_ACTUAL_BF16_DPO_SERIAL_REPLAY_ADAM_AND_PARETO_PRESSURE',at_utc=stamp(),
                pressure_pairs=len(order),actual_backward_calls=2*len(order),coefficient_forward_calls=2*len(order),
                real_Adam_updates=1,pareto_indices=plan['pareto_indices'],durations=durations,
                final_inventory=observed,formal_estimated_seconds=estimated,
                formal_budget_gate=estimated<=1800,probe_state_must_be_discarded=True,
                note='Actual score replay passed prewritten tolerance; not a BF16 bitwise-equivalence claim.')


def formal(plan,out,torch,np,model,named,rows,refs):
    from transformers import get_linear_schedule_with_warmup
    optimizer=torch.optim.AdamW(list(named.values()),lr=1e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,1,10)
    order=list(range(74));random.Random(plan['seed']).shuffle(order)
    groups=[order[i:i+8] for i in range(0,74,8)]
    assert len(groups)==10 and len(groups[-1])==2
    save(out/'fixed-order.json',dict(order=order,groups=groups,seed=plan['seed']))
    counts=dict(pairs=0,sequences=0,tokens=0,supervised=0);updates=[];started=time.monotonic();changed=False
    for step,group in enumerate(groups):
        optimizer.zero_grad(set_to_none=True);loss=0.;lr=optimizer.param_groups[0]['lr']
        watched=next(p for n,p in named.items() if '.q_proj.lora_B.' in n);before=watched.detach().clone() if lr>0 and not changed else None
        tick=time.monotonic()
        for index in group:
            row=rows[index];update(out/'progress.json',dict(stage='formal_DPO',at_utc=stamp(),global_step=step,index=index,row_id=row['row_id'],counts=counts,memory=memory(torch)))
            result=apply_pair(plan,out,torch,model,row,refs[index],len(group),'formal-'+str(step));loss+=result['loss']/len(group)
            counts['pairs']+=1;counts['sequences']+=2;counts['tokens']+=result['tokens'];counts['supervised']+=result['tokens']-2*row['prompt_length']
        observed=inventory(torch,model,named,optimizer);assert observed['all_grad_finite'] and observed['frozen_base_no_gradient']
        norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.);assert bool(torch.isfinite(norm)) and float(norm)>0
        optimizer.step();scheduler.step()
        if before is not None:changed=not torch.equal(before,watched.detach());assert changed
        item=dict(global_step=step+1,pairs=len(group),loss=loss,learning_rate=lr,grad_norm=float(norm),seconds=time.monotonic()-tick,memory=memory(torch))
        save(out/f'update-{step+1:02d}.json',item);updates.append(item);print(json.dumps(item),flush=True)
    assert changed and counts==dict(pairs=74,sequences=148,tokens=519954,supervised=9344)
    ack=checkpoint(torch,np,model,optimizer,scheduler,out,1,10,counts,plan)
    return dict(status='COMPLETE_FIXED_ONE_EPOCH_DPO_WITH_DURABLE_ACK',at_utc=stamp(),completed_epochs=1,global_step=10,
                counts=counts,updates=updates,acknowledgement=ack,training_seconds=time.monotonic()-started,memory=memory(torch),
                fresh_optimizer_not_resume=True,parent_adapter_files=plan['parent_adapter_files'])


def main(phase,path):
    plan=json.loads(Path(path).read_text());source_gate(plan);out=Path(plan['run_root'])/phase;out.mkdir(exist_ok=False)
    save(out/'start.json',dict(at_utc=stamp(),pid=os.getpid(),phase=phase,plan=ref(path)))
    assert plan['execution_authorized'] and phase in ['reference','probe','formal']
    cpu=json.loads(checked(plan['CPU_ready']).read_text());assert cpu['status']=='PASS_ACTUAL_TARGET_HF_NATIVE_ALL148_AND_INHERITED_ASSET_STAT'
    assets=verify_assets(plan,full=False,previous=json.loads(checked(cpu['assets']).read_text()));save(out/'verified-assets.json',assets)
    for r in plan['parent_adapter_files'].values():checked(r)
    gpu_gate(plan,out)
    rows=encoded(plan)
    refs=None
    if phase!='reference':
        inherited=plan.get('inherited_reference')
        done=json.loads(checked(inherited['complete']).read_text()) if inherited else json.loads((Path(plan['run_root'])/'reference/complete.json').read_text())
        assert done['status']=='COMPLETE_ALL148_FROZEN_REFERENCE_LOGPS'
        assert done['plan']==(inherited['plan'] if inherited else ref(path)) and done['encoded']==plan['encoded']
        if inherited:
            assert done['reference']==inherited['logps']
            original=json.loads(checked(inherited['plan']).read_text())
            assert all(original[k]==plan[k] for k in ['recipe','parent_adapter_files','encoded','pairs','CPU_ready','model_files','native_forward_sha256','seed'])
        assert done['parent_adapter_files']==plan['parent_adapter_files']
        refs=json.loads(checked(done['reference']).read_text());assert len(refs)==74
        assert all(a['row_id']==b['row_id'] and a['finite'] for a,b in zip(refs,rows))
    if phase=='formal':
        prior=json.loads((Path(plan['run_root'])/'probe/complete.json').read_text())
        assert prior['plan']==ref(path) and prior['formal_budget_gate']
    torch,np,model,named,manifest=load_model(plan,out,adapter=plan['parent_adapter_dir'],trainable=phase!='reference')
    if phase=='reference':result=reference(plan,out,torch,model,rows)
    elif phase=='probe':result=probe(plan,out,torch,model,named,rows,refs,manifest)
    else:result=formal(plan,out,torch,np,model,named,rows,refs)
    result.update(plan=ref(path),model_manifest=ref(out/'model-manifest.json'),source=ref(__file__),no_quality_calls=True)
    save(out/'complete.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['reference','probe','formal']);p.add_argument('--plan',required=True);a=p.parse_args();main(a.phase,a.plan)
