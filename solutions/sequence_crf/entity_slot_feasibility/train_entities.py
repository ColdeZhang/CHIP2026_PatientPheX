#!/usr/bin/env python3
"""Fresh fixed two-epoch entity slot training. No heldout data or evaluation."""
from pathlib import Path
import argparse, gzip, json, os, random, time
import slot_model as common
import microbatch_slots as batching


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--model',required=True)
    a=ap.parse_args();b=a.directory;started=time.monotonic()
    plan=json.loads((b/'plan.json').read_text());assert common.sha(__file__)==plan['trainer_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
    assert all(common.sha(b/name)==digest for name,digest in plan['local_files'].items())
    assert not (b/'run_manifest.json').exists()
    import torch,numpy as np
    from transformers import AutoTokenizer,get_linear_schedule_with_warmup
    import qwen35_relation_cost as protocol
    data=json.loads(gzip.decompress((b/'train.json.gz').read_bytes()))
    assert len(data)==637 and len({r['pmc_id'] for r in data})==32
    assert sum(len(r['keys']) for r in data)==2430 and sum(sum(r['targets']) for r in data)==1645
    assert sum(len(r['input_ids']) for r in data)==460069
    assert len({r['row_id'] for r in data})==637
    for r in data:
        assert len(r['targets'])==len(r['keys'])==len(r['slot_positions']) and set(r['targets'])<={0,1}
        assert r['slot_positions']==sorted(set(r['slot_positions'])) and r['slot_positions'][-1]==len(r['input_ids'])-1
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    assert tokenizer.pad_token_id is not None
    model,named,manifest=common.load_fresh(a.model,b/'model-source-manifest.json',20261023)
    manifest.update(plan_sha256=common.sha(b/'plan.json'),trainer_sha256=common.sha(__file__),
        batch_source_sha256=common.sha(batching.__file__),train_sha256=common.sha(b/'train.json.gz'),
        data=dict(rows=637,documents=32,candidates=2430,positive=1645,tokens=460069,gate='train32 resubstitution HGB77 >=.1'),
        configuration=plan['recipe'],no_heldout_labels=True,no_quality=True,fresh_no_probe_state=True)
    common.save(b/'run_manifest.json',manifest)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train();params=list(named.values())
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,8,160)
    head=model.get_base_model().lm_head;head_initial=head.weight[[15,16]].detach().clone()
    step=0;checked=False;torch.cuda.reset_peak_memory_stats()

    def checkpoint(name,completed_epochs,next_epoch,next_group,final):
        stage=b/('.'+name+'.incomplete');target=b/name
        assert not stage.exists() and not target.exists();stage.mkdir()
        model.save_pretrained(stage/'adapter',safe_serialization=True)
        torch.save(dict(completed_epochs=completed_epochs,global_step=step,next_epoch_index=next_epoch,
            next_group_index=next_group,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()),stage/'training_state.pt')
        files={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=common.sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()}
        common.save(stage/'checkpoint.json',dict(completed_epochs=completed_epochs,global_step=step,
            next_epoch_index=next_epoch,next_group_index=next_group,files=files,
            training_manifest_sha256=common.sha(b/'run_manifest.json'),plan_sha256=common.sha(b/'plan.json'),
            final_fixed_epoch=final,not_for_quality=not final,no_quality_selection=True))
        stage.replace(target);return target

    with (b/'updates.jsonl').open('x') as trace:
        for epoch in range(2):
            groups=batching.optimizer_groups(637,epoch=epoch);assert len(groups)==80 and len(groups[-1])==5
            tick=time.monotonic();total_loss=0.
            for gi,group in enumerate(groups):
                # 180s fixed checkpoint margin +40s next-group allowance; no unseen
                # training quality or heldout data decides this budget stop.
                if time.monotonic()-started+180+40>=2200:
                    target=checkpoint(f'partial-step-{step:03d}',epoch,epoch,gi,False)
                    common.save(b/'terminal.json',dict(status='partial_budget_stop_not_final',step=step,
                        checkpoint=str(target),checkpoint_sha256=common.sha(target/'checkpoint.json'),
                        seconds=time.monotonic()-started,no_inference=True,no_automatic_resume=True))
                    print('PARTIAL_SAVED_NOT_FOR_INFERENCE',flush=True);return
                micro=batching.microbatches(data,group,8192);optimizer.zero_grad(set_to_none=True)
                lr=optimizer.param_groups[0]['lr'];probes={};value=0.
                if not checked and lr>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:
                            probes[protocol.family(name)]=(name,p,p.detach().clone())
                for ids in micro:
                    batch=batching.collate([data[i] for i in ids],torch,'cuda',tokenizer.pad_token_id)
                    assert batch['input_ids'].numel()<=8192
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=batching.decision_logits(model,batch,torch)
                        loss=batching.group_loss(logits,batch,len(group),torch)
                    assert torch.isfinite(loss);loss.backward();value+=float(loss.detach());del logits,loss,batch
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                assert torch.isfinite(norm) and norm>0
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                optimizer.step();scheduler.step();step+=1
                if probes:
                    assert set(probes)=={'DeltaNet','full_attention','MLP'}
                    assert all(not torch.equal(old,p.detach()) for _,p,old in probes.values())
                    assert head.weight.grad is None and torch.equal(head_initial,head.weight[[15,16]].detach())
                    common.save(b/'first-nonzero-update.json',dict(step=step,learning_rate=lr,
                        gradient_norm=float(norm),changed_families={k:v[0] for k,v in probes.items()},head_unchanged=True))
                    checked=True
                record=dict(epoch=epoch+1,step=step,group_index=gi,rows=len(group),
                    microbatch_sizes=[len(x) for x in micro],loss=value,learning_rate=lr,
                    seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                trace.write(json.dumps(record)+'\n');trace.flush();total_loss+=value
                if step%10==0:print(json.dumps(record),flush=True)
            target=checkpoint(f'epoch-{epoch+1:03d}',epoch+1,epoch+1,0,epoch==1)
            common.save(b/f'epoch-{epoch+1:03d}-summary.json',dict(epoch=epoch+1,step=step,
                seconds=time.monotonic()-tick,mean_group_loss=total_loss/80,
                checkpoint_sha256=common.sha(target/'checkpoint.json'),not_for_quality=epoch==0))
    assert step==160 and checked and scheduler.last_epoch==160 and optimizer.param_groups[0]['lr']==0
    assert torch.equal(head_initial,head.weight[[15,16]].detach())
    elapsed=time.monotonic()-started;assert elapsed<2200
    common.save(b/'training_complete.json',dict(status='fresh_entity_637_epoch002_160_complete',
        epochs=2,global_step=160,successful_optimizer_calls=160,overflow_skips=0,seconds=elapsed,
        manifest_sha256=common.sha(b/'run_manifest.json'),plan_sha256=common.sha(b/'plan.json'),
        checkpoint_sha256=common.sha(b/'epoch-002/checkpoint.json'),updates_sha256=common.sha(b/'updates.jsonl'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        no_F1_labels=True,no_quality=True,final_epoch_only=True))
    common.save(b/'terminal.json',dict(status='complete_epoch002_160',complete_sha256=common.sha(b/'training_complete.json')))
    print('ENTITY_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
