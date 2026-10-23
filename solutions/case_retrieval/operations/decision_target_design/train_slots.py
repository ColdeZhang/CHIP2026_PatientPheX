#!/usr/bin/env python3
"""Fresh fixed 124-row/2-epoch slot classification; final checkpoint only, no resume."""
from pathlib import Path
import argparse,json,time,os,random,hashlib
import slot_model as common

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--source-manifest',required=True)
    args=parser.parse_args();b=args.directory;plan=json.loads((b/'remote-plan.json').read_text())
    assert common.sha(__file__)==plan['trainer_sha256'] and common.sha(common.__file__)==plan['common_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and not (b/'run_manifest.json').exists()
    import torch,numpy as np
    from transformers import get_linear_schedule_with_warmup
    import qwen35_relation_cost as protocol
    import old_train_meta32 as previous
    assert common.sha(previous.__file__)=='08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
    started=time.monotonic();data=common.rows(b/'slot-train-encoded.json.gz',plan['encoded_sha256'],True)
    model,named,manifest=common.load_fresh(args.model,args.source_manifest,20261023)
    manifest.update(plan_sha256=common.sha(b/'remote-plan.json'),trainer_sha256=common.sha(__file__),
        common_sha256=common.sha(common.__file__),encoded_sha256=common.sha(b/'slot-train-encoded.json.gz'),
        configuration=dict(epochs=2,updates=32,warmup=2,seed=20261023,lr=1e-4,weight_decay=.01,
            max_grad_norm=1.,batch_size=1,accumulation=8,rank=16,alpha=32,dropout=.05,max_sequence_length=7168,
            padding='none needed with batch1',truncation=False,loss='per-row mean two-class CE, per-group actual count mean',
            position='direct slot end; no causal shift',classes=[15,16],head='existing frozen',threshold=.5),
        training_rows=124,training_docs=32,targets=2506,positive_targets=594,full_tokens=424087,
        final_checkpoint_only=True,no_probe_or_old_adapter_reuse=True,no_resume=True,no_quality=True)
    common.save(b/'run_manifest.json',manifest)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();assert model.is_gradient_checkpointing
    params=list(named.values());optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,2,32)
    model.train();step=0;checked=False;torch.cuda.reset_peak_memory_stats()
    with (b/'updates.jsonl').open('x') as trace:
        for epoch in range(2):
            groups=previous.batch_plan(124,1,8,20261023,epoch)
            assert len(groups)==16 and sum(len(g) for g in groups)==124
            epoch_started=time.monotonic();sum_loss=0.
            for group in groups:
                optimizer.zero_grad(set_to_none=True);lr=optimizer.param_groups[0]['lr'];probes={};group_loss=0.
                if not checked and lr>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:
                            probes[protocol.family(name)]=(name,p,p.detach().clone())
                for indices in group:
                    assert len(indices)==1
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=common.row_loss(model,data[indices[0]],torch)
                    (loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                assert torch.isfinite(norm) and norm>0
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                optimizer.step();scheduler.step();step+=1
                if probes:
                    assert set(probes)=={'DeltaNet','full_attention','MLP'}
                    assert all(not torch.equal(before,p.detach()) for _,p,before in probes.values())
                    common.save(b/'first-nonzero-update.json',dict(step=step,learning_rate=lr,
                        gradient_norm_before_clip=float(norm),changed_B_probes={k:v[0] for k,v in probes.items()}))
                    checked=True
                record=dict(epoch=epoch+1,step=step,group_rows=len(group),loss=group_loss,learning_rate=lr,
                    elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                trace.write(json.dumps(record)+'\n');trace.flush();sum_loss+=group_loss
                print(json.dumps(record),flush=True)
            common.save(b/f'epoch-{epoch+1:03d}-summary.json',dict(completed_epochs=epoch+1,global_step=step,
                mean_group_loss=sum_loss/16,epoch_seconds=time.monotonic()-epoch_started,no_quality=True))
    assert checked and step==32 and scheduler.last_epoch==32 and optimizer.param_groups[0]['lr']==0
    staging=b/'.epoch-002.incomplete';destination=b/'epoch-002'
    assert not staging.exists() and not destination.exists();staging.mkdir()
    model.save_pretrained(staging/'adapter',safe_serialization=True)
    state=dict(completed_epochs=2,global_step=32,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(state,staging/'training_state.pt')
    files={str(p.relative_to(staging)):dict(sha256=common.sha(p),bytes=p.stat().st_size)
           for p in sorted(staging.rglob('*')) if p.is_file()}
    assert {'adapter/adapter_config.json','adapter/adapter_model.safetensors','training_state.pt'} <= set(files)
    common.save(staging/'checkpoint.json',dict(completed_epochs=2,global_step=32,files=files,
        manifest_sha256=common.sha(b/'run_manifest.json'),plan_sha256=common.sha(b/'remote-plan.json'),
        final_fixed_epoch_only=True,optimizer_schedule_exhausted=True,no_quality_selection=True))
    staging.replace(destination)
    elapsed=time.monotonic()-started;assert elapsed<1800
    common.save(b/'training_complete.json',dict(status='fresh_fixed_slot_training_complete',epochs=2,global_step=32,
        successful_optimizer_calls=32,no_overflow_skips=True,manifest_sha256=common.sha(b/'run_manifest.json'),
        checkpoint_sha256=common.sha(destination/'checkpoint.json'),updates_sha256=common.sha(b/'updates.jsonl'),
        first_update_sha256=common.sha(b/'first-nonzero-update.json'),elapsed_seconds=elapsed,
        completed_epoch=time.time(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),no_F1_labels=True,no_quality=True))
    print('TRAINING_COMPLETE_FRESH_EPOCH002_32',flush=True)

if __name__=='__main__':main()
