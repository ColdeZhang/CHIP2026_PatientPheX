#!/usr/bin/env python3
"""One fresh 180s fixed-slot interface/longest-gradient probe; discard all state."""
from pathlib import Path
import argparse,json,time,os
import slot_model as common

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--source-manifest',required=True)
    args=parser.parse_args();b=args.directory;plan=json.loads((b/'remote-plan.json').read_text())
    assert common.sha(__file__)==plan['probe_source_sha256']
    assert common.sha(common.__file__)==plan['model_source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and not (b/'run_manifest.json').exists()
    started=time.monotonic()
    import torch
    import qwen35_relation_cost as protocol
    data=common.rows(b/'slot-train-encoded.json.gz',plan['encoded_sha256'],True)
    short=min((r for r in data if len(r['keys'])>=2),key=lambda r:(len(r['input_ids']),r['row_id']))
    longest=min(data,key=lambda r:(-len(r['input_ids']),r['row_id']))
    assert short['row_id']=='5866036:relations:16' and len(short['input_ids'])==880
    assert longest['row_id']=='11235872:relations:8' and len(longest['input_ids'])==6777
    model,named,manifest=common.load_fresh(args.model,args.source_manifest,20261023)
    manifest.update(plan_sha256=common.sha(b/'remote-plan.json'),probe_source_sha256=common.sha(__file__),
        common_source_sha256=common.sha(common.__file__),encoded_sha256=common.sha(b/'slot-train-encoded.json.gz'))
    common.save(b/'run_manifest.json',manifest)
    # Same actual short row, no truncation. Eval disables LoRA dropout for equivalence.
    model.eval();ids,positions=common.tensor_input(short,torch)
    with torch.inference_mode():
        full=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                   logits_to_keep=0,return_dict=True)
        reference=full.logits[0].index_select(0,positions).index_select(-1,
            torch.tensor([15,16],device='cuda')).float().clone()
        shifted=full.logits[0].index_select(0,positions-1).index_select(-1,
            torch.tensor([15,16],device='cuda')).float().clone()
        del full
        selected=common.decision_logits(model,ids,positions,torch)
        assert torch.isfinite(reference).all() and torch.isfinite(selected).all()
        # Different GEMM shapes may differ at BF16 rounding; report exactness separately.
        assert torch.allclose(reference,selected,rtol=.01,atol=.01)
        assert not torch.equal(reference,shifted)
        equality=dict(row_id=short['row_id'],input_tokens=len(short['input_ids']),
            slot_positions=short['slot_positions'],reference=reference.tolist(),selected=selected.tolist(),
            max_abs_difference=float((reference-selected).abs().max()),exact_equal=torch.equal(reference,selected),
            rtol=.01,atol=.01,wrong_causal_shift_is_distinct=True,
            wrong_shift_max_abs_difference=float((reference-shifted).abs().max()))
    common.save(b/'native-selection-equivalence.json',equality)
    del ids,positions,reference,shifted,selected
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();assert model.is_gradient_checkpointing
    model.train();optimizer=torch.optim.AdamW(list(named.values()),lr=1e-4,weight_decay=.01)
    before={}
    for name,p in named.items():
        if '.lora_B.' in name and protocol.family(name) not in before:
            before[protocol.family(name)]=(name,p,p.detach().clone())
    assert set(before)=={'DeltaNet','full_attention','MLP'}
    head=model.get_base_model().lm_head;head_before=head.weight[[15,16]].detach().clone()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,logits,targets=common.row_loss(model,longest,torch)
    logits.retain_grad();loss.backward()
    assert torch.isfinite(logits.grad).all()
    selected_gradient=logits.grad.gather(1,targets[:,None]).squeeze(1)
    assert (selected_gradient<=0).all() and (selected_gradient<0).any()
    family_norms={family:0. for family in before}
    for name,p in named.items():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        family_norms[protocol.family(name)]+=float(p.grad.detach().float().square().sum())
    assert all(v>0 for v in family_norms.values())
    norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
    assert torch.isfinite(norm) and norm>0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert all(not torch.equal(old,p.detach()) for _,p,old in before.values())
    assert head.weight.grad is None and torch.equal(head_before,head.weight[[15,16]].detach())
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    torch.cuda.synchronize();step_seconds=time.monotonic()-tick
    result=dict(row_id=longest['row_id'],input_tokens=6777,slots=112,positive=12,loss=float(loss.detach()),
        gradient_norm_before_clip=float(norm),family_gradient_squared_norms=family_norms,
        changed_B_probes={k:v[0] for k,v in before.items()},learning_rate=1e-4,
        first_probe_step_nonzero_lr_not_formal_warmup=True,head_frozen_and_rows_unchanged=True,
        step_seconds=step_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0],
        projection_token_linear_twoepochs_seconds=2*424087/6777*step_seconds,
        projection_includes_per_row_optimizer_overhead_not_actual_training_time=True,
        optimizer_state_dtypes=sorted({str(v.dtype) for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)}))
    common.save(b/'longest-gradient-step.json',result)
    assert time.monotonic()-started<180
    common.save(b/'complete.json',dict(status='fresh_slot_probe_PASS_state_discarded_no_checkpoint',
        manifest_sha256=common.sha(b/'run_manifest.json'),equivalence_sha256=common.sha(b/'native-selection-equivalence.json'),
        longest_step_sha256=common.sha(b/'longest-gradient-step.json'),elapsed_seconds=time.monotonic()-started,
        completed_epoch=time.time(),no_F1_labels=True,no_quality=True,no_saved_adapter=True))
    print(json.dumps(dict(complete=True,step=result)),flush=True)

if __name__=='__main__':main()
