#!/usr/bin/env python3
"""One fresh 150-second numerical/throughput probe; never saves model state."""
from pathlib import Path
import argparse, gzip, json, os, time
import slot_model as common
import microbatch_slots as batching


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--model',required=True)
    a=ap.parse_args();b=a.directory;started=time.monotonic()
    plan=json.loads((b/'plan.json').read_text())
    assert common.sha(__file__)==plan['source_sha256'] and os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
    assert all(common.sha(b/name)==sha for name,sha in plan['local_files'].items())
    import torch
    import qwen35_relation_cost as protocol
    from transformers import AutoTokenizer
    cpu=batching.cpu_contract(torch);common.save(b/'actual-cpu-contract.json',cpu)
    data=json.loads(gzip.decompress((b/'probe-groups.json.gz').read_bytes()))
    rows=data['rows']; groups=data['groups']
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    for row in rows:
        prefix=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        text=prefix;ends=[]
        for i in range(len(row['keys'])):text+=f'Question C{i}:\n';ends.append(len(text))
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        assert encoded['input_ids']==row['input_ids']
        positions=[]
        for end in ends:
            matches=[i for i,(_,right) in enumerate(encoded['offset_mapping']) if right==end]
            assert len(matches)==1;positions.append(matches[0])
        assert positions==row['slot_positions']
    model,named,manifest=common.load_fresh(a.model,b/'model-source-manifest.json',20261023)
    base=model.get_base_model();head=base.lm_head
    manifest.update(plan_sha256=common.sha(b/'plan.json'),source_sha256=common.sha(__file__),
        batch_source_sha256=common.sha(batching.__file__),engineering_variant=True,
        actual_tokenization_rows=len(rows),all_actual_tokenization_exact=True,
        no_new_head_parameters=True,backbone_class=type(base.model).__name__)
    common.save(b/'run_manifest.json',manifest)
    pad=tokenizer.pad_token_id
    assert pad is not None
    model.eval()
    # Two unequal-length rows from the input-selected shortest optimizer group.
    sample=[rows[i] for i in data['equivalence_indices']]
    batch=batching.collate(sample,torch,'cuda',pad)
    with torch.inference_mode():
        union=torch.unique(batch['positions'],sorted=True)
        native=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],use_cache=False,logits_to_keep=union,return_dict=True)
        reference=native.logits[batch['owners'],torch.searchsorted(union,batch['positions'])][:,[15,16]].float().clone()
        del native
        compact=batching.decision_logits(model,batch,torch)
        assert torch.allclose(reference,compact,atol=.125,rtol=.02)
        per_row=[]
        for row in sample:
            ids,positions=common.tensor_input(row,torch)
            per_row.append(common.decision_logits(model,ids,positions,torch))
        unpadded=torch.cat(per_row)
        assert torch.allclose(reference,unpadded,atol=.125,rtol=.02)
        poisoned={**batch,'input_ids':batch['input_ids'].clone()}
        poisoned['input_ids'][batch['attention_mask']==0]=(pad+1)%head.weight.shape[0]
        altered=batching.decision_logits(model,poisoned,torch)
        assert torch.allclose(compact,altered,atol=.125,rtol=.02)
        numerical=dict(native_vs_two_head_max_abs=float((reference-compact).abs().max()),
            batch_vs_unpadded_max_abs=float((reference-unpadded).abs().max()),
            padding_poison_max_abs=float((compact-altered).abs().max()),
            fixed_tolerance=dict(atol=.125,rtol=.02),rows=[r['row_id'] for r in sample])
    del reference,compact,altered,unpadded,per_row,poisoned,batch
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    # Actual-backbone derivatives, eval disables LoRA dropout for this equality check.
    small=sample[0];grads={}; losses={}
    for mode in ['native','two_head']:
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if mode=='native':loss,_,_=common.row_loss(model,small,torch)
            else:
                one=batching.collate([small],torch,'cuda',pad)
                loss=batching.group_loss(batching.decision_logits(model,one,torch),one,1,torch)
        loss.backward();losses[mode]=float(loss.detach())
        grads[mode]={name:p.grad.detach().clone() for name,p in named.items() if '.lora_B.' in name}
        assert all(torch.isfinite(v).all() for v in grads[mode].values())
        del loss
    gradient_comparison={}
    for family in ['DeltaNet','full_attention','MLP']:
        names=[name for name in grads['native'] if protocol.family(name)==family]
        ref=torch.cat([grads['native'][name].reshape(-1) for name in names]).float()
        actual=torch.cat([grads['two_head'][name].reshape(-1) for name in names]).float()
        relative=float((ref-actual).norm()/ref.norm().clamp_min(1e-12))
        cosine=float(torch.nn.functional.cosine_similarity(ref,actual,dim=0))
        assert ref.norm()>0 and actual.norm()>0 and relative<=.05 and cosine>=.995
        gradient_comparison[family]=dict(relative_L2=relative,cosine=cosine)
    numerical.update(losses=losses,actual_backbone_gradient=gradient_comparison,
        gradient_tolerance=dict(relative_L2=.05,cosine_min=.995),LoRA_dropout_disabled_for_equivalence_only=True)
    common.save(b/'equivalence.json',numerical)
    del grads,ref,actual,one
    model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW(list(named.values()),lr=1e-4,weight_decay=.01)
    head_before=head.weight[[15,16]].detach().clone()
    model.train();results=[]
    for spec in groups:
        # No reordering within the original shuffled optimizer group.
        group=spec['indices'];micro=batching.microbatches(rows,group,8192)
        optimizer.zero_grad(set_to_none=True);before={}
        for name,p in named.items():
            if '.lora_B.' in name and protocol.family(name) not in before:
                before[protocol.family(name)]=(p,p.detach().clone())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic();loss_value=0.
        for indices in micro:
            batch=batching.collate([rows[i] for i in indices],torch,'cuda',pad)
            assert batch['input_ids'].numel()<=8192
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=batching.decision_logits(model,batch,torch)
                loss=batching.group_loss(logits,batch,len(group),torch)
            assert torch.isfinite(loss);loss.backward();loss_value+=float(loss.detach())
            del logits,loss,batch
        norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
        assert torch.isfinite(norm) and norm>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
        assert all(not torch.equal(old,p.detach()) for p,old in before.values())
        assert head.weight.grad is None and torch.equal(head_before,head.weight[[15,16]].detach())
        torch.cuda.synchronize()
        result=dict(kind=spec['kind'],source_group=spec['source_group'],rows=len(group),
            row_ids=[rows[i]['row_id'] for i in group],microbatch_sizes=[len(x) for x in micro],
            tokens=sum(len(rows[i]['input_ids']) for i in group),
            padded_tokens=sum(len(x)*max(len(rows[i]['input_ids']) for i in x) for x in micro),
            seconds=time.monotonic()-tick,loss=loss_value,gradient_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            LoRA_all_three_family_updated=True,head_frozen_unchanged=True)
        results.append(result);common.save(b/(spec['kind']+'.json'),result)
        print(json.dumps(result),flush=True);del before
    elapsed=time.monotonic()-started;assert elapsed<150
    common.save(b/'complete.json',dict(status='microbatch_probe_PASS_state_discarded',seconds=elapsed,
        source_sha256=common.sha(__file__),plan_sha256=common.sha(b/'plan.json'),groups=results,
        no_saved_adapter=True,no_F1_input_or_gold=True,no_quality=True,no_formal_training=True,
        outputs={p.name:common.sha(p) for p in b.glob('*.json') if p.name not in ['plan.json','probe-groups.json.gz','model-source-manifest.json']}))
    print('MICROBATCH_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
