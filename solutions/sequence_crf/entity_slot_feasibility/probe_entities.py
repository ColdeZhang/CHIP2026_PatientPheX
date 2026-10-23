#!/usr/bin/env python3
"""Fixed train shortest/longest gradient steps and unlabeled F1 forward only."""
from pathlib import Path
import argparse
import json
import os
import time
import slot_model as common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    started = time.monotonic()
    plan = json.loads((a.directory/'plan.json').read_text())
    assert common.sha(__file__) == plan['source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpu_uuid']
    for name, digest in plan['local_files'].items():
        assert common.sha(a.directory/name) == digest
    rows = json.loads((a.directory/'probe-records.json').read_text())
    short, longest, held = [rows[k] for k in ['train_shortest', 'train_longest', 'F1_longest']]
    assert len(short['input_ids']) == 259 and len(longest['input_ids']) == 2418 and len(held['input_ids']) == 2810
    import torch
    import qwen35_relation_cost as protocol
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    for key, row in rows.items():
        prefix = tokenizer.apply_chat_template(row['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        text = prefix
        ends = []
        for i in range(len(row['keys'])):
            text += f'Question C{i}:\n'
            ends.append(len(text))
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == row['input_ids']
        assert tokenizer.decode(encoded['input_ids'], skip_special_tokens=False) == text
        positions = []
        for end in ends:
            hits = [i for i,(_,right) in enumerate(encoded['offset_mapping']) if right == end]
            assert len(hits) == 1
            positions.append(hits[0])
        assert positions == row['slot_positions'] and positions[-1] == len(row['input_ids'])-1
        if key.startswith('train'):
            assert len(row['targets']) == len(positions) and set(row['targets']) <= {0, 1}
        else: assert 'targets' not in row
    # Synthetic CPU derivative check is separate from actual data, never relabels it.
    artificial = torch.tensor([[2., -2.], [-2., 2.]], requires_grad=True)
    synthetic = torch.nn.functional.cross_entropy(artificial, torch.tensor([1, 0]))
    synthetic.backward()
    assert artificial.grad[0,1] < 0 < artificial.grad[0,0]
    assert artificial.grad[1,0] < 0 < artificial.grad[1,1]
    model, named, manifest = common.load_fresh(a.model, a.directory/'model-source-manifest.json', 20261023)
    manifest.update(plan_sha256=common.sha(a.directory/'plan.json'),
                    source_sha256=common.sha(__file__), actual_record_tokenization_exact=True,
                    synthetic_positive_negative_gradient_direction=True,
                    candidate_task='Task1 span plus ID validity; no patient target')
    common.save(a.directory/'run_manifest.json', manifest)
    model.eval()
    ids, positions = common.tensor_input(short, torch)
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                     logits_to_keep=0, return_dict=True)
        classes = torch.tensor([15,16], device='cuda')
        reference = full.logits[0].index_select(0,positions).index_select(-1,classes).float().clone()
        shifted = full.logits[0].index_select(0,positions-1).index_select(-1,classes).float().clone()
        del full
        selected = common.decision_logits(model,ids,positions,torch)
        assert torch.isfinite(reference).all() and torch.isfinite(selected).all()
        assert torch.allclose(reference,selected,rtol=.01,atol=.01)
        assert not torch.equal(reference,shifted)
        equivalence = dict(row_id=short['row_id'], tokens=len(short['input_ids']),
            positions=short['slot_positions'], max_abs_diff=float((reference-selected).abs().max()),
            exact=torch.equal(reference,selected), wrong_shift_max_abs_diff=float((reference-shifted).abs().max()))
    common.save(a.directory/'equivalence.json', equivalence)
    del ids,positions,reference,shifted,selected
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(list(named.values()),lr=1e-4,weight_decay=.01)
    head = model.get_base_model().lm_head
    head_before = head.weight[[15,16]].detach().clone()
    results = []
    for label, row in [('shortest',short),('longest',longest)]:
        model.train()
        before = {}
        for name,p in named.items():
            if '.lora_B.' in name and protocol.family(name) not in before:
                before[protocol.family(name)] = (p,p.detach().clone())
        assert set(before) == {'DeltaNet','full_attention','MLP'}
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss, logits, targets = common.row_loss(model,row,torch)
        logits.retain_grad(); loss.backward()
        target_grad = logits.grad.gather(1, targets[:,None]).squeeze(1)
        assert torch.isfinite(logits.grad).all() and (target_grad<=0).all() and (target_grad<0).any()
        norms = {key:0. for key in before}
        for name,p in named.items():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            norms[protocol.family(name)] += float(p.grad.detach().float().square().sum())
        assert all(v>0 for v in norms.values())
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
        assert torch.isfinite(norm) and norm>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
        assert all(not torch.equal(old,p.detach()) for p,old in before.values())
        assert head.weight.grad is None and torch.equal(head_before,head.weight[[15,16]].detach())
        assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        torch.cuda.synchronize()
        results.append(dict(kind=label,row_id=row['row_id'],tokens=len(row['input_ids']),
            slots=len(row['keys']),positive=sum(row['targets']),loss=float(loss.detach()),
            gradient_norm=float(norm),family_squared_norms=norms,seconds=time.monotonic()-tick,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            LoRA_updated=True,LM_head_frozen_and_unchanged=True))
        common.save(a.directory/f'{label}-step.json',results[-1])
        del logits,targets,loss,before
    optimizer.zero_grad(set_to_none=True)
    model.eval(); torch.cuda.synchronize(); tick=time.monotonic()
    with torch.inference_mode():
        ids,positions=common.tensor_input(held,torch)
        logits=common.decision_logits(model,ids,positions,torch)
        assert torch.isfinite(logits).all()
    torch.cuda.synchronize()
    common.save(a.directory/'heldout-shape-only.json',dict(row_id=held['row_id'],tokens=len(held['input_ids']),
        slots=len(held['keys']),seconds=time.monotonic()-tick,finite=True,probabilities_not_saved=True,
        no_targets=True,no_quality=True,updated_probe_model_not_quality_reference=True))
    elapsed=time.monotonic()-started
    assert elapsed<120
    common.save(a.directory/'complete.json',dict(status='probe_PASS_state_discarded',seconds=elapsed,
        source_sha256=common.sha(__file__),plan_sha256=common.sha(a.directory/'plan.json'),
        outputs={p.name:common.sha(p) for p in a.directory.glob('*.json') if p.name not in ['plan.json','probe-records.json','model-source-manifest.json']},
        no_saved_adapter=True,no_F1_gold=True,no_quality=True,no_formal_training=True))
    print(json.dumps(dict(completed=True,seconds=elapsed,steps=results)),flush=True)


if __name__=='__main__': main()
