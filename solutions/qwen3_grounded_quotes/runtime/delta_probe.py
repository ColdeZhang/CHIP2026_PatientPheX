"""Only new literal-quote shapes; old native CE/state mathematics are reused."""
import gc
import time


def probe(plan, out, cpu, torch, model, named, model_manifest, common, shared):
    from loss_ops import indexed_causal_forward
    rows = shared.read_encoded(cpu, 'train')
    assert len(rows) == 87
    assert cpu['pareto_shapes'] == plan['probe_pareto_shapes']
    selected = list(cpu['pareto_indices'])
    assert selected and len(set(selected)) == len(selected)
    model.train()
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
    index = min(range(len(rows)), key=lambda i: (len(rows[i]['input_ids']), i))
    first_b = next(p for name, p in named.items() if '.lora_B.' in name)
    before = first_b.detach().clone()
    durations = []
    # One real update creates actual resident Adam; its result is discarded.
    for step, i in enumerate([index] + selected):
        row = rows[i]
        start = time.monotonic()
        common.update(out / 'progress.json', dict(stage='delta_forward', step=step,
            index=i, row_id=row['row_id'], tokens=len(row['input_ids']), memory=common.memory(torch)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = indexed_causal_forward(model, **common.batch(torch, row))
        finite = bool(torch.isfinite(value.loss))
        common.save(out / f'{step:02d}-forward.json', dict(index=i, finite=finite,
            loss=float(value.loss.detach()) if finite else None,
            shape=[len(row['input_ids']), value.supervised_tokens], memory=common.memory(torch)))
        assert finite
        (value.loss / 8).backward()
        torch.cuda.synchronize()
        durations.append(dict(index=i, tokens=len(row['input_ids']),
            supervised_tokens=value.supervised_tokens, seconds=time.monotonic() - start))
        del value
        inventory = shared.inventory(torch, model, named, optimizer)
        common.save(out / f'{step:02d}-backward.json', inventory)
        assert inventory['all_grad_finite'] and inventory['frozen_base_no_gradient']
        if step == 0:
            norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            optimizer.step()
            assert not torch.equal(before, first_b.detach())
            optimizer.zero_grad(set_to_none=True)
            common.save(out / 'real-adam.json', shared.inventory(torch, model, named, optimizer))
    final = shared.inventory(torch, model, named, optimizer)
    assert final['optimizer_states'] == final['gradient_tensors'] == 504
    assert final['optimizer_tensor_entries'] == 1512
    estimate = 1.5 * max(x['seconds'] / x['tokens'] for x in durations) * (
        2 * cpu['counts']['train']['tokens']) + model_manifest['load_seconds'] + 120
    result = dict(status='PASS_QWEN3_GROUNDED_NEW_PARETO_ONE_ADAM',
        indexed_forward_backwards=len(durations), actual_optimizer_updates=1,
        formal_estimated_seconds=estimate, formal_budget_gate=estimate <= plan['budgets']['formal'],
        shapes=cpu['pareto_shapes'], durations=durations, final_inventory=final,
        old_CE_and_checkpoint_contract_reused=True, checkpoint_saved_this_probe=False,
        fresh_state_discarded=True, no_held_generation=True, quality_calls=0)
    del optimizer, before
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result
