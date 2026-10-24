"""One immutable probe, fresh two-epoch fit, or complete sixty-four-row inference."""
import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from common import (batch, checked, checkpoint, gpu_gate, load_model, memory,
                    ref, save, sha, source_gate, stamp, update, verify_assets)


def read_encoded(cpu, split):
    return [json.loads(line) for line in gzip.decompress(checked(cpu['counts'][split]['encoded']).read_bytes()).splitlines()]


def unique_object(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError('Duplicate JSON key')
        obj[key] = value
    return obj


def parse_complete(text, row):
    # Exact original normal/fulltext strict policy, not the later R1 codec.
    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != set(row['code_map']):
        raise ValueError('Expected exactly the supplied candidate keys')
    allowed = set(row['patient_ids'])
    for patients in value.values():
        if not isinstance(patients, list):
            raise ValueError('Patient value must be a list')
        if any(type(pid) is not str or pid not in allowed for pid in patients):
            raise ValueError('Unknown patient ID or wrong type')
        if len(patients) != len(set(patients)):
            raise ValueError('Duplicate patient ID')
    return value


def inventory(torch, model, named, optimizer):
    grads = [p.grad for p in named.values() if p.grad is not None]
    tensors = [v for item in optimizer.state.values() for v in item.values() if isinstance(v, torch.Tensor)]
    return {'gradient_tensors': len(grads), 'all_grad_finite': all(bool(torch.isfinite(x).all()) for x in grads),
            'optimizer_states': len(optimizer.state), 'optimizer_tensor_entries': len(tensors),
            'optimizer_tensor_bytes': sum(x.numel() * x.element_size() for x in tensors),
            'frozen_base_no_gradient': all(p.grad is None for p in model.parameters() if not p.requires_grad),
            'memory': memory(torch)}


def probe(plan, output, cpu, torch, model, named, model_manifest):
    """Only the newly authorized full64 shapes, with resident Adam/gradients."""
    from loss_ops import indexed_causal_forward
    rows = read_encoded(cpu, 'train')
    model.train()
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
    sequence = cpu['stress_initial8'] + cpu['stress_accumulated3'] + cpu['pareto_indices']
    assert len(sequence) == 13 and cpu['pareto_shapes'] == [[4992, 116], [6155, 113]]
    assert len(rows) == 265
    first_B = next(p for name, p in named.items() if '.lora_B.' in name)
    before = first_B.detach().clone()
    durations = []
    for step, index in enumerate(sequence):
        row = rows[index]
        tick = time.monotonic()
        update(output / 'progress.json', {'stage': 'new_shape_indexed_forward', 'index': index,
               'row_id': row['row_id'], 'step': step, 'tokens': len(row['input_ids']), 'memory': memory(torch)})
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = indexed_causal_forward(model, **batch(torch, row))
        finite = bool(torch.isfinite(value.loss))
        save(output / f'step-{step:02d}-forward.json', {'finite': finite,
             'loss': float(value.loss.detach()) if finite else None, 'index': index,
             'tokens': len(row['input_ids']), 'supervised_tokens': value.supervised_tokens,
             'logit_shape': list(value.logits.shape), 'memory': memory(torch)})
        assert finite
        (value.loss / 8).backward()
        torch.cuda.synchronize()
        durations.append({'index': index, 'tokens': len(row['input_ids']), 'seconds': time.monotonic() - tick})
        del value
        observed = inventory(torch, model, named, optimizer)
        save(output / f'step-{step:02d}-backward.json', observed)
        assert observed['all_grad_finite'] and observed['frozen_base_no_gradient']
        if step == 7:
            norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            optimizer.step()
            assert not torch.equal(before, first_B.detach())
            optimizer.zero_grad(set_to_none=True)
            save(output / 'real-adam-update.json', inventory(torch, model, named, optimizer))
    final = inventory(torch, model, named, optimizer)
    assert final['optimizer_states'] == final['gradient_tensors'] == 504
    worst = max(item['seconds'] / item['tokens'] for item in durations)
    estimate = 1.25 * worst * (2 * cpu['counts']['train']['tokens']) + model_manifest['load_seconds'] + 120
    return {'status': 'PASS_NEW_FULL64_PARETO_REAL_ADAM_PRESSURE',
            'native_forward_backward': 0, 'indexed_forward_backward': 13, 'optimizer_updates': 1,
            'pareto_indices': cpu['pareto_indices'], 'pareto_shapes': cpu['pareto_shapes'],
            'durations': durations, 'final_inventory': final, 'formal_estimated_seconds': estimate,
            'formal_budget_gate': estimate <= plan['budgets']['formal'],
            'probe_state_must_be_discarded': True, 'old_CE_contract_reused_without_rerun': True,
            'held_prefill_or_generation': 0, 'quality_calls': 0}


def epoch_groups(plan, epoch):
    order = list(range(plan['recipe']['rows']))
    random.Random(plan['seed'] + epoch).shuffle(order)
    groups = [order[k:k + 8] for k in range(0, plan['recipe']['rows'], 8)]
    assert len(groups) == plan['recipe']['updates_per_epoch'] and len(groups[-1]) == 1
    return order, groups


def train(plan, output, cpu, torch, np, model, named):
    from transformers import get_linear_schedule_with_warmup
    from loss_ops import indexed_causal_forward
    rows = read_encoded(cpu, 'train')
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, plan['recipe']['warmup_steps'], plan['recipe']['updates'])
    global_step = 0
    counts = {'rows': 0, 'tokens': 0, 'supervised': 0}
    receipts, updates = [], []
    first_update_verified = False
    started = time.monotonic()
    for epoch in range(2):
        order, groups = epoch_groups(plan, epoch)
        save(output / f'epoch-{epoch + 1:03d}-order.json', {'order': order, 'groups': groups, 'seed': plan['seed'] + epoch})
        for group in groups:
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.
            lr = optimizer.param_groups[0]['lr']
            p = next(p for name, p in named.items() if '.lora_B.' in name)
            before = p.detach().clone() if not first_update_verified and lr > 0 else None
            for index in group:
                row = rows[index]
                update(output / 'progress.json', {'stage': 'formal_forward', 'at_utc': stamp(), 'epoch': epoch + 1,
                       'global_step': global_step, 'row_id': row['row_id'], 'index': index,
                       'tokens': len(row['input_ids']), 'memory': memory(torch)})
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    value = indexed_causal_forward(model, **batch(torch, row))
                finite = bool(torch.isfinite(value.loss))
                update(output / 'last-forward.json', {'row_id': row['row_id'], 'finite': finite,
                       'loss': float(value.loss.detach()) if finite else None, 'memory': memory(torch)})
                assert finite
                (value.loss / len(group)).backward()
                group_loss += float(value.loss.detach()) / len(group)
                counts['rows'] += 1
                counts['tokens'] += len(row['input_ids'])
                counts['supervised'] += value.supervised_tokens
                del value
            norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
            assert bool(torch.isfinite(norm)) and float(norm) > 0
            optimizer.step()
            scheduler.step()
            global_step += 1
            if before is not None:
                assert not torch.equal(before, p.detach())
                first_update_verified = True
            observed = {'epoch': epoch + 1, 'global_step': global_step, 'rows': len(group),
                        'loss': group_loss, 'learning_rate': lr, 'grad_norm': float(norm),
                        'seconds': time.monotonic() - tick, 'memory': memory(torch)}
            updates.append(observed)
            print(json.dumps(observed), flush=True)
        assert global_step == plan['recipe']['updates_per_epoch'] * (epoch + 1)
        receipts.append(checkpoint(torch, np, model, optimizer, scheduler, output,
                                   epoch + 1, global_step, dict(counts), plan))
    assert counts == {'rows': 530, 'tokens': 1732990, 'supervised': 30820}
    assert first_update_verified
    return {'status': 'COMPLETE_FRESH_TWO_EPOCHS_WITH_TWO_DURABLE_ACKS', 'global_step': global_step,
            'counts': counts, 'updates': updates, 'acknowledgements': receipts,
            'training_seconds': time.monotonic() - started, 'memory': memory(torch),
            'no_held_labels_or_quality': True}


def infer(plan, output, cpu, torch, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    rows = read_encoded(cpu, 'dev')
    output.mkdir(exist_ok=True)
    (output / 'generated-ids').mkdir()
    (output / 'raw-responses').mkdir()
    (output / 'canonical-records').mkdir()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    eos_ids.discard(None)
    raw_records, canonical = {}, {}
    for _, _, index in sorted((len(row['input_ids']), row['row_id'], i) for i, row in enumerate(rows)):
        row = rows[index]
        update(output / 'progress.json', {'at_utc': stamp(), 'completed': len(canonical), 'original_index': index,
                                        'row_id': row['row_id'], 'input_tokens': len(row['input_ids'])})
        ids = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
        tick = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                use_cache=True, max_new_tokens=1600, eos_token_id=sorted(eos_ids), pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        tokens = generated[0, ids.shape[1]:].tolist()
        natural = bool(tokens and tokens[-1] in eos_ids)
        cap = len(tokens) >= 1600 and not natural
        identity = {'row_id': row['row_id'], 'pmc_id': row['pmc_id'], 'original_index': index,
                    'natural_eos': natural, 'hit_token_limit': cap, 'generation_seconds': time.monotonic() - tick}
        save(output / 'generated-ids' / f'{index:03d}.json', {**identity, 'generated_ids': tokens, 'configured_eos_ids': sorted(eos_ids)})
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        raw = {**identity, 'response': response, 'input_tokens': ids.shape[1], 'output_tokens': len(tokens), 'provenance': 'fresh'}
        save(output / 'raw-responses' / f'{index:03d}.json', raw)
        parsed, error = None, None
        try:
            if not natural or cap:
                raise ValueError('No natural EOS or generation reached cap')
            parsed = parse_complete(response, row)
        except (ValueError, TypeError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        item = {**identity, 'candidate_patients': parsed if parsed is not None else {},
                'strict_complete': parsed is not None, 'abstention': error, 'provenance': 'fresh'}
        save(output / 'canonical-records' / f'{index:03d}.json', item)
        raw_records[index], canonical[index] = raw, item
        del ids, generated
        print(json.dumps({'completed': len(canonical), 'original_index': index, 'strict_complete': parsed is not None}), flush=True)
    assert len(raw_records) == len(canonical) == 64
    for name, records in [('responses.jsonl', raw_records), ('canonical.jsonl', canonical)]:
        with (output / name).open('x') as stream:
            for index in range(64):
                stream.write(json.dumps(records[index], ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return {'status': 'COMPLETE_SIXTY_FOUR_FRESH_FULLTEXT_NO_QUALITY', 'fresh_rows': 64, 'cached_rows': 0,
            'strict_complete_rows': sum(item['strict_complete'] for item in canonical.values()),
            'whole_row_abstentions': sum(not item['strict_complete'] for item in canonical.values()),
            'responses': ref(output / 'responses.jsonl'), 'canonical': ref(output / 'canonical.jsonl'),
            'memory': memory(torch), 'gold_read': False, 'quality_calls': 0,
            'raw_ids_saved_before_decode': True, 'raw_response_saved_before_parse': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'formal', 'infer'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(plan['run_root']) / args.mode
    output.mkdir(exist_ok=False)
    save(output / 'started.json', {'at_utc': stamp(), 'mode': args.mode, 'pid': os.getpid(),
                                 'plan': ref(args.plan), 'source': ref(__file__)})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original phase deadline reached')))
    stage = 'source_data_asset_gate'
    torch = None
    try:
        source_gate(plan)
        gpu_gate(plan, output)
        loss_contract = json.loads(checked(plan['cpu_loss_result']).read_text())
        assert loss_contract['status'] == plan['cpu_loss_expected_status']
        cpu = json.loads(checked(plan['cpu_result']).read_text())
        assert cpu['status'] == 'PASS_CONDITIONAL_NATIVE_FULL64_CPU_ONLY'
        assert cpu['counts']['train']['rows'] == 265 and cpu['counts']['dev']['rows'] == 64
        assert cpu['planned_2epoch_updates'] == 68 and cpu['last_batch_size'] == 1
        if args.mode == 'probe':
            assets = verify_assets(plan, full=True)
        else:
            previous = json.loads((Path(plan['run_root']) / 'probe' / 'verified-assets.json').read_text())
            assets = verify_assets(plan, full=False, previous=previous)
        save(output / 'verified-assets.json', assets)
        adapter = None
        if args.mode == 'formal':
            previous = json.loads((Path(plan['run_root']) / 'probe/complete.json').read_text())
            assert previous['formal_budget_gate'] and previous['probe_state_must_be_discarded']
        if args.mode == 'infer':
            formal = json.loads((Path(plan['run_root']) / 'formal/complete.json').read_text())
            assert formal['global_step'] == 68 and len(formal['acknowledgements']) == 2
            for epoch in [1, 2]:
                ack = json.loads((Path(plan['run_root']) / f'formal/epoch-{epoch:03d}-persistent-ack.json').read_text())
                checked(ack['checkpoint'])
                for item in ack['files'].values():
                    checked(item)
            adapter = str(Path(plan['run_root']) / 'formal/epoch-002/adapter')
        stage = 'model_load'
        torch, np, model, named, model_manifest = load_model(plan, output, adapter)
        stage = args.mode
        if args.mode == 'probe':
            result = probe(plan, output, cpu, torch, model, named, model_manifest)
        elif args.mode == 'formal':
            result = train(plan, output, cpu, torch, np, model, named)
        else:
            result = infer(plan, output, cpu, torch, model)
        result.update({'at_utc': stamp(), 'elapsed_seconds': time.monotonic() - start, 'phase': args.mode})
        save(output / 'complete.json', result)
    except BaseException as exc:
        failure = {'status': 'FAILED_NO_RETRY', 'at_utc': stamp(), 'stage': stage, 'error': type(exc).__name__,
                   'message': str(exc)[:3000], 'elapsed_seconds': time.monotonic() - start}
        if torch is not None:
            failure['memory'] = memory(torch)
        save(output / 'failure.json', failure)
        raise


if __name__ == '__main__':
    main()
