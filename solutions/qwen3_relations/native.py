"""One immutable probe, fresh two-epoch fit, or complete eighty-row inference."""
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
    return [json.loads(line) for line in gzip.decompress(checked(cpu['counts'][split]['file']).read_bytes()).splitlines()]


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
    from transformers.loss.loss_utils import ForCausalLMLoss
    from loss_ops import indexed_causal_forward, indexed_mean_cross_entropy, shifted_supervision
    rows = read_encoded(cpu, 'train')
    shortest = cpu['shortest_index']
    model.eval()
    b = batch(torch, rows[shortest])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        native = model(**b, use_cache=False)
    finite = bool(torch.isfinite(native.loss))
    save(output / 'native-shortest-observed.json', {'row_id': rows[shortest]['row_id'],
         'loss': float(native.loss.detach()) if finite else None, 'finite': finite,
         'logits_shape': list(native.logits.shape), 'memory': memory(torch)})
    assert finite
    # Compare objective and logits derivatives on exactly the same actual logits.
    positions, targets, count = shifted_supervision(b['labels'])
    frozen = native.logits.detach().clone().requires_grad_(True)
    expected = ForCausalLMLoss(frozen, b['labels'], vocab_size=frozen.shape[-1])
    indexed = indexed_mean_cross_entropy(frozen.index_select(1, positions), targets)
    g1 = torch.autograd.grad(expected, frozen, retain_graph=True)[0]
    g2 = torch.autograd.grad(indexed, frozen)[0]
    comparison = {'native_loss': float(native.loss.detach()), 'same_logits_native': float(expected.detach()),
                  'same_logits_indexed': float(indexed.detach()), 'loss_exact': bool(torch.equal(expected, indexed)),
                  'gradient_exact': bool(torch.equal(g1, g2)),
                  'loss_allclose': bool(torch.allclose(expected, indexed, rtol=1e-6, atol=1e-7)),
                  'gradient_allclose': bool(torch.allclose(g1, g2, rtol=1e-6, atol=1e-7)),
                  'predeclared_tolerance': {'rtol': 1e-6, 'atol': 1e-7},
                  'gradient_max_abs': float((g1.float() - g2.float()).abs().max()),
                  'finite': bool(torch.isfinite(g1).all() and torch.isfinite(g2).all()),
                  'supervised_tokens': count, 'positions': positions.tolist(), 'memory': memory(torch)}
    save(output / 'actual-same-logits-contract.json', comparison)
    assert comparison['loss_allclose'] and comparison['gradient_allclose'] and comparison['finite']
    del expected, indexed, g1, g2, frozen
    native.loss.backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in named.values())
    save(output / 'native-shortest-backward.json', {'complete': True, 'memory': memory(torch)})
    model.zero_grad(set_to_none=True)
    del native, b
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
    durations = []
    sequence = cpu['stress_initial_eight'] + cpu['stress_accumulated_three'] + cpu['pareto_indices']
    assert len(sequence) == 11 + len(cpu['pareto_indices'])
    first_B = next(p for name, p in named.items() if '.lora_B.' in name)
    before = first_B.detach().clone()
    for step, index in enumerate(sequence):
        row = rows[index]
        tick = time.monotonic()
        update(output / 'progress.json', {'stage': 'indexed_forward', 'index': index, 'row_id': row['row_id'],
               'step': step, 'tokens': len(row['input_ids']), 'memory': memory(torch)})
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
        seconds = time.monotonic() - tick
        durations.append({'index': index, 'tokens': len(row['input_ids']), 'seconds': seconds})
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
    assert final['optimizer_states'] == 504 and final['gradient_tensors'] == 504
    # Measured worst seconds/token is applied to all original tokens; add slack
    # and two durable checkpoint allowances. This is a cost estimate, not quality.
    worst = max(item['seconds'] / item['tokens'] for item in durations)
    estimate = 1.25 * worst * (2 * 403151) + model_manifest['load_seconds'] + 120
    # Pure-input longest inference prefill: no sampling or task predictions.
    # Training pressure above retained live Adam and accumulated gradients;
    # the independent inference stage does not need those allocations.
    model.zero_grad(set_to_none=True)
    del optimizer, before
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    model.gradient_checkpointing_disable()
    held = read_encoded(cpu, 'held')
    longest = max(held, key=lambda row: len(row['input_ids']))
    ids = torch.tensor([longest['input_ids']], device='cuda', dtype=torch.long)
    prefill_start = time.monotonic()
    with torch.inference_mode():
        prefill = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        logits_to_keep=1, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    held_receipt = {'row_id': longest['row_id'], 'tokens': ids.shape[1], 'forward_only_no_generation': True,
                    'last_logits_shape': list(prefill.logits.shape), 'cache_created': prefill.past_key_values is not None,
                    'finite': bool(torch.isfinite(prefill.logits).all()),
                    'seconds': time.monotonic() - prefill_start, 'memory': memory(torch)}
    save(output / 'longest-fulltext-prefill.json', held_receipt)
    assert held_receipt['tokens'] == 14322 and held_receipt['finite'] and held_receipt['cache_created']
    assert held_receipt['last_logits_shape'] == [1, 1, 151936]
    result = {'status': 'PASS_NATIVE_AND_INDEXED_STANDARD_CE_REAL_ADAM_PARETO_PRESSURE',
              'native_forward_backward': 1, 'indexed_forward_backward': len(sequence),
              'optimizer_updates': 1, 'pareto_indices': cpu['pareto_indices'],
              'durations': durations, 'final_inventory': final, 'formal_estimated_seconds': estimate,
              'formal_budget_gate': estimate <= 2400, 'probe_state_must_be_discarded': True,
              'longest_held_prefill': held_receipt,
              'actual_full_loss_and_logits_derivative_contract': ref(output / 'actual-same-logits-contract.json')}
    return result


def train(plan, output, cpu, torch, np, model, named):
    from transformers import get_linear_schedule_with_warmup
    from loss_ops import indexed_causal_forward
    rows = read_encoded(cpu, 'train')
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 2, 32)
    global_step = 0
    counts = {'rows': 0, 'tokens': 0, 'supervised': 0}
    receipts, updates = [], []
    first_update_verified = False
    started = time.monotonic()
    for epoch in range(2):
        order = list(range(124))
        random.Random(plan['seed'] + epoch).shuffle(order)
        groups = [order[k:k + 8] for k in range(0, 124, 8)]
        assert len(groups) == 16 and len(groups[-1]) == 4
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
        assert global_step == 16 * (epoch + 1)
        receipts.append(checkpoint(torch, np, model, optimizer, scheduler, output,
                                   epoch + 1, global_step, dict(counts), plan))
    assert counts == {'rows': 248, 'tokens': 806302, 'supervised': 14340}
    assert first_update_verified
    return {'status': 'COMPLETE_FRESH_TWO_EPOCHS_WITH_TWO_DURABLE_ACKS', 'global_step': global_step,
            'counts': counts, 'updates': updates, 'acknowledgements': receipts,
            'training_seconds': time.monotonic() - started, 'memory': memory(torch),
            'no_held_labels_or_quality': True}


def infer(plan, output, cpu, torch, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    rows = read_encoded(cpu, 'held')
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
    assert len(raw_records) == len(canonical) == 80
    for name, records in [('responses.jsonl', raw_records), ('canonical.jsonl', canonical)]:
        with (output / name).open('x') as stream:
            for index in range(80):
                stream.write(json.dumps(records[index], ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return {'status': 'COMPLETE_EIGHTY_FRESH_FULLTEXT_NO_QUALITY', 'fresh_rows': 80, 'cached_rows': 0,
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
        cpu = json.loads((Path(plan['cpu_dir']) / 'result.json').read_text())
        assert cpu['status'] == 'PASS_NATIVE_COMPLETE_124_AND_FULL80_PERSISTED'
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
            assert formal['global_step'] == 32 and len(formal['acknowledgements']) == 2
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
