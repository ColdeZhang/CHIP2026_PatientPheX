"""Bounded native GLiNER2.5 Task1 training probe; all updates are discarded."""
import argparse
from collections import Counter
import functools
import dataclasses
import random
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.pending')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def observe_batch(batch, lookup, audits, processor, native, cache):
    """Record current stochastic schemas and independently encode their actual tokens."""
    observations=[]
    for j,text in enumerate(batch.original_texts):
        i=lookup[text];n=int(batch.attention_mask[j].sum());combined=[]
        for seq in batch.schema_tokens_list[j]:combined.extend(seq);combined.append(processor.SEP_STRUCT)
        if combined:combined.pop()
        combined.append(processor.SEP_TEXT);combined.extend(batch.text_tokens[j]);expected=[]
        for token in combined:
            if token not in cache:cache[token]=native.encode(token,add_special_tokens=False).ids
            expected.extend(cache[token])
        actual=batch.targets.mention_pairs[j,0][batch.targets.mention_mask[j,0]].tolist()
        sm={a:k for k,a in enumerate(batch.start_mappings[j])};em={b:k+1 for k,b in enumerate(batch.end_mappings[j])}
        observations.append({'row_id':i,'actual_schema_tokens':batch.schema_tokens_list[j],
            'query_layout':dataclasses.asdict(batch.query_layouts[j]),'logical_schema':batch.original_schemas[j],
            'actual_input_ids':batch.input_ids[j,:n].tolist(),'independent_tokenizer_json_ids':expected,
            'actual_positive_pairs':actual,'expected_positive_pairs':audits[i]['actual_target_pairs'],
            'NO_token_pairs':[[sm[a],em[b]] for a,b in audits[i]['NO_chars']],
            'input_ids':n,'query_mask':batch.query_marker_mask[j].tolist(),
            'mention_mask':batch.targets.mention_mask[j].tolist()})
    return observations


def main():
    ap = argparse.ArgumentParser()
    for name in ('model', 'train', 'audit', 'plan', 'output'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {'phase': 'imports', 'started_monotonic': started, 'steps': []}
    write(out/'progress.json', state)
    try:
        import torch, transformers, gliner2
        from gliner2.joint_ie import JointIE
        from tokenizers import Tokenizer
        from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
        import gliner2.models.boundary.model as boundary_module
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        plan = json.loads(Path(args.plan).read_text())
        assert sha(args.train) == plan['data']['train_native_sha256']
        assert sha(args.audit) == plan['data']['target_audit_sha256']
        for asset in plan['assets']:
            p = Path(args.model)/asset['name']
            assert p.stat().st_size == asset['bytes'] and sha(p) == asset['sha256']
        rows = [json.loads(x) for x in Path(args.train).read_text().splitlines()]
        audits = [json.loads(x) for x in Path(args.audit).read_text().splitlines()]
        indices = plan['probe']['indices']; selected = [rows[i] for i in indices]
        assert len(indices) == len(set(indices)) == 6
        assert max(audits[i]['input_ids'] for i in indices) == 698
        assert max(len(audits[i]['positive_token_pairs']) for i in indices) == 49
        assert any(not audits[i]['positive_token_pairs'] for i in indices)
        assert any(audits[i]['NO_chars'] for i in indices)
        lookup = {}
        for i in indices:
            for text in (rows[i]['input'], rows[i]['input'] + '.'):
                assert text not in lookup or lookup[text] == i
                lookup[text] = i
        write(out/'selected.json', {'indices': indices, 'rows': [audits[i] for i in indices],
                                   'train_sha256': sha(args.train), 'audit_sha256': sha(args.audit)})
        state['phase'] = 'load'; write(out/'progress.json', state)
        engine = JointIE.from_pretrained(args.model, device='cpu', dtype='float32', local_files_only=True)
        model = engine.model
        native_tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'));token_cache={}
        assert model.processor.sampling_config.synthetic_entity_label_prob==.2
        assert model.processor.sampling_config.remove_entity_prob==model.processor.sampling_config.remove_entities_prob==0
        assert sum(p.numel() for p in model.parameters()) == 193581591
        assert len(model.state_dict()) == 334
        assert model.architecture == 'boundary'
        assert model.config.boundary_head['max_gold_per_query'] == 64
        inventory = {'parameters': sum(p.numel() for p in model.parameters()),
                     'tensor_count': len(model.state_dict()), 'strict_public_checkpoint_load': True,
                     'torch': torch.__version__, 'transformers': transformers.__version__,
                     'transformers_file': transformers.__file__, 'gliner2_file': gliner2.__file__,
                     'source_sha256': sha(__file__), 'plan_sha256': sha(args.plan),
                     'sampling_config':dataclasses.asdict(model.processor.sampling_config),
                     'boundary_settings': model.boundary_settings.to_dict() if hasattr(model.boundary_settings, 'to_dict') else dict(vars(model.boundary_settings))}
        write(out/'inventory.json', inventory)
        ctx = {'step': None}
        native_finite = boundary_module._finite_loss_term
        @functools.wraps(native_finite)
        def finite_observer(value):
            finite = bool(torch.isfinite(value.detach()).all().item())
            scalar = float(value.detach().float().item()) if finite else None
            if ctx['step'] is not None:
                ctx['step']['raw_loss_terms'].append({'finite': finite, 'value': scalar})
            if not finite:
                write(out/'nonfinite-raw-loss.json', ctx['step'])
                raise FloatingPointError('Native raw loss term is nonfinite before nan_to_num')
            return native_finite(value)
        boundary_module._finite_loss_term = finite_observer
        native_pair = boundary_module.candidate_pair_loss
        native_compute = model.boundary_head._compute_losses
        compute_signature = inspect.signature(native_compute)
        @functools.wraps(native_compute)
        def compute_observer(*a, **kw):
            bound = compute_signature.bind(*a, **kw); bound.apply_defaults()
            pooled = bound.arguments['pooled']; proposals = bound.arguments['proposals']
            ctx['candidate_indices'] = pooled.indices.detach() if pooled is not None else proposals.indices.detach()
            return native_compute(*a, **kw)
        model.boundary_head._compute_losses = compute_observer
        @functools.wraps(native_pair)
        def pair_observer(logits, labels, valid_mask, hard_negative_mask=None, **kw):
            qa, ca = kw.get('query_axis', 1), kw.get('candidate_axis', 2)
            reorder = lambda x: x.movedim((qa, ca), (1, 2))
            labs, valid = reorder(labels.detach()), reorder(valid_mask.detach())
            hard = None if hard_negative_mask is None else reorder(hard_negative_mask.detach())
            eff = valid if hard is None else valid & ((labs > .5) | hard)
            query_mask = kw.get('query_mask')
            if query_mask is not None: eff = eff & query_mask.detach().unsqueeze(-1)
            idx = ctx['candidate_indices']
            if idx.ndim == 3: idx = idx[:, None].expand(-1, labs.shape[1], -1, -1)
            item = {'labels_binary': bool(((labs == 0) | (labs == 1)).all().item()),
                    'valid_pairs': int(valid.sum().item()), 'effective_pairs': int(eff.sum().item()),
                    'effective_positive': int((eff & (labs > .5)).sum().item()),
                    'effective_negative': int((eff & (labs <= .5)).sum().item()),
                    'query_mask': None if query_mask is None else query_mask.detach().cpu().tolist(),
                    'negative_pairs_by_row': [idx[b, 0][eff[b, 0] & (labs[b, 0] <= .5)].cpu().tolist() for b in range(labs.shape[0])]}
            item['NO_coverage'] = []
            for b, no_spans in enumerate(ctx['step']['NO_token_pairs']):
                proposed = {tuple(x) for x in idx[b, 0][valid[b, 0]].cpu().tolist()}
                selected_negatives = {tuple(x) for x in item['negative_pairs_by_row'][b]}
                item['NO_coverage'].append({'annotated': len(no_spans), 'proposed': sum(tuple(x) in proposed for x in no_spans), 'effective_negative': sum(tuple(x) in selected_negatives for x in no_spans)})
            ctx['step']['pair_loss_calls'].append(item)
            return native_pair(logits, labels, valid_mask, hard_negative_mask, **kw)
        boundary_module.candidate_pair_loss = pair_observer

        class ObservedTrainer(ExtractorTrainer):
            def _prepare_data(self, data, is_train=True):
                result = super()._prepare_data(data, is_train=is_train)
                expected = Counter(json.dumps(x, sort_keys=True) for x in selected)
                actual = Counter(json.dumps(x, sort_keys=True) for x in result.data)
                write(out/'loader-identity.json', {'rows': len(result), 'multiset_unchanged': expected == actual})
                assert expected == actual and len(result) == 6
                return result
            def _create_dataloader(self, dataset, batch_size, shuffle=True, is_training=True):
                loader = super()._create_dataloader(dataset, batch_size, shuffle, is_training)
                assert len(loader) == 3 and loader.batch_size == 2 and loader.drop_last
                original_collate=loader.collate_fn
                def observed_collate(raw):
                    before=random.getstate();batch=original_collate(raw);after=random.getstate()
                    ctx['collation_rng']={'python_before':before,'python_after':after}
                    return batch
                loader.collate_fn=observed_collate
                return loader
            def _record_failed_batch(self, batch, *, data_loader_step, error):
                super()._record_failed_batch(batch, data_loader_step=data_loader_step, error=error)
                raise error
            def _evaluate(self, *a, **kw):
                raise AssertionError('No holdout/evaluation permitted')
            def _save_checkpoint(self, name):
                write(out/(name + '-discarded.json'), {'probe_weights_discarded': True, 'global_step': self.global_step, 'epoch': self.epoch})
            def _backward_one(self, batch, step, use_amp, amp_dtype, *, is_last_micro=True):
                torch.cuda.synchronize(); t = time.monotonic()
                row_ids = [lookup[x] for x in batch.original_texts]
                obs = {'index': len(state['steps']), 'epoch': self.epoch + 1, 'batch': step,
                       'row_ids': row_ids, 'raw_loss_terms': [], 'pair_loss_calls': [], 'NO_token_pairs': [],
                       'optimizer_states_before': len(self.optimizer.state), 'input_ids_shape': list(batch.input_ids.shape)}
                ctx['step'] = obs
                raw=observe_batch(batch,lookup,audits,self.processor,native_tokenizer,token_cache)
                write(out/f'batch-{obs["index"]:03d}-actual.json',{'rows':raw,'collation_rng':ctx['collation_rng']})
                for record in raw:
                    assert record['actual_input_ids']==record['independent_tokenizer_json_ids']
                    assert sorted(record['actual_positive_pairs'])==sorted(record['expected_positive_pairs'])
                    assert record['query_mask']==[True] and record['input_ids']<=4096
                    obs['NO_token_pairs'].append(record['NO_token_pairs'])
                obs['actual_input_lengths']=[r['input_ids'] for r in raw]
                state['phase'] = 'backward'; state['active_step'] = obs; write(out/'progress.json', state)
                loss = super()._backward_one(batch, step, use_amp, amp_dtype, is_last_micro=is_last_micro)
                torch.cuda.synchronize()
                obs['forward_backward_seconds'] = time.monotonic() - t
                outputs = self._last_train_outputs
                obs['loss'] = float(loss.item()); obs['losses'] = {k: float(v.detach().item()) for k, v in outputs.losses.items()}
                obs['proposal_metrics'] = {k: float(v.detach().item()) for k, v in (outputs.metrics or {}).items()}
                obs['finite_loss'] = bool(torch.isfinite(loss).item())
                grad_values = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
                obs['gradient_tensors'] = len(grad_values)
                obs['finite_gradients'] = all(bool(torch.isfinite(g).all().item()) for g in grad_values)
                obs['gradient_norm'] = float(torch.stack([g.float().norm() for g in grad_values]).norm().item())
                obs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                obs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                write(out/f'step-{obs["index"]:03d}-backward.json', obs)
                assert obs['finite_loss'] and obs['finite_gradients'] and obs['gradient_norm'] > 0
                assert obs['pair_loss_calls'] and obs['raw_loss_terms']
                return loss
            def _optimizer_step(self):
                obs = ctx['step']; snapshots = []
                for name, p in self.model.named_parameters():
                    if p.grad is None: continue
                    g = p.grad.detach().flatten(); k = int(g.abs().argmax().item())
                    if not bool(g[k] != 0): continue
                    group = 'encoder' if 'encoder' in name else 'task'
                    if any(s['group'] == group for s in snapshots): continue
                    snapshots.append({'name': name, 'group': group, 'index': k, 'before': float(p.detach().flatten()[k].item()), 'parameter': p})
                    if len(snapshots) == 2: break
                torch.cuda.synchronize(); t = time.monotonic()
                applied = super()._optimizer_step()
                torch.cuda.synchronize(); obs['optimizer_seconds'] = time.monotonic() - t
                changes = []
                for s in snapshots:
                    after = float(s['parameter'].detach().flatten()[s['index']].item())
                    changes.append({k: v for k, v in s.items() if k != 'parameter'} | {'after': after, 'changed': after != s['before']})
                obs['parameter_changes'] = changes; obs['optimizer_states_after'] = len(self.optimizer.state)
                obs['applied'] = applied; obs['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(); obs['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                write(out/f'step-{obs["index"]:03d}-complete.json', obs)
                assert applied and any(x['changed'] for x in changes) and obs['optimizer_states_after'] > 0
                state['steps'].append(obs); state['phase'] = 'between_steps'; state.pop('active_step', None); write(out/'progress.json', state)
                return applied

        config = TrainingConfig(output_dir=str(out/'native'), num_epochs=2, batch_size=2,
            encoder_lr=1e-5, task_lr=5e-4, weight_decay=.01, max_grad_norm=1.,
            gradient_accumulation_steps=1, fp16=False, bf16=True, seed=20261024,
            num_workers=0, eval_strategy='no', early_stopping=False, save_best=False,
            max_train_samples=-1, max_steps=-1, strict_training=True, allow_invalid_samples=False,
            skip_step_errors=False, ignore_nonfinite_losses=False, dedupe_contained_entity_spans=False,
            on_capacity_exceeded='raise', log_proposal_metrics=True, diagnostics_every_n_steps=1,
            logging_steps=1, report_to_wandb=False, gold_injection_start=1., gold_injection_end=.25,
            gold_injection_hold_frac=.15, warmup_ratio=.1, scheduler_type='linear')
        trainer = ObservedTrainer(model, config, train_data=selected, eval_data=None)
        torch.cuda.reset_peak_memory_stats()
        summary = trainer.train(train_data=selected, eval_data=None)
        assert summary['total_steps'] == 6 and summary['total_epochs'] == 2 and not summary['eval_metrics_history']
        for epoch in (1, 2):
            seen = [i for x in state['steps'] if x['epoch'] == epoch for i in x['row_ids']]
            assert sorted(seen) == sorted(indices)
        assert all(x['optimizer_states_before'] > 0 for x in state['steps'] if x['epoch'] == 2)
        complete = {'status': 'NATIVE_FB_ADAM_PRESSURE_PROBE_PASS_WEIGHTS_DISCARDED', 'summary': summary,
                    'epochs': 2, 'optimizer_steps': 6, 'full_train_windows': 320, 'probe_windows': 6,
                    'all_selected_shapes_repeated_with_Adam_state': True, 'finite_all_raw_losses_and_gradients': True,
                    'max_gold':49,'prior_uncontrolled_CPU_census_max_wordpieces':698,
                    'actual_max_wordpieces':max(n for x in state['steps'] for n in x['actual_input_lengths']),
                    'native_query_augmentation_probability':.2,
                    'step_seconds': [x['forward_backward_seconds'] + x['optimizer_seconds'] for x in state['steps']],
                    'peak_allocated_bytes': max(x['peak_allocated_bytes'] for x in state['steps']),
                    'peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in state['steps']),
                    'elapsed_seconds': time.monotonic() - started, 'quality_evaluations': 0,
                    'formal_training_authorized': False, 'source_sha256': sha(__file__), 'plan_sha256': sha(args.plan)}
        write(out/'complete.json', complete); print(json.dumps(complete), flush=True)
    except BaseException as exc:
        write(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
              'elapsed_seconds': time.monotonic() - started, 'phase': state.get('phase'), 'completed_steps': len(state['steps'])})
        raise


if __name__ == '__main__':
    main()
