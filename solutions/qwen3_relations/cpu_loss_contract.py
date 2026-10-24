"""Actual Torch/TF4.51.3 CPU loss/gradient and tiny Qwen3 interface contract.

No pretrained weights, competition data, GPU, or quality evaluation are used.
Every numerical observation is fsynced before its assertion, including NaN.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time

from loss_ops import indexed_causal_forward, indexed_mean_cross_entropy, shifted_supervision


def ref(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU must be hidden before importing Torch'
    import torch
    import transformers
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen3 import modeling_qwen3
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261024)
    assert not torch.cuda.is_available()
    assert transformers.__version__ == '4.51.3'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    events = out / 'observations.jsonl'
    observations = []

    def record(value):
        observations.append(value)
        with events.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def compare(name, actual, expected, *, rtol=1e-6, atol=1e-7, expect_pass=True):
        a, b = actual.detach().double(), expected.detach().double()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        error = float((a - b).abs().max()) if finite and a.numel() else (0.0 if finite else None)
        passed = finite and bool(torch.allclose(a, b, rtol=rtol, atol=atol))
        record({'name': name, 'shape': list(a.shape), 'finite': finite,
                'max_abs_error': error, 'rtol': rtol, 'atol': atol, 'passed': passed,
                'expected_pass': expect_pass})
        assert passed == expect_pass, name

    forward_source = inspect.getsource(Qwen3ForCausalLM.forward)
    loss_source = inspect.getsource(ForCausalLMLoss)
    (out / 'actual_qwen3_forward.py.txt').write_text(forward_source)
    (out / 'actual_native_causal_loss.py.txt').write_text(loss_source)
    assert 'logits_to_keep' in inspect.signature(Qwen3ForCausalLM.forward).parameters
    assert 'hidden_states[:, slice_indices, :]' in forward_source
    assert 'self.lm_head(' in forward_source
    assert 'else logits_to_keep' in forward_source
    record({'name': 'actual_TF4513_Qwen3_source', 'forward_module': ref(modeling_qwen3.__file__),
            'native_loss_module': ref(inspect.getfile(ForCausalLMLoss)),
            'index_before_lm_head': True, 'whole_decoder_before_index': True})

    masks = [
        [[-100, -100, -100, 3, 4, 5, 6]],
        [[-100, 1, 2, -100, 4, -100, 6], [-100, -100, 3, 4, -100, 6, -100]],
        [[7, -100, -100, 4, 5, -100, -100], [-100, -100, -100, -100, -100, -100, 8],
         [-100, 3, -100, -100, -100, 5, -100]],
        [[-100, 4, 5, 6, 7, 8, 9]],
    ]
    for case, mask in enumerate(masks):
        labels = torch.tensor(mask, dtype=torch.long)
        B, L = labels.shape
        logits = torch.randn(B, L, 19, dtype=torch.float32)
        full = logits.clone().requires_grad_()
        indexed = logits.clone().requires_grad_()
        positions, targets, count = shifted_supervision(labels)
        loss_full = ForCausalLMLoss(full, labels, vocab_size=19)
        loss_indexed = indexed_mean_cross_entropy(indexed[:, positions, :], targets)
        compare(f'frozen_logits_{case}_loss', loss_indexed, loss_full)
        g_full, = torch.autograd.grad(loss_full, full)
        g_indexed, = torch.autograd.grad(loss_indexed, indexed)
        compare(f'frozen_logits_{case}_grad', g_indexed, g_full)
        active = torch.nn.functional.pad(labels[:, 1:] != -100, (0, 1), value=False)
        assert torch.count_nonzero(g_indexed[~active]).item() == 0
        assert count == int(active.sum())
        record({'name': f'frozen_logits_{case}_selection', 'positions': positions.tolist(),
                'supervised_targets': count, 'unselected_logit_gradient_exact_zero': True})

        # Head multiplication is moved after selection. Compare gradients to
        # BOTH hidden states and shared vocabulary projection parameters.
        hidden = torch.randn(B, L, 13)
        weight = torch.randn(19, 13)
        h0, h1 = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
        w0, w1 = weight.clone().requires_grad_(), weight.clone().requires_grad_()
        l0 = ForCausalLMLoss(h0 @ w0.T, labels, vocab_size=19)
        l1 = indexed_mean_cross_entropy(h1[:, positions, :] @ w1.T, targets)
        compare(f'linear_head_{case}_loss', l1, l0)
        gs0, gs1 = torch.autograd.grad(l0, (h0, w0)), torch.autograd.grad(l1, (h1, w1))
        for what, a, b in zip(['hidden', 'weight'], gs1, gs0):
            compare(f'linear_head_{case}_{what}_grad', a, b)

    config = Qwen3Config(vocab_size=31, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=32,
                        attention_dropout=0.0, pad_token_id=0,
                        bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
                        use_sliding_window=False, sliding_window=None)
    config._attn_implementation = 'sdpa'
    base = Qwen3ForCausalLM(config).float().eval()
    for case, (ids, attention, labels) in enumerate([
        ([[1,3,4,5,6,7,8,2]], [[1,1,1,1,1,1,1,1]], [[-100,-100,-100,-100,6,7,8,2]]),
        ([[1,3,4,5,6,2,0,0], [0,0,1,9,10,11,12,2]],
         [[1,1,1,1,1,1,0,0], [0,0,1,1,1,1,1,1]],
         [[-100,-100,4,-100,6,2,-100,-100], [-100,-100,-100,-100,10,11,12,2]])]):
        x, a, y = [torch.tensor(z, dtype=torch.long) for z in (ids, attention, labels)]
        native, sliced = deepcopy(base), deepcopy(base)
        observed = {}
        def capture_outer(module, args, kwargs):
            observed['labels_passed'] = 'labels' in kwargs
            observed['use_cache'] = kwargs['use_cache']
            observed['index'] = kwargs['logits_to_keep'].tolist()
        def capture_body(module, args, kwargs):
            observed['input_identity'] = kwargs['input_ids'] is x
            observed['attention_identity'] = kwargs['attention_mask'] is a
            observed['body_shape'] = list(kwargs['input_ids'].shape)
        def capture_head(module, args):
            observed['head_shape'] = list(args[0].shape)
        handles = [sliced.register_forward_pre_hook(capture_outer, with_kwargs=True),
                   sliced.model.register_forward_pre_hook(capture_body, with_kwargs=True),
                   sliced.lm_head.register_forward_pre_hook(capture_head)]
        native_out = native(input_ids=x, attention_mask=a, labels=y, use_cache=False, return_dict=True)
        actual = indexed_causal_forward(sliced, input_ids=x, attention_mask=a, labels=y)
        positions = actual.original_positions
        record({'name': f'tiny_native_Qwen3_{case}_interface', **observed,
                'original_sequence_length': x.shape[1], 'selected_position_count': len(positions)})
        assert not observed['labels_passed'] and observed['use_cache'] is False
        assert observed['input_identity'] and observed['attention_identity']
        assert observed['head_shape'][1] == len(positions) < x.shape[1]
        compare(f'tiny_native_Qwen3_{case}_selected_logits', actual.logits, native_out.logits[:, positions])
        compare(f'tiny_native_Qwen3_{case}_loss', actual.loss, native_out.loss)
        native_out.loss.backward()
        actual.loss.backward()
        for (name0, p0), (name1, p1) in zip(native.named_parameters(), sliced.named_parameters()):
            assert name0 == name1 and p0.grad is not None and p1.grad is not None
            compare(f'tiny_native_Qwen3_{case}_gradient:{name0}', p1.grad, p0.grad)
        # A prefix token with no local LM target still receives a decoder
        # gradient: attention context was not pruned along with head positions.
        # Token1 is only present at positions whose next label is ignored in
        # BOTH fixtures; token3 in the second fixture would have a direct target.
        next_active = torch.nn.functional.pad(y[:, 1:] != -100, (0, 1), value=False)
        assert not bool(next_active[x == 1].any())
        prefix_grad = sliced.model.embed_tokens.weight.grad[1].abs().sum().item()
        record({'name': f'tiny_native_Qwen3_{case}_unsupervised_prefix_context',
                'token1_grad_l1': prefix_grad, 'direct_supervised_logit_positions': 0,
                'positive': prefix_grad > 0})
        assert prefix_grad > 0
        for handle in handles:
            handle.remove()

    # Exercise the production PEFT wrapper, including its ordinary diagnostic
    # kwargs. Identical RNG restores paired LoRA dropout masks at train time.
    assert peft.__version__ == '0.15.2'
    wrapped = get_peft_model(deepcopy(base), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'], bias='none')).train()
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(mean=0, std=.001)
    native, sliced = deepcopy(wrapped), deepcopy(wrapped)
    forwarded = {}
    original_forward = sliced.base_model.model.forward
    @wraps(original_forward)
    def capture_peft_native(*args, **kwargs):
        forwarded['native_labels_none'] = kwargs.get('labels') is None
        forwarded['logits_to_keep'] = kwargs['logits_to_keep'].tolist()
        forwarded['use_cache_false'] = kwargs.get('use_cache') is False
        forwarded['whole_input_identity'] = kwargs['input_ids'] is x
        forwarded['whole_attention_identity'] = kwargs['attention_mask'] is a
        return original_forward(*args, **kwargs)
    # PEFT may call the wrapped model's .forward directly, bypassing Module
    # __call__ hooks. Preserve its exact bound forward behind a transparent
    # observer rather than treating an untriggered hook as a model failure.
    sliced.base_model.model.forward = capture_peft_native
    rng = torch.random.get_rng_state()
    native_out = native(input_ids=x, attention_mask=a, labels=y, use_cache=False,
                        return_dict=True, output_attentions=None, output_hidden_states=None)
    torch.random.set_rng_state(rng)
    actual = indexed_causal_forward(sliced, input_ids=x, attention_mask=a, labels=y,
                                    output_attentions=None, output_hidden_states=None)
    record({'name': 'PEFT_actual_wrapper_forward_contract', 'peft': peft.__version__, **forwarded})
    assert all(forwarded[key] for key in ['native_labels_none', 'use_cache_false',
                                         'whole_input_identity', 'whole_attention_identity'])
    compare('PEFT_actual_selected_logits', actual.logits, native_out.logits[:, actual.original_positions])
    compare('PEFT_actual_loss', actual.loss, native_out.loss)
    native_out.loss.backward()
    actual.loss.backward()
    nonzero_adapter_gradients = 0
    adapter_parameters = 0
    for (name0, p0), (name1, p1) in zip(native.named_parameters(), sliced.named_parameters()):
        assert name0 == name1
        if p0.requires_grad:
            assert p0.grad is not None and p1.grad is not None
            compare(f'PEFT_actual_gradient:{name0}', p1.grad, p0.grad)
            adapter_parameters += 1
            nonzero_adapter_gradients += int(bool(torch.count_nonzero(p1.grad)))
        else:
            assert p0.grad is None and p1.grad is None
    record({'name': 'PEFT_gradient_coverage', 'trainable_tensors_compared': adapter_parameters,
            'nonzero_adapter_gradients': nonzero_adapter_gradients, 'frozen_base_gradients_absent': True})
    assert adapter_parameters > 0 and nonzero_adapter_gradients == adapter_parameters
    sliced.base_model.model.forward = original_forward

    # Verify failed numerical evidence is durable before assertion handling.
    before = len(observations)
    nan = torch.tensor(float('nan'))
    compare('intentional_NaN_failure_is_saved', nan, nan, expect_pass=False)
    on_disk = [json.loads(line) for line in events.read_text().splitlines()]
    assert len(on_disk) == before + 1 and on_disk[-1]['finite'] is False
    rejected = []
    for label, call in [
        ('all_targets_ignored', lambda: shifted_supervision(torch.full((2, 3), -100, dtype=torch.long))),
        ('float_labels', lambda: shifted_supervision(torch.zeros((1, 3)))),
        ('rank_one_labels', lambda: shifted_supervision(torch.ones(3, dtype=torch.long))),
        ('override_cache', lambda: indexed_causal_forward(base, input_ids=x, attention_mask=a, labels=y, use_cache=True)),
        ('override_indices', lambda: indexed_causal_forward(base, input_ids=x, attention_mask=a, labels=y, logits_to_keep=2)),
        ('sum_normalization', lambda: indexed_causal_forward(base, input_ids=x, attention_mask=a, labels=y, num_items_in_batch=3)),
    ]:
        try:
            call()
        except (ValueError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError(label)
    record({'name': 'explicit_invalid_input_rejections', 'rejected': rejected})
    result = {'status': 'PASS_NATIVE_TF4513_CPU_INDEXED_HEAD_STANDARD_MEAN_CAUSAL_CE',
              'elapsed_seconds': time.monotonic() - started, 'torch': torch.__version__,
              'transformers': transformers.__version__, 'peft': peft.__version__, 'python': sys.version,
              'device': 'cpu', 'torch_cuda_available': torch.cuda.is_available(),
              'threads': torch.get_num_threads(), 'frozen_logit_cases': len(masks),
              'tiny_random_Qwen3_cases': 2, 'actual_PEFT_forward_backward_cases': 1,
              'pretrained_weights_loaded': False,
              'competition_data_read': False, 'GPU_calls': 0,
              'observations': ref(events), 'source': ref(__file__),
              'loss_ops': ref(Path(__file__).with_name('loss_ops.py')),
              'native_forward': ref(out / 'actual_qwen3_forward.py.txt'),
              'native_loss': ref(out / 'actual_native_causal_loss.py.txt'),
              'scope': 'CPU frozen-logit, hidden/head-gradient and actual tiny-model interface checks. Does not establish production GPU throughput, memory, or cross-kernel bitwise equivalence.'}
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
