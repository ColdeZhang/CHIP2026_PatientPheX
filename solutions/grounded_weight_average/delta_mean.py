"""Exact equal-effective-delta concatenation for the frozen G32 vanilla LoRAs.

Pure NumPy helper. No model loading, downloading, training or quality entry.
Actual checkpoint deserialization remains unexecuted in this readiness phase.
"""
from copy import deepcopy
import numpy as np

TARGETS = {
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z',
    'in_proj_b', 'in_proj_a', 'out_proj', 'gate_proj', 'up_proj', 'down_proj',
}


def validate_configs(first, second):
    if first != second:
        raise ValueError('The two actual G32 config objects must agree completely')
    required = dict(peft_type='LORA', task_type='CAUSAL_LM', r=16,
        lora_alpha=32, lora_dropout=.05, bias='none', lora_bias=False,
        use_dora=False, use_rslora=False, use_qalora=False,
        fan_in_fan_out=False, modules_to_save=None, rank_pattern={},
        alpha_pattern={}, inference_mode=True, target_parameters=None,
        trainable_token_indices=None, layer_replication=None,
        exclude_modules=None, layers_pattern=None, layers_to_transform=None,
        alora_invocation_tokens=None, init_lora_weights=True,
        ensure_weight_tying=False, peft_version='0.21.2')
    for name, expected in required.items():
        if first.get(name) != expected:
            raise ValueError('Unsupported G32 LoRA configuration: ' + name)
    for name in ['arrow_config', 'corda_config', 'eva_config', 'kasa_config',
                 'lora_ga_config', 'megatron_config', 'monteclora_config',
                 'use_bdlora', 'velora_config']:
        if first.get(name) is not None:
            raise ValueError('Nonvanilla LoRA variant: ' + name)
    if first.get('loftq_config') != {} or set(first['target_modules']) != TARGETS:
        raise ValueError('Target module set or quantized initialization differs')
    if len(first['target_modules']) != len(TARGETS):
        raise ValueError('Repeated target module')
    result = deepcopy(first)
    result.update(r=32, lora_alpha=32)
    return result


def concatenate_pair(a1, b1, a2, b2):
    arrays = (a1, b1, a2, b2)
    if any(not isinstance(value, np.ndarray) or value.ndim != 2 for value in arrays):
        raise ValueError('Every LoRA factor must be a 2D NumPy array')
    if not (a1.shape == a2.shape and b1.shape == b2.shape
            and a1.shape[0] == b1.shape[1] == 16):
        raise ValueError('Expected two matching ordinary rank16 A/B pairs')
    if not all(value.dtype == a1.dtype and np.issubdtype(value.dtype, np.floating)
               and np.isfinite(value).all() for value in arrays):
        raise ValueError('Factors require one finite floating dtype')
    # .5 * old_scaling2 = 1, so factor values themselves are copied exactly.
    return np.concatenate((a1, a2), axis=0), np.concatenate((b1, b2), axis=1)


def compose_state(first, second, config1, config2, expected_prefixes):
    """Reject missing/extra/non-LoRA state; preserve all saved factor keys."""
    config = validate_configs(config1, config2)
    prefixes = tuple(expected_prefixes)
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise ValueError('Expected complete unique target-module inventory')
    if any(prefix.rsplit('.', 1)[-1] not in TARGETS for prefix in prefixes):
        raise ValueError('Unexpected target family')
    expected = {prefix + suffix for prefix in prefixes
                for suffix in ['.lora_A.weight', '.lora_B.weight']}
    if set(first) != expected or set(second) != expected:
        raise ValueError('Both saved adapters must have exact full A/B key inventory')
    result = {}
    for prefix in prefixes:
        a, b = prefix + '.lora_A.weight', prefix + '.lora_B.weight'
        result[a], result[b] = concatenate_pair(first[a], first[b], second[a], second[b])
    return result, config
