"""Two explicitly scoped schedules for the same verified two-epoch training loop."""
from pathlib import Path
import math
import random

COMMON = dict(epochs=2, batch_size=1, gradient_accumulation=8, seed=20261023,
              learning_rate=1e-4, weight_decay=.01, max_grad_norm=1., rank=16,
              lora_alpha=32, dropout=.05, dtype='bfloat16', attention='sdpa',
              gradient_checkpointing={'use_reentrant': False}, packing=False,
              quantization=False,
              loss='standard per-row mean causal CE; divide each row by actual group size')
SCHEDULES = {
    'grounded-full64-v1': dict(rows=173, documents=64, held_rows=36, steps_per_epoch=22,
        optimizer_calls=44, warmup_steps=4, tail_group_size=5,
        full_training_tokens=957793, supervised_tokens=24997, maximum_full_tokens=10637),
    'canonical-name-train32-v1': dict(rows=87, documents=32, held_rows=39, steps_per_epoch=11,
        optimizer_calls=22, warmup_steps=2, tail_group_size=7),
}


def validate(configuration, route):
    expected = {**COMMON, **SCHEDULES[route]}
    for name, value in expected.items():
        if configuration.get(name) != value: raise ValueError('Frozen configuration differs: ' + name)
    keys = set(COMMON) | set(SCHEDULES['grounded-full64-v1'])
    if set(configuration) != keys: raise ValueError('Unexpected or missing configuration field')
    for name in ['full_training_tokens', 'supervised_tokens', 'maximum_full_tokens']:
        if type(configuration[name]) is not int or configuration[name] <= 0:
            raise ValueError('Require actual positive token accounting: ' + name)
    if configuration['supervised_tokens'] >= configuration['full_training_tokens']:
        raise ValueError('Prompt must remain masked')
    rows = configuration['rows']; steps = math.ceil(rows / 8)
    if steps != configuration['steps_per_epoch'] or 2 * steps != configuration['optimizer_calls']:
        raise ValueError('Complete two-epoch schedule required')
    return configuration


def batch_plan(configuration, epoch):
    if epoch not in (0, 1): raise ValueError('Only two complete epochs')
    indices = list(range(configuration['rows']))
    random.Random(configuration['seed'] + epoch).shuffle(indices)
    groups = [indices[i:i + 8] for i in range(0, len(indices), 8)]
    assert len(groups) == configuration['steps_per_epoch']
    assert [len(g) for g in groups] == [8] * (len(groups) - 1) + [configuration['tail_group_size']]
    assert sorted(i for group in groups for i in group) == list(range(configuration['rows']))
    return groups


def validate_storage(plan):
    output = Path(plan['output_dir']); parent = Path(plan['storage_parent'])
    if not output.is_absolute() or not parent.is_absolute(): raise ValueError('Absolute owned SHM paths required')
    if parent.parent != Path('/dev/shm') or not parent.name.startswith('patientphex-'):
        raise ValueError('Explicit private SHM parent required')
    if output.parent != parent or output.name != 'output': raise ValueError('One private output root required')
    persistent = Path(plan['persistent_checkpoint_root'])
    project = Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
    persistent.relative_to(project)
    if persistent == project or persistent.name != 'checkpoints': raise ValueError('Private persistent checkpoint root required')
    if plan['persistent_mount_id'] != '5a9955da-5240-41cc-9e52-05d3d8da8a3c':
        raise ValueError('Fixed .100 persistence mount required')
    if plan.get('minimum_SHM_free_bytes', 0) < 4 * 1024**3:
        raise ValueError('Do not lower original SHM4GiB headroom')
    if plan.get('minimum_persistent_free_bytes', 0) < 2 * 1024**3:
        raise ValueError('Do not lower original persistent2GiB headroom')
    return True
