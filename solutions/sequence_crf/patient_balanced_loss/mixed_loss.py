"""Fixed patient/chunk-balanced mixture after an unchanged complete LM head.

This is the loss core, not a trainer or model launcher. Alpha zero is supported
only as the CPU reference anchor; the proposed training objective is alpha .5.
Metadata positions refer to full labels q, hence causal logits at q - 1.
"""
import torch
from torch.nn import functional as F


def value_coefficients(row, group_size, patients_total, groups_per_epoch):
    """Coefficients inside one row, before the existing outer /len(group)."""
    assert type(group_size) is int and group_size > 0
    assert type(patients_total) is int and patients_total > 0
    assert type(groups_per_epoch) is int and groups_per_epoch > 0
    q = row['supervised_label_positions']
    assert q and q == sorted(set(q)) and all(type(x) is int and x >= 1 for x in q)
    assert all(x < row['sequence_length'] for x in q)
    assert row['supervised_logit_positions'] == [x - 1 for x in q]
    rank = {position: i for i, position in enumerate(q)}
    weights = [0.] * len(q)
    occupied = set()
    patients = row['patients']
    assert patients and [p['patient_index'] for p in patients] == list(range(len(patients)))
    assert len({p['opaque_patient_key'] for p in patients}) == len(patients)
    for patient in patients:
        positions = patient['value_label_positions']
        count = patient['value_token_count']; chunks = patient['n_p']
        assert type(chunks) is int and chunks > 0
        assert type(count) is int and count > 0 and count == len(positions)
        assert positions == sorted(set(positions)) and set(positions) <= set(q)
        assert patient['value_logit_positions'] == [x - 1 for x in positions]
        if occupied.intersection(positions):
            raise ValueError('A supervised token intersects two patient values; no implicit splitting')
        occupied.update(positions)
        coefficient = group_size * groups_per_epoch / (patients_total * chunks * count)
        for position in positions:
            weights[rank[position]] = coefficient
    return weights


def mixed_causal_loss(logits, labels, row, group_size, patients_total,
                      groups_per_epoch, native_fixed_cross_entropy, alpha=.5):
    """The unchanged row CE plus patient/chunk-equal value CE, one full row.

    The future trainer must retain its outer loss / actual group size. This
    function neither constructs a model nor updates parameters. All structures,
    keys, EOS and newline remain supervised by the original half of the loss.
    Full native BPE tokens intersecting one value are included, even when those
    tokens also carry punctuation or key bytes. Empty lists are not discarded.
    """
    assert alpha in (0., .5), 'Only fixed .5 and the alpha-zero CPU anchor exist'
    assert callable(native_fixed_cross_entropy)
    assert labels.ndim == 2 and labels.shape[0] == 1 and labels.dtype == torch.long
    assert logits.ndim == 3 and logits.shape[:2] == labels.shape
    assert row['sequence_length'] <= labels.shape[1]
    assert labels.device == logits.device
    weights = value_coefficients(row, group_size, patients_total, groups_per_epoch)
    q = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    assert q.tolist() == row['supervised_label_positions']
    # Select supervision before FP32. The model's complete head/GEMM is not
    # replaced, projected on selected hidden states, or cast to FP32 here.
    selected = logits[0].index_select(0, q - 1).float()
    target = labels[0].index_select(0, q)
    old_loss = native_fixed_cross_entropy(selected, target, None, -100)
    if alpha == 0.:
        return old_loss
    # Keep the native original-mean call intact. The extra unreduced CE uses
    # the same selected logits and labels, without another model forward.
    token_loss = F.cross_entropy(selected, target, reduction='none')
    coefficient = torch.tensor(weights, dtype=torch.float32, device=logits.device)
    balanced_value_loss = torch.sum(token_loss * coefficient)
    return (1. - alpha) * old_loss + alpha * balanced_value_loss
