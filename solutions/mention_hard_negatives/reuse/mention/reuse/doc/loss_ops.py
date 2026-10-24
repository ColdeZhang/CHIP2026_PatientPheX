"""Indexed LM head for the ordinary mean shifted causal cross entropy.

All input tokens still run through the complete causal decoder. Qwen3's native
``logits_to_keep`` indexes the sequence axis immediately before ``lm_head``.
For multiple rows we select the union of required sequence positions, and
retain -100 targets where a particular row has no supervision at that index.
No patient weighting, token reweighting, truncation, or attention change occurs.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class IndexedCausalOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    original_positions: torch.Tensor
    targets: torch.Tensor
    supervised_tokens: int


def shifted_supervision(labels: torch.Tensor, ignore_index: int = -100):
    """Return original logit positions i and their unshifted labels[i + 1]."""
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError('labels must be [batch, sequence >= 2]')
    if labels.dtype != torch.long:
        raise TypeError('labels must have dtype torch.long')
    valid = labels[:, 1:] != ignore_index
    count = int(valid.sum().item())
    if count == 0:
        # Native mean CE is undefined (NaN) here. Reject this invalid training
        # batch explicitly instead of silently returning a zero training loss.
        raise ValueError('mean causal CE requires at least one supervised target')
    positions = valid.any(dim=0).nonzero(as_tuple=False).flatten()
    targets = labels.index_select(1, positions + 1)
    return positions, targets, count


def indexed_mean_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                               ignore_index: int = -100):
    """Same FP32 softmax CE and global valid-token mean as native causal CE."""
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError('indexed logits [B, K, V] and targets [B, K] must align')
    if targets.dtype != torch.long:
        raise TypeError('targets must have dtype torch.long')
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           targets.to(logits.device).reshape(-1),
                           ignore_index=ignore_index, reduction='mean')


def indexed_causal_forward(model, *, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           labels: torch.Tensor, **model_kwargs):
    """Invoke a Qwen3-compatible model once and return loss plus provenance.

Do not pass labels into the model: its automatic causal shift would otherwise
shift the already indexed positions a second time. Cache is disabled for SFT.
The caller owns precision, model mode, RNG, and gradient accumulation.
"""
    forbidden = {'logits_to_keep', 'use_cache', 'num_logits_to_keep', 'num_items_in_batch'}
    if forbidden & model_kwargs.keys():
        raise ValueError('caller cannot override indexed-head/mean-loss semantics')
    if model_kwargs.get('return_dict', True) is not True:
        raise ValueError('indexed causal forward requires return_dict=True')
    if input_ids.ndim != 2 or input_ids.shape != labels.shape:
        raise ValueError('input_ids and labels must have the same [B, L] shape')
    if attention_mask.shape != input_ids.shape:
        raise ValueError('attention_mask must preserve the entire [B, L] input')
    if labels.device != input_ids.device:
        raise ValueError('input_ids and labels must be on the same device')
    positions, targets, count = shifted_supervision(labels)
    model_kwargs['return_dict'] = True
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    logits_to_keep=positions, use_cache=False, **model_kwargs)
    logits = outputs.logits
    loss = indexed_mean_cross_entropy(logits, targets)
    return IndexedCausalOutput(loss, logits, positions, targets, count)
