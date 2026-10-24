"""Selected-supervision FP32 CE after the unchanged full BF16 LM head."""
import torch
from torch import nn
from transformers.loss.loss_utils import fixed_cross_entropy as native_fixed_cross_entropy

def selected_causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                         ignore_index=-100, shift_labels=None, **kwargs):
    # Keep the complete original BF16 head output. No selected hidden states,
    # logits_to_keep, model forward or head GEMM changes occur in this function.
    if shift_labels is None:
        padded = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = padded[..., 1:].contiguous()
    target = shift_labels.view(-1).to(logits.device)
    keep = target != ignore_index
    source = logits.view(-1, vocab_size)[keep].float()
    # Empty selected rows retain native CE mean NaN or sum/explicit-denominator
    # behavior. No invented zero-loss success or group/token reweighting.
    return native_fixed_cross_entropy(source, target[keep], num_items_in_batch,
                                      ignore_index, **kwargs)
