"""GlobalPointer head and multi-label categorical cross entropy.

Equations follow Su et al., arXiv:2208.03054 (RoPE, one phenotype head).
This PyTorch implementation is local; reference implementation bert4keras is
Apache-2.0. Padding/invalid spans are excluded before either logsumexp.
"""
import math
import torch
from torch import nn


def rotate(x):
    half = x.shape[-1] // 2
    pos = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    inv = torch.pow(10000.0, -torch.arange(half, device=x.device).float() / half)
    phase = pos[:, None] * inv[None]
    cos, sin = phase.cos()[None], phase.sin()[None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class GlobalPointer(nn.Module):
    def __init__(self, encoder, head_size=64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 2 * head_size)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.head_size = head_size

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        # Keep the score/loss path FP32 even when encoder uses BF16 autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            q, k = self.head(hidden.float()).chunk(2, -1)
            return torch.einsum('bid,bjd->bij', rotate(q), rotate(k)) / math.sqrt(self.head_size)


def multilabel_loss(logits, positive, valid):
    assert logits.shape == positive.shape == valid.shape
    assert positive.dtype == valid.dtype == torch.bool
    assert not torch.any(positive & ~valid)
    pos = (-logits).masked_fill(~positive, -torch.inf).flatten(1)
    neg = logits.masked_fill(~(valid & ~positive), -torch.inf).flatten(1)
    zeros = logits.new_zeros((logits.shape[0], 1))
    return (torch.logsumexp(torch.cat([pos, zeros], 1), 1)
            + torch.logsumexp(torch.cat([neg, zeros], 1), 1)).mean()
