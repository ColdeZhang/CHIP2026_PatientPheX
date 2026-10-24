"""Actual HF processors: check raw scores before finite source-trie masking."""
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor


class FiniteLogitsOnly(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError('Nonfinite generation logits; stop without retry')
        return scores


def ordered_processors(codec, prompt_width):
    # Do NOT also pass prefix_allowed_tokens_fn to generate: the default
    # prefix processor could then mask before this raw-score numerical guard.
    return LogitsProcessorList([
        FiniteLogitsOnly(),
        PrefixConstrainedLogitsProcessor(codec.callback(prompt_width), num_beams=1),
    ])
