"""OpenAI-style presence penalty over generated suffix only; no frequency term."""


class GeneratedSuffixPresencePenalty:
    def __init__(self, prompt_length, penalty=1.5):
        if type(prompt_length) is not int or prompt_length < 1 or penalty != 1.5:
            raise ValueError("This fixed diagnostic requires positive prefix length and penalty1.5")
        self.prompt_length = prompt_length
        self.penalty = penalty
        self.calls = 0
        self.maximum_suffix_length = 0

    def __call__(self, input_ids, scores):
        import torch
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError("Batch dimensions differ")
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("Input shorter than frozen prompt boundary")
        suffix = input_ids[:, self.prompt_length:]
        present = torch.zeros_like(scores, dtype=torch.bool)
        if suffix.shape[1]:
            present.scatter_(1, suffix, True)
        self.calls += 1
        self.maximum_suffix_length = max(self.maximum_suffix_length, suffix.shape[1])
        return scores - self.penalty * present.to(scores.dtype)
