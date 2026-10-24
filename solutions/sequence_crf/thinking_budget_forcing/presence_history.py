"""Presence over saved model-generated tokens and NEW final tokens only.

Exclusion concerns source positions, never token identities: a newline also
present in the artificial separator remains penalized if model-generated.
"""


class HistoricalGeneratedPresence:
    def __init__(self, replay_input_length, generated_history, penalty=1.5):
        if type(replay_input_length) is not int or replay_input_length < 1 or penalty != 1.5:
            raise ValueError("fixed positive replay length and penalty1.5 required")
        if not generated_history or not all(type(x) is int and x >= 0 for x in generated_history):
            raise ValueError("nonempty valid saved generated history required")
        self.prefix_length = replay_input_length
        self.history = tuple(sorted(set(generated_history)))
        self.penalty = penalty
        self.calls = 0
        self.maximum_new_suffix_length = 0
        self.history_tensor = None

    def __call__(self, input_ids, scores):
        import torch
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError("this fixed single-row diagnostic requires batch1")
        if input_ids.shape[1] < self.prefix_length:
            raise ValueError("input shorter than frozen replay prefix")
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("nonfinite logits before presence/sampling warpers")
        if self.history_tensor is None:
            self.history_tensor = torch.tensor(self.history, device=scores.device, dtype=torch.long)
            if self.history[-1] >= scores.shape[-1]:
                raise ValueError("history ID exceeds vocabulary")
        if self.history_tensor.device != scores.device:
            raise ValueError("device changed during this fixed diagnostic")
        present = torch.zeros_like(scores, dtype=torch.bool)
        present[:, self.history_tensor] = True
        suffix = input_ids[:, self.prefix_length:]
        if suffix.shape[1]:
            present.scatter_(1, suffix, True)
        self.calls += 1
        self.maximum_new_suffix_length = max(self.maximum_new_suffix_length, suffix.shape[1])
        return scores - self.penalty * present.to(scores.dtype)


def cpu_contract():
    import torch
    from transformers import LogitsProcessorList
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    # prompt positions [0,1,2], history [3,3,6], injected boundary [4,6,7].
    # ID6 appears both in history and boundary; it MUST still be penalized.
    prefix = torch.tensor([[0, 1, 2, 3, 3, 6, 4, 6, 7]])
    scores = torch.tensor([[3., -2., 1., 0., -1., 2., -3., 4., .5, -4.]])
    original = scores.clone()
    process = HistoricalGeneratedPresence(9, [3, 3, 6])
    wrapped = LogitsProcessorList([process])
    first = wrapped(prefix, scores)
    expected = scores.clone(); expected[0, [3, 6]] -= 1.5
    assert torch.equal(first, expected) and torch.equal(scores, original)
    names = ["saved_history_applies_before_first_final_token", "history_duplicates_count_once",
             "original_prompt_positions_excluded", "artificial_boundary_only_positions_excluded",
             "history_ID_also_in_boundary_still_penalized", "positive_negative_additive", "scores_unmutated"]
    # IDs0 and4 originate in prompt/boundary but also appear in new generation:
    # retain their NEW occurrence; never globally blacklist IDs by source type.
    extra = torch.cat([prefix, torch.tensor([[0, 4, 4, 3]])], dim=1)
    second = wrapped(extra, scores)
    expected = scores.clone(); expected[0, [0, 3, 4, 6]] -= 1.5
    assert torch.equal(second, expected)
    assert process.calls == 2 and process.maximum_new_suffix_length == 4
    names += ["new_ID_also_in_prompt_still_penalized", "new_ID_also_in_boundary_still_penalized",
              "history_and_new_overlap_count_once", "new_duplicates_count_once", "actual_HF_LogitsProcessorList"]
    for name, ids, values, exc in [
        ("short_prefix_rejected", prefix[:, :-1], scores, ValueError),
        ("batch2_rejected", prefix.repeat(2, 1), scores.repeat(2, 1), ValueError),
        ("nonfinite_logits_rejected", prefix, scores * float('nan'), FloatingPointError),
    ]:
        try:
            process(ids, values)
            raise AssertionError(name)
        except exc:
            names.append(name)
    assert not torch.cuda.is_initialized()
    return names
