"""Fixed summed-log-prob DPO with one live sequence graph at a time.

The coefficient pass requires deterministic disabled-dropout model forwards.
The actual runtime must check repeated forward agreement before formal use.
"""


def sequence_logp(model, input_ids, prompt_length):
    import torch
    import torch.nn.functional as F
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    assert 0 < prompt_length < input_ids.shape[1]
    positions = torch.arange(prompt_length - 1, input_ids.shape[1] - 1,
                             device=input_ids.device)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                logits_to_keep=positions, use_cache=False, return_dict=True,
                output_attentions=None, output_hidden_states=None)
    logits = out.logits[0].float()
    targets = input_ids[0, prompt_length:]
    assert logits.shape[0] == targets.shape[0]
    return -F.cross_entropy(logits, targets, reduction='sum')


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    import torch.nn.functional as F
    return -F.logsigmoid(beta * ((chosen-rejected)-(ref_chosen-ref_rejected)))


def serial_dpo_backward(model, chosen_ids, rejected_ids, prompt_length,
                        reference_chosen, reference_rejected, group_size,
                        observe, beta=.1):
    """Observe before assert, then backprop exact chain-rule coefficients.

    Caller owns zero_grad/optimizer. No optimizer change occurs between a pair
    or within the accumulation group. Reference values must be frozen caches.
    The small tolerance only accepts numerical repeat variation, not policy or
    dropout changes; runtime engineering evidence must independently pass.
    """
    import torch
    assert group_size > 0 and beta == .1
    with torch.no_grad():
        chosen0 = sequence_logp(model, chosen_ids, prompt_length)
        rejected0 = sequence_logp(model, rejected_ids, prompt_length)
        margin = beta * ((chosen0-rejected0)-(reference_chosen-reference_rejected))
        coefficient = beta * torch.sigmoid(-margin)
        loss = dpo_loss(chosen0, rejected0, reference_chosen, reference_rejected, beta)
    for name, ids, expected, sign in [('chosen',chosen_ids,chosen0,-1),
                                      ('rejected',rejected_ids,rejected0,1)]:
        actual = sequence_logp(model, ids, prompt_length)
        difference = (actual.detach()-expected).abs()
        finite = bool(torch.isfinite(actual) & torch.isfinite(expected))
        # No forced equivalence claim. Record actual values and fail closed if
        # no-grad/grad evaluation differs materially on this exact input.
        passed = finite and bool(torch.isclose(actual.detach(),expected,rtol=1e-5,atol=1e-5))
        observe(dict(kind='coefficient_forward_replay',side=name,finite=finite,
                     no_grad_logp=float(expected),grad_logp=float(actual.detach()),
                     absolute_difference=float(difference) if finite else None,
                     passed=passed,rtol=1e-5,atol=1e-5))
        assert passed, 'Current weights no-grad/grad forward disagreement'
        (sign * coefficient * actual / group_size).backward()
        del actual
    return dict(loss=float(loss),coefficient=float(coefficient),
                chosen_logp=float(chosen0),rejected_logp=float(rejected0))


def disable_dropout(model):
    import torch
    count=0
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.;count+=1
    return count
