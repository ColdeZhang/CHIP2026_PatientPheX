"""Proposed weighted assistant JSON loss. No model load, training, or labels native shift."""

def row_loss(model, batch, token_weights, torch):
    """One complete old-JSON row; caller averages row losses over actual group size."""
    labels=batch['labels']
    assert labels.ndim==2 and labels.shape[0]==1
    target_positions=torch.nonzero(labels[0]!=-100,as_tuple=False).flatten()
    assert target_positions.numel()>0 and bool((target_positions>0).all())
    # The logit at j-1 predicts token j. Explicit tensor positions; never take j.
    selected_positions=target_positions-1
    weights=token_weights.to(device=labels.device,dtype=torch.float32)
    assert weights.shape==labels.shape and bool((weights[labels==-100]==0).all())
    selected_weights=weights[0,target_positions]
    assert bool(((selected_weights==1)|(selected_weights==2)).all())
    arguments={k:v for k,v in batch.items() if k!='labels'}
    output=model(**arguments,use_cache=False,logits_to_keep=selected_positions)
    logits=output.logits
    assert logits.shape[:2]==(1,target_positions.numel())
    token_losses=torch.nn.functional.cross_entropy(logits[0].float(),labels[0,target_positions],reduction='none')
    return (token_losses*selected_weights).sum()/selected_weights.sum()
