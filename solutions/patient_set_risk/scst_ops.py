"""New loss only: sum of full categorical suffix log probabilities, plus CE.

No token selection/mask is derived from rewards. Real EOS is part of the suffix.
The caller separately freezes detached rewards and checks exact prompt identity.
"""

def disable_dropout(model):
    import torch
    changed={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Dropout):changed[name]=module.p;module.p=0.
        # Native SDPA passes this scalar directly instead of a Dropout module.
        if hasattr(module,'attention_dropout'):
            assert isinstance(module.attention_dropout,(int,float))
            changed[name+'.attention_dropout']=module.attention_dropout;module.attention_dropout=0.
    return changed

def token_logps(model,input_ids,labels):
    import torch
    import torch.nn.functional as F
    assert input_ids.ndim==labels.ndim==2 and input_ids.shape==labels.shape and input_ids.shape[0]==1
    positions=torch.where(labels[0,1:]!=-100)[0]
    assert positions.numel()>0
    targets=labels[0,positions+1]
    assert torch.equal(input_ids[0,positions+1],targets)
    output=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),labels=None,
                 use_cache=False,logits_to_keep=positions,return_dict=True)
    assert output.logits.shape[:2]==(1,positions.numel())
    return -F.cross_entropy(output.logits[0].float(),targets,reduction='none')

def replay_observation(reference,replay):
    import torch
    diff=(reference.detach().float()-replay.detach().float()).abs()
    finite=bool(torch.isfinite(reference).all() and torch.isfinite(replay.detach()).all())
    def value(x):return float(x) if bool(torch.isfinite(x)) else None
    return dict(tokens=reference.numel(),max_absolute_token_difference=value(diff.max()),
                sum_absolute_token_difference=value(diff.sum()),reference_sum=value(reference.sum()),
                replay_sum=value(replay.detach().sum()),all_finite=finite,
                nonfinite_reference=int((~torch.isfinite(reference)).sum()),nonfinite_replay=int((~torch.isfinite(replay.detach())).sum()),
                allclose_rtol_atol_1e5=finite and bool(torch.allclose(reference,replay.detach(),rtol=1e-5,atol=1e-5)))

def family(name):
    if '.linear_attn.' in name:return 'DeltaNet'
    if '.self_attn.' in name:return 'full_attention'
    if '.mlp.' in name:return 'MLP'
    raise AssertionError('Unknown LoRA family: '+name)
