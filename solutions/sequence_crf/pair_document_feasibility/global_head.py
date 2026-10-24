"""Independent NumPy contract for a LEARNABLE head, not a trained encoder/model.

Same parameters and classifier in both arms; only context weights differ.
Every U/W parameter has a classifier path even with uniform context weights.
No third-party ATLOP implementation is copied or imported.
"""
import numpy as np


def initialize(d, width, hidden, seed=20261024):
    rng = np.random.default_rng(seed)
    return dict(U=rng.normal(0,.2,(2*d,width)), bq=np.zeros(width),
                W=rng.normal(0,.2,(d,width)), bz=np.zeros(width),
                A=rng.normal(0,.2,(3*width,hidden)), ba=np.zeros(hidden),
                o=rng.normal(0,.2,hidden), bo=np.array(0.))


def forward(parameters, H, patient, concept, mask, mode):
    if mode not in ('uniform', 'pair'): raise ValueError('unknown arm')
    H, mask = np.asarray(H,dtype=np.float64), np.asarray(mask,dtype=bool)
    assert H.ndim == 2 and mask.shape == (len(H),) and mask.any()
    # Index BEFORE nonlinear projection; poisoned padding/separator states cannot leak.
    valid = H[mask]; x = np.concatenate([patient,concept]).astype(np.float64)
    assert np.isfinite(valid).all() and np.isfinite(x).all()
    q = np.tanh(x @ parameters['U'] + parameters['bq'])
    z = np.tanh(valid @ parameters['W'] + parameters['bz'])
    s = z @ q / np.sqrt(len(q))
    if mode == 'pair':
        exp = np.exp(s-s.max()); weights = exp/exp.sum()
    else: weights = np.full(len(z),1./len(z))
    c = weights @ z
    f = np.concatenate([q,c,q*c]); t = np.tanh(f @ parameters['A']+parameters['ba'])
    logit = t @ parameters['o'] + parameters['bo']
    assert np.isfinite(logit)
    return float(logit), dict(valid=valid,mask=mask,x=x,q=q,z=z,weights=weights,c=c,f=f,t=t,mode=mode)


def loss_and_grad(parameters, H, patient, concept, mask, mode, target):
    """Analytic derivatives for toy gradient contracts; never optimizer-fit real data."""
    assert target in (0,1)
    logit, cache = forward(parameters,H,patient,concept,mask,mode)
    loss = np.logaddexp(0.,logit)-target*logit
    # Stable sigmoid and BCE derivative.
    prob = np.exp(-np.logaddexp(0.,-logit)); dl = prob-target
    x,q,z,a,c,f,t = (cache[k] for k in ('x','q','z','weights','c','f','t'))
    grads = dict(o=t*dl,bo=np.array(dl)); dt = parameters['o']*dl*(1-t*t)
    grads['A'] = np.outer(f,dt); grads['ba'] = dt
    df = parameters['A'] @ dt; m = len(q)
    dq = df[:m]+df[2*m:]*c; dc = df[m:2*m]+df[2*m:]*q
    dz = a[:,None]*dc
    if mode == 'pair':
        da = z @ dc; ds = a*(da-a@da)
        dq += z.T @ ds / np.sqrt(m)
        dz += ds[:,None]*q / np.sqrt(m)
    uz = dz*(1-z*z); uq = dq*(1-q*q)
    grads['W'] = cache['valid'].T @ uz; grads['bz'] = uz.sum(axis=0)
    grads['U'] = np.outer(x,uq); grads['bq'] = uq
    dh = np.zeros_like(H,dtype=np.float64); dh[cache['mask']] = uz @ parameters['W'].T
    dx = parameters['U'] @ uq
    return float(loss),grads,dict(H=dh,patient=dx[:H.shape[1]],concept=dx[H.shape[1]:],logit=logit,prob=float(prob),context=c,weights=a)


def merge_window_copies(windows, maps, n_tokens):
    """maps use canonical document-token indices, -1 for CLS/SEP/padding."""
    assert len(windows) == len(maps) and windows
    d = windows[0].shape[1]; result=np.zeros((n_tokens,d)); counts=np.zeros(n_tokens,dtype=np.int64)
    for hidden,index in zip(windows,maps):
        index=np.asarray(index); assert hidden.shape == (len(index),d)
        for v,i in zip(hidden,index):
            assert -1 <= i < n_tokens
            if i >= 0:
                assert np.isfinite(v).all(); result[i] += v; counts[i] += 1
    assert (counts>0).all(), 'uncovered original token'
    return result/counts[:,None]


def endpoint(H, mentions, valid_mask):
    assert mentions, 'missing endpoint occurrence'
    values=[]
    for positions in mentions:
        assert positions, 'empty occurrence'
        assert all(isinstance(i,int) and 0<=i<len(H) and valid_mask[i] for i in positions)
        values.append(H[positions].mean(axis=0))
    return np.mean(values,axis=0)


def pair_endpoints(H,patient_id,concept_id,patient_mentions,concept_mentions,valid_mask):
    if patient_id not in patient_mentions or concept_id not in concept_mentions:
        raise ValueError('pair outside frozen patient/concept set')
    return endpoint(H,patient_mentions[patient_id],valid_mask),endpoint(H,concept_mentions[concept_id],valid_mask)
