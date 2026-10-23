#!/usr/bin/env python3
"""Synthetic-only numerical contract; never reads task model scores or gold."""
from pathlib import Path
import hashlib,itertools,json,random,time
import numpy as np

def expected_f1_by_size(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('probabilities must be finite and between zero and one')
    order=np.argsort(-p,kind='stable');p=p[order];n=len(p)
    if n==0:return order,np.array([1.])
    prefix=[np.ones(1)]
    for v in p:prefix.append(np.convolve(prefix[-1],[1-v,v]))
    suffix=[None]*(n+1);suffix[n]=np.ones(1)
    for i in range(n-1,-1,-1):suffix[i]=np.convolve(suffix[i+1],[1-p[i],p[i]])
    # C[i,k-1] = E[2 Y_i / (k + sum(Y))]. All coefficients nonnegative;
    # no unstable polynomial division when p is close to zero or one.
    reciprocal=1./(np.arange(1,n+1)[:,None]+1+np.arange(n)[None,:])
    contributions=np.stack([2*p[i]*(reciprocal@np.convolve(prefix[i],suffix[i+1])) for i in range(n)])
    score=np.concatenate(([prefix[n][0]],np.cumsum(contributions,axis=0).diagonal()))
    return order,score

def brute(probabilities,selected):
    n=len(probabilities);total=0.
    for bits in itertools.product([0,1],repeat=n):
        probability=np.prod([p if y else 1-p for p,y in zip(probabilities,bits)])
        den=len(selected)+sum(bits)
        value=2*sum(bits[i] for i in selected)/den if den else 1.
        total+=probability*value
    return float(total)

def main():
    started=time.monotonic();rng=random.Random(20261023)
    cases=[[],[0.],[1.],[.5],[.4]*4,[.8,.2],[1.,0.,.5],
           [1e-12,1-1e-12,.5]]+[[rng.random() for _ in range(n)] for n in range(1,8) for _ in range(4)]
    maximum_error=0.;full_subsets=0
    for p in cases:
        order,score=expected_f1_by_size(p)
        for k,v in enumerate(score):maximum_error=max(maximum_error,abs(v-brute(p,set(order[:k]))))
        optimum=-1.
        for bits in itertools.product([0,1],repeat=len(p)):
            optimum=max(optimum,brute(p,{i for i,b in enumerate(bits) if b}));full_subsets+=1
        assert abs(max(score)-optimum)<1e-12
    assert maximum_error<1e-12
    probe=[rng.random() for _ in range(100)];t=time.monotonic();expected_f1_by_size(probe);cost=time.monotonic()-t
    source=Path(__file__)
    report={'scope':'Synthetic-only contract; no real model scores, dataset labels or quality predictions.',
        'cases':len(cases),'full_subsets_checked':full_subsets,'max_absolute_error':maximum_error,
        'ranked_top_k_global_optimum_matches_exhaustive':True,'empty_truth_empty_prediction_f1':1.,
        'single_100_label_cpu_seconds':cost,'total_seconds':time.monotonic()-started,
        'assumption':'Independent Bernoulli labels with calibrated probabilities. Does not reproduce full GFM joint-count model.',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    out=source.parent/'synthetic-contract.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
