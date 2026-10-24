"""NumPy DPO/serial-gradient contracts; not a Torch or large-model proof."""
import argparse
import datetime
import json
import time
from pathlib import Path
import numpy as np


def logp_gradient(theta, features, labels):
    logits = features @ theta
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    mask = labels >= 0
    selected = np.flatnonzero(mask)
    logp = np.log(probs[selected, labels[mask]]).sum()
    grad_logits = -probs
    grad_logits[selected, labels[mask]] += 1
    grad_logits[~mask] = 0
    return logp, features.T @ grad_logits


def dpo(pc, pr, rc, rr, beta=.1):
    z = beta * ((pc-pr)-(rc-rr))
    return np.logaddexp(0., -z)


def main(output):
    started = time.monotonic()
    rng = np.random.default_rng(20261024)
    theta = rng.normal(size=(3,7))
    chosen = (rng.normal(size=(6,3)), np.array([-100,-100,2,4,5,6]))
    rejected = (rng.normal(size=(7,3)), np.array([-100,-100,2,1,4,5,6]))
    pc,gc = logp_gradient(theta,*chosen)
    pr,gr = logp_gradient(theta,*rejected)
    rc,rr = -12.3,-18.6
    beta=.1
    z=beta*((pc-pr)-(rc-rr))
    coefficient=beta/(1+np.exp(z))
    serial_gradient=-coefficient*gc+coefficient*gr
    finite=np.zeros_like(theta)
    eps=1e-5
    for idx in np.ndindex(theta.shape):
        up=theta.copy();down=theta.copy();up[idx]+=eps;down[idx]-=eps
        def value(t):
            return dpo(logp_gradient(t,*chosen)[0],logp_gradient(t,*rejected)[0],rc,rr,beta)
        finite[idx]=(value(up)-value(down))/(2*eps)
    error=float(np.max(np.abs(finite-serial_gradient)))
    assert error < 1e-8
    assert abs(float(dpo(pc,pr,pc,pr))-np.log(2)) < 1e-14
    assert dpo(pc+.5,pr,rc,rr) < dpo(pc,pr,rc,rr)
    assert dpo(pc,pr+.5,rc,rr) > dpo(pc,pr,rc,rr)
    assert abs(float(dpo(pc+30,pr+30,rc-20,rr-20)-dpo(pc,pr,rc,rr))) < 1e-14
    polluted=chosen[0].copy();polluted[:2]=1e9
    p,g=logp_gradient(theta,polluted,chosen[1])
    assert p==pc and np.array_equal(g,gc)
    # A genuine reference ratio affects the objective; it is not reference-free
    # BCE on one token or an extra gold-only SFT pass.
    assert dpo(pc,pr,rc+2,rr) > dpo(pc,pr,rc,rr)
    result=dict(status='PASS_NUMPY_DPO_SERIAL_GRADIENT_AND_MASK_CONTRACT',
                at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seconds=time.monotonic()-started,finite_difference_coordinates=21,
                max_gradient_abs_error=error,scalar_and_mask_contracts=7,
                unequal_response_lengths_tested=True,reference_ratio_active=True,
                gradient_formula='c=beta*sigmoid(-z); chosen:-c*dlogp; rejected:+c*dlogp',
                note='Exact real-arithmetic chain rule. Not proof that real no-grad/grad CUDA forwards are identical; actual tiny Torch/PEFT and native repeated-forward contract still required.',
                model_loaded=False,GPU_calls=0,fit_calls=0,quality_calls=0)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    main(ap.parse_args().output)
