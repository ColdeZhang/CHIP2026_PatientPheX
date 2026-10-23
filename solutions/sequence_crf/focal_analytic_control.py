#!/usr/bin/env python3
"""One input-independent focal decision-boundary control; no threshold fitting."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path

ALPHA = .75
GAMMA = 2.
CUT = .6011759516702939


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def risk_derivative(q, eta, gamma=GAMMA):
    a = (1-q)**(gamma-1)*((1-q)/q-gamma*math.log(q))
    b = q**(gamma-1)*(q/(1-q)-gamma*math.log(1-q))
    return -eta*ALPHA*a+(1-eta)*(1-ALPHA)*b


def check(output):
    import torch
    import focal_loss
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    # This solves the theoretical risk, without any dataset or model scores.
    lo, hi = .5, .75
    for _ in range(80):
        mid = (lo+hi)/2
        if risk_derivative(mid, .5) < 0:
            lo = mid
        else:
            hi = mid
    solved = (lo+hi)/2
    assert abs(solved-CUT) < 1e-14
    records = []
    for q, eta, gamma in [(CUT, .5, 2.), (.5, .25, 2.),
                          (CUT-.001, .5, 2.), (CUT+.001, .5, 2.),
                          (.75, .5, 0.)]:
        z = torch.tensor(math.log(q/(1-q)), dtype=torch.float64, requires_grad=True)
        logits = z.reshape(1, 1, 1)
        mask = torch.ones((1, 1), dtype=torch.bool)
        pos = focal_loss.masked_span_focal(logits, torch.ones_like(logits), mask,
                                         alpha=ALPHA, gamma=gamma)
        neg = focal_loss.masked_span_focal(logits, torch.zeros_like(logits), mask,
                                         alpha=ALPHA, gamma=gamma)
        risk = eta*pos+(1-eta)*neg
        actual = torch.autograd.grad(risk, z)[0].item()
        expected = risk_derivative(q, eta, gamma)*q*(1-q)
        assert abs(actual-expected) < 1e-13
        if q in [CUT, .5, .75]:
            assert abs(actual) < 1e-13
        else:
            assert (actual > 0) == (q > CUT)
        records.append(dict(q=q, eta=eta, gamma=gamma, risk=risk.item(),
                            autograd_logit_derivative=actual,
                            analytic_logit_derivative=expected))
    result = dict(status='PASS', at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256=sha(__file__), actual_loss_sha256=sha(focal_loss.__file__),
                  alpha=ALPHA, gamma=GAMMA, fixed_cut=CUT, solved_cut=solved,
                  checks=records, torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
                  limitations=['Ideal conditional-risk boundary, not a claim of actual calibration.',
                               'The final pipeline also uses unchanged Sap .65/.92 and rules.'])
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def apply(args):
    original = json.loads(Path(args.manifest).read_text())
    assert sha(args.scored) == original['output']['sha256']
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'manifest.json').exists()
    total = kept = 0
    with Path(args.scored).open() as source, (out/'scored-candidates.jsonl').open('w') as target:
        for line in source:
            row = json.loads(line)
            selected = []
            for candidate in row['candidates']:
                score = candidate['score']
                assert isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
                total += 1
                if score >= CUT:
                    selected.append(candidate)
                    kept += 1
            target.write(json.dumps(dict(row, candidates=selected), ensure_ascii=False)+'\n')
    manifest = copy.deepcopy(original)
    manifest['output'] = dict(path=str(out/'scored-candidates.jsonl'),
                              sha256=sha(out/'scored-candidates.jsonl'))
    manifest['analytic_control'] = dict(source_sha256=sha(__file__),
        parent_manifest_sha256=sha(args.manifest), parent_scored_sha256=sha(args.scored),
        alpha=ALPHA, gamma=GAMMA, fixed_span_score_cut=CUT,
        input_candidates=total, kept_candidates=kept,
        gold_read=False, frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy='Only retain cached GLiNER score >= analytic cut; all other fields/order unchanged.',
        limitation='Ideal risk boundary is not an estimate of actual calibrated posterior.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['analytic_control']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only')
    p.add_argument('--scored')
    p.add_argument('--manifest')
    p.add_argument('--output-dir')
    a = p.parse_args()
    if a.check_only:
        assert not any([a.scored, a.manifest, a.output_dir])
        check(a.check_only)
    else:
        assert all([a.scored, a.manifest, a.output_dir])
        apply(a)
