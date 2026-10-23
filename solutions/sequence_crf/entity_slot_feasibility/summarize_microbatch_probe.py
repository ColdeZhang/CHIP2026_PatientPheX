#!/usr/bin/env python3
"""Describe completed probe and transparent cost proxies, never run models."""
from pathlib import Path
import argparse, hashlib, json

def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    a=ap.parse_args();b=a.directory;t=b/'terminal'
    done=json.loads((t/'complete.json').read_text());assert done['status']=='microbatch_probe_PASS_state_discarded'
    census=json.loads((b/'group-census.json').read_text())
    samples={r['kind']:r for r in done['groups']}
    points=[(samples[k]['padded_tokens'],samples[k]['seconds']) for k in ['warm-short','warm-median','warm-max-padded']]
    assert points==sorted(points)
    def estimate(x):
        left,right=points[:2] if x<=points[1][0] else points[1:]
        return left[1]+(right[1]-left[1])*(x-left[0])/(right[0]-left[0])
    full_groups=[r for r in census['groups'] if r['rows']==8]
    tail=[r for r in census['groups'] if r['rows']==5]
    step_proxy=sum(estimate(g['padded_tokens']) for g in full_groups)
    step_proxy+=sum(samples['warm-last5']['seconds']*g['padded_tokens']/samples['warm-last5']['padded_tokens'] for g in tail)
    warm=[r for r in done['groups'] if r['kind']!='warmup-short']
    ratio_proxy=census['padded_tokens']*sum(x['seconds'] for x in warm)/sum(x['padded_tokens'] for x in warm)
    report=dict(scope='successful engineering probe only; no formal training or quality',
        actual=done,actual_equivalence=json.loads((t/'equivalence.json').read_text()),
        release=json.loads((t/'gpu-release.json').read_text()),
        population=dict(epochs=2,optimizer_groups=len(census['groups']),rows=census['total_rows'],
            microbatches=census['total_microbatches'],tokens=census['tokens'],padded_tokens=census['padded_tokens']),
        cost_estimates_not_measured=dict(piecewise_padded_cost_seconds=step_proxy,
            pooled_warm_padded_rate_seconds=ratio_proxy,
            suggested_extra_loading_and_checkpoint_margin_seconds=150,
            measured_max_pad=points[-1][0],actual_population_max_pad=max(g['padded_tokens'] for g in census['groups']),
            assumptions='piecewise linear interpolation/extrapolation of three observed full8 groups; real last5 scaled by padding. No scan or fitting quality.',
            limitations=['only four warm groups, not random throughput sample',
                'fixed group selection from epoch0; epoch1 length shape differs',
                'microbatch shape/count, attention shape, dropout draw order and kernel choice affect speed',
                'load/SHA/model save are excluded from estimated update sum; extra150 is only a proposed margin',
                'same objective, not bitwise same batch1 gradient or training trajectory']),
        references={p.name:ref(p) for p in [b/'plan.json',b/'job.json',b/'group-census.json',b/'collection.json',t/'complete.json',t/'equivalence.json',t/'gpu-release.json']},
        source=ref(Path(__file__).resolve()))
    output=b/'post-probe-summary.json';assert not output.exists()
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=ref(output),cost=report['cost_estimates_not_measured'])))

if __name__=='__main__':main()
