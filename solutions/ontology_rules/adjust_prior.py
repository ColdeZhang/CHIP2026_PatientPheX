#!/usr/bin/env python3
"""Correct a class-weighted, subsampled verifier's prior using training counts.

This is an approximation: document-conditioned negative sampling can change
class-conditional distributions. No inference labels or chosen validation
threshold enter the correction.
"""
import argparse
import gzip
import json
from pathlib import Path
import run as baseline


def corrected(p,population_prior,effective_prior):
    ratio=population_prior/(1-population_prior)/(effective_prior/(1-effective_prior))
    return p*ratio/(1-p+p*ratio)


def main():
    ap=argparse.ArgumentParser()
    for k in ['population-manifest','model-manifest','candidates','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--positive-weight',type=float,required=True)
    a=ap.parse_args()
    population=json.loads(Path(a.population_manifest).read_text());model=json.loads(Path(a.model_manifest).read_text())
    population_prior=population['training_positive']/population['training_examples']
    pos=model['training_positive'];neg=model['training_examples']-pos
    effective_prior=a.positive_weight*pos/(a.positive_weight*pos+neg)
    assert 0<population_prior<1 and 0<effective_prior<1
    with gzip.open(a.candidates,'rt') as f:rows=list(map(json.loads,f))
    for row in rows:
        for c in row['candidates']:
            c['uncorrected_probability']=c['probability']
            c['probability']=corrected(c['probability'],population_prior,effective_prior)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    meta={'config':vars(a),'source_sha256':baseline.sha256(__file__),'inputs':{k:baseline.sha256(getattr(a,k)) for k in ['population_manifest','model_manifest','candidates']},
          'population_prior':population_prior,'effective_prior':effective_prior,'correction_source':'training population and effective weighted sampled training counts only',
          'assumption':'class conditional scores stable under sampling; approximation, not guaranteed posterior calibration','output_sha256':baseline.sha256(out)}
    out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


if __name__=='__main__':main()
