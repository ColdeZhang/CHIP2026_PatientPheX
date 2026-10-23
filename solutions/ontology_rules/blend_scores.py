#!/usr/bin/env python3
"""Combine complementary verifiers on exactly the same candidate set."""
import argparse
import gzip
import json
from pathlib import Path
import run as baseline


def load(path):
    with gzip.open(path,'rt') as f:return {r['pmc_id']:r for r in map(json.loads,f)}
def key(c):return c['offset'],c['length'],c['identifier']


def main():
    ap=argparse.ArgumentParser()
    for k in ['primary','secondary','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--secondary-weight',type=float,default=.25)
    ap.add_argument('--mode',choices=['arithmetic','geometric'],default='arithmetic')
    a=ap.parse_args();assert 0<=a.secondary_weight<=1
    primary=load(a.primary);secondary=load(a.secondary);assert set(primary)==set(secondary)
    output=[];w=a.secondary_weight
    for pmc,row in primary.items():
        others={key(c):c for c in secondary[pmc]['candidates']};assert {key(c) for c in row['candidates']}==set(others)
        cs=[]
        for c in row['candidates']:
            p=c['probability'];s=others[key(c)]['probability']
            score=(1-w)*p+w*s if a.mode=='arithmetic' else p**(1-w)*s**w
            cs.append(dict(c,probability=score,primary_probability=p,secondary_probability=s))
        output.append({'pmc_id':pmc,'candidates':cs})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:
        for row in output:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    out.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(a),'source_sha256':baseline.sha256(__file__),'inputs':{k:baseline.sha256(getattr(a,k)) for k in ['primary','secondary']},'input_annotations_used':False,'output_sha256':baseline.sha256(out)},indent=2)+'\n')


if __name__=='__main__':main()
