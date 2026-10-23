#!/usr/bin/env python3
"""Training-internal ablation of independent patient relation decisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import run as base


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',nargs='+',required=True);p.add_argument('--votes',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    inputs=[{str(d['pmc_id']):d for d in base.read_jsonl(x)} for x in a.predictions];assert all(set(x)==set(inputs[0]) for x in inputs)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as h:
        for pmc,pr in inputs[0].items():
            assert all(x[pmc]['entities']==pr['entities'] for x in inputs),'Keep the identical entity pool'
            votes=Counter()
            for inp in inputs:votes.update({(row['patient_id'],v) for row in inp[pmc]['association'] for v in row.get('phenotype',[])})
            association=[{'patient_id':row['patient_id'],'phenotype':sorted(v for (pid,v),n in votes.items() if pid==row['patient_id'] and n>=a.votes)} for row in pr['association']]
            h.write(json.dumps(dict(pr,association=association),ensure_ascii=False)+'\n')
    out.with_suffix('.manifest.json').write_text(json.dumps({'config':vars(a),'source_sha256':base.sha(__file__),'source_predictions':{x:base.sha(x) for x in a.predictions},'output_sha256':base.sha(out),'uses_gold':False},indent=2)+'\n')
if __name__=='__main__':main()
