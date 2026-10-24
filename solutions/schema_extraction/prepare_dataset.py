"""Reproduce an explicitly frozen pure-input dataset with unchanged passage maps."""
import argparse
import json
from pathlib import Path
from dataset_contract import check_dataset
from prepare import build,contract,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--dataset-contract',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    raw=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x];data=check_dataset(args.dataset_contract,args.input,raw,'input');docs=[build(d) for d in raw]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);p=out/'held.jsonl';p.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs));check_dataset(args.dataset_contract,p,docs,'prepared')
    report=dict(contract=contract(),documents=len(docs),patients=sum(len(d['patients']) for d in docs),source_sha256=sha(__file__),data_contract=data,input_sha256=sha(args.input),prepared_sha256=sha(p),gold_accessed=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
