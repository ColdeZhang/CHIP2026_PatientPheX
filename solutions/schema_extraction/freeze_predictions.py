"""Freeze both complete relation endpoints before any F1 evaluator access."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--normalization',required=True);ap.add_argument('--baseline',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    assert sha(args.baseline)=='f33a30f1f3c505022dbcf846a19e6496b3810627808e97cde595a9ed53d17866'
    assert sha(args.input)=='962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
    folder=Path(args.normalization);summary=json.loads((folder/'complete.json').read_text());assert summary['documents']==16 and summary['patients']==39
    pairs=json.loads((folder/'relations.json').read_text());assert sha(folder/'relations.json')==summary['outputs']['relations.json']['sha256']
    inputs={str(d['pmc_id']):d for d in read(args.input)};baseline=read(args.baseline)
    assert len(baseline)==16 and {str(d['pmc_id']) for d in baseline}==set(inputs)==set(pairs)
    assert sum(len(d['entities']) for d in baseline)==1189
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);outputs={}
    for endpoint in ['N-union-new','new-alone']:
        result=[];count=0
        for doc in baseline:
            pmc=str(doc['pmc_id']);known={str(p['patient_id']) for p in inputs[pmc]['patient']};assert set(pairs[pmc])==known
            old={str(r['patient_id']):r['phenotype'] for r in doc['association']};assert set(old)==known
            new=copy.deepcopy(doc);new['association']=[]
            for patient in inputs[pmc]['patient']:
                pid=str(patient['patient_id']);values=set(pairs[pmc][pid])
                if endpoint=='N-union-new':values.update(old[pid])
                new['association'].append(dict(patient_id=patient['patient_id'],phenotype=sorted(values)));count+=len(values)
            assert new['entities']==doc['entities']
            assert {k:v for k,v in new.items() if k!='association'}=={k:v for k,v in doc.items() if k!='association'}
            result.append(new)
        p=out/(endpoint+'.pred.jsonl')
        p.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in result))
        outputs[endpoint]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),documents=16,patients=39,entities=1189,relations=count)
    manifest=dict(status='BOTH_COMPLETE_PREDICTIONS_FROZEN_BEFORE_EVALUATION',outputs=outputs,
                  normalization_complete_sha256=sha(folder/'complete.json'),baseline_sha256=sha(args.baseline),source_sha256=sha(__file__),quality_evaluations_so_far=0)
    (out/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
