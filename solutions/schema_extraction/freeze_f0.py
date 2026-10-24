"""Freeze a complete F0 donor, without loading gold or executing an evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
from dataset_contract import check_dataset

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--input',required=True);ap.add_argument('--dataset-contract',required=True);ap.add_argument('--relations',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    assert sha(args.baseline)=='94504b974d5203fdde6b68ff10ca12bb67d24b0db235023023920b0ec595e1bc'
    docs=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x];check_dataset(args.dataset_contract,args.input,docs,'input')
    inputs={str(d['pmc_id']):d for d in docs};baseline=[json.loads(x) for x in Path(args.baseline).read_text().splitlines() if x];relations=json.loads(Path(args.relations).read_text());assert set(inputs)==set(relations)=={str(d['pmc_id']) for d in baseline}
    assert len(baseline)==16 and sum(len(d['entities']) for d in baseline)==947
    outputs=[];pairs=0
    for original in baseline:
        pmc=str(original['pmc_id']);patient_ids=[str(p['patient_id']) for p in inputs[pmc]['patient']];assert set(relations[pmc])==set(patient_ids)
        assert len(patient_ids)==len(set(patient_ids))
        new=json.loads(json.dumps(original));new['association']=[]
        for patient in patient_ids:
            values=relations[pmc][patient];assert isinstance(values,list) and all(isinstance(x,str) and x for x in values);assert len(values)==len(set(values));pairs+=len(values)
            new['association'].append(dict(patient_id=patient,phenotype=sorted(values)))
        assert {k:v for k,v in new.items() if k!='association'}=={k:v for k,v in original.items() if k!='association'}
        outputs.append(new)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);pred=out/'new-alone.pred.jsonl';pred.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in outputs))
    report=dict(kind='F0_COMPLETE_RELATION_DONOR_NO_QUALITY',documents=16,patients=sum(len(x['association']) for x in outputs),entities=947,relation_pairs=pairs,nonassociation_objects_exact=True,nonassociation_literal_bytes_exact=False,gold_accessed=False,quality_evaluations=0,
                source_sha256=sha(__file__),baseline=dict(path=args.baseline,sha256=sha(args.baseline)),input=dict(path=args.input,sha256=sha(args.input)),relations=dict(path=args.relations,sha256=sha(args.relations)),prediction=dict(path=str(pred),bytes=pred.stat().st_size,sha256=sha(pred)))
    assert report['patients']==47
    (out/'freeze.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
