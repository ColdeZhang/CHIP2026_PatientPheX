"""Bind existing train-only canonical targets to complete pure-input reports."""
import argparse
import json
from pathlib import Path
import time
from common import ref,write,load_jsonl,validate_refs
from reuse.grounded_schema import INPUT_KEYS,messages as grounded_messages
from schema_contract import SYSTEM,strict_parse

def render_names(names):
    return '[\n'+',\n'.join(json.dumps(x,ensure_ascii=False) for x in names)+'\n]' if names else '[]'

def messages(doc,target):
    result=grounded_messages(doc,target)
    result[0]={'role':'system','content':SYSTEM}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();start=time.monotonic();plan=json.loads(Path(a.plan).read_text());validate_refs(plan['inputs'])
    assert plan['phase']=='CPU_PREPARATION_ONLY' and plan['source']==ref(__file__)
    r=plan['inputs'];train=load_jsonl(r['train32']['path']);held=load_jsonl(r['F1_input']['path'])
    inventory=load_jsonl(r['target_inventory']['path']);names=json.loads(Path(r['public_names']['path']).read_text())
    assert len(train)==32 and len(held)==16 and len(inventory)==87 and len(names)==19119
    targets={(x['pmc_id'],x['patient_id']):x for x in inventory};assert len(targets)==87
    assert all(set(d)==INPUT_KEYS for d in held)
    assert not {str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in held}
    rows=[];held_rows=[];counts={'target_relations':0,'true_empty':0,'partial_target_patients':0}
    for doc in train:
        pure={k:doc[k] for k in INPUT_KEYS}
        for patient in pure['patient']:
            pid=patient['patient_id'];t=targets[str(doc['pmc_id']),pid];target=t['canonical_names']
            # Existing train-only inventory binds the exact raw file SHA; no entity/quote target is selected here.
            answer=render_names(target);assert strict_parse(answer,names,eos=True,capped=False)==t['canonical_hpo_ids']
            assert t['canonical_hpo_ids']==sorted(t['canonical_hpo_ids'])
            rows.append(dict(row_id=f"{doc['pmc_id']}:{pid}:canonical",pmc_id=str(doc['pmc_id']),target_patient_id=pid,
                messages=messages(pure,pid),answer=answer))
            counts['target_relations']+=len(target);counts['true_empty']+=t['raw_relation_values']==0
            counts['partial_target_patients']+=bool(t['excluded'])
    for doc in held:
        for patient in doc['patient']:
            pid=patient['patient_id'];held_rows.append(dict(row_id=f"{doc['pmc_id']}:{pid}:canonical",
                pmc_id=str(doc['pmc_id']),target_patient_id=pid,messages=messages(doc,pid)))
    assert len(rows)==87 and len(held_rows)==39 and len({x['row_id'] for x in rows+held_rows})==126
    assert counts=={'target_relations':796,'true_empty':5,'partial_target_patients':14}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    artifacts=dict(train=write(out/'train.jsonl',rows,jsonl=True),held_input=write(out/'F1.input.jsonl',held_rows,jsonl=True))
    result=dict(status='COMPLETE_CANONICAL_TARGETS_INPUT_ONLY',source=ref(__file__),plan=ref(a.plan),
        artifacts=artifacts,inputs=r,train_rows=87,held_rows=39,train_documents=32,held_documents=16,counts=counts,
        target_policy='Exact canonical-name HPO projection, deterministic primary-ID order, all patients retained; NIL/outside-branch positives not silently called negatives.',
        input_policy='Complete original report + given patient mentions; no entity/association/candidate injection.',
        no_held_or_original_dev_gold=True,no_model_no_GPU_no_quality=True,seconds=time.monotonic()-start)
    print(json.dumps(write(out/'manifest.json',result)))

if __name__=='__main__':main()
