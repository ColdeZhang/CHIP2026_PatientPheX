"""Create both full majority predictions and one common freeze; no scoring."""
import argparse
from pathlib import Path
import json
import time
from common import (load_plan, inspect_fold, literal_keys, vote, filter_line,
                    ref, checked, read, publish, save, now)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--push-receipt',type=Path,required=True); args=ap.parse_args()
    started=time.monotonic(); plan,guard=load_plan(args.plan)
    receipt=read(args.push_receipt)
    assert receipt['schema']=='majority-source-push-receipt-v1'
    assert receipt['plan']==ref(args.plan) and receipt['sources']==plan['sources']
    assert receipt['remote_verified'] is True
    assert len(receipt['commit'])==40 and all(c in '0123456789abcdef' for c in receipt['commit'])
    root=Path(plan['output_directory']); preparation=read(root/'preparation.json')
    assert preparation['plan']==ref(args.plan) and preparation['sources']==plan['sources']
    output=root/'assembled'; output.mkdir(exist_ok=False)
    predictions={}; reports={}
    for fold in plan['fold_order']:
        lines,baseline,members=inspect_fold(plan['folds'][fold])
        selected,counts=vote(members)
        new_lines=[filter_line(line,selected) for line in lines]
        result=[json.loads(line) for line in new_lines]
        old=literal_keys(baseline); assert literal_keys(result)==selected and selected<=old
        target=output/(fold+'.majority.jsonl'); publish(target,''.join(new_lines).encode())
        predictions[fold]=ref(target)
        reports[fold]={'documents':len(result),'patients':sum(len(d['association']) for d in result),
            'entities':sum(len(d['entities']) for d in result),'baseline_relations':len(old),
            'majority_relations':len(selected),'removed_relations':len(old-selected),'added_relations':0,
            'support_histogram':{str(n):sum(v==n for v in counts.values()) for n in (1,2,3)},
            'outside_phenotype_literal_bytes_exact':True,'retained_phenotype_order_exact':True}
    assert not guard['enabled']
    save(output/'contract.json',reports)
    save(output/'common-freeze.json',{'at_utc':now(),'plan':ref(args.plan),'sources':plan['sources'],
        'push_receipt':ref(args.push_receipt),'preparation':ref(root/'preparation.json'),
        'predictions':predictions,'contract':ref(output/'contract.json'),'both_folds_complete':True,
        'gold_read_or_hashed':False,'evaluator_calls':0,'only_arm':'N_R_P_at_least_two_votes',
        'seconds':time.monotonic()-started})
    print(json.dumps({'freeze':ref(output/'common-freeze.json'),'folds':reports}))


if __name__=='__main__': main()
