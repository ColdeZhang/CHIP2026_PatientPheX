"""Frozen model + complete F0 raw union; assembly only, cannot read gold/score."""
import argparse
import json
import math
from pathlib import Path
from common import FEATURES, append_relations, checked, mkdir_new, now, read_json, ref, write_bytes, write_json

from common_f0 import candidates


def run(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    assert plan['status'] == 'BOUND_COMPLETE_F0_SOURCES'
    assert plan['documents'] == 16 and plan['patients'] == 47 and plan['sample_rows_per_seed'] == 58
    assert plan['features'] == FEATURES and plan['cut'] == 0.5
    for r in plan['sources'].values(): checked(r)
    assert read_json(plan['F0_ancestry'])['all_task_ancestors_exclude_F0']
    model = read_json(plan['model_state'])
    assert model['features'] == FEATURES and model['cut'] == 0.5 and model['classes'] == [0,1]
    assert len(model['coef']) == 1 and len(model['coef'][0]) == 3 and len(model['intercept']) == 1
    out=mkdir_new(output)
    write_json(out/'started.json',{'utc':now(),'plan':ref(plan_path),'gold_access':False})
    u,rows,counts=candidates(plan)
    # Coefficients are already fitted on F1. Stable sigmoid, no sklearn/fit here.
    scored=[]; selected=set()
    for row in rows:
        z=model['intercept'][0]+sum(a*b for a,b in zip(model['coef'][0],row['features']))
        assert math.isfinite(z)
        probability=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        scored.append({**row,'probability':probability,'selected':probability>=0.5})
        if probability>=0.5: selected.add(tuple(row['key']))
    scores=write_bytes(out/'all-raw-union-scores.jsonl',b''.join((json.dumps(x,separators=(',',':'))+'\n').encode() for x in scored))
    # B0 is not opened until all raw features and selections have been produced.
    payload, preserved=append_relations(plan['baseline'],u,selected)
    assert preserved['entities']==947
    pred=write_bytes(out/'prediction.jsonl',payload)
    write_json(out/'freeze.json',{'utc':now(),'prediction':pred,'plan':ref(plan_path),'model_state':plan['model_state'],'raw_sources':plan['raw_sources'],'all_union_scores':scores,'counts':counts,'selected_raw_relations':len(selected),'preservation':preserved,'documents':16,'patients':47,'nonassociation_bytes_exact':True,'baseline_all_associations_and_order_preserved':True,'gold_read':False,'v2_calls':0,'next':'Wait for root joint freeze binding this prediction and fixed2of3; no evaluation is provided by this entry.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.plan,a.output)
