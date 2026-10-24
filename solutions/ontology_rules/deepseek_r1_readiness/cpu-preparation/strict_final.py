"""Strict native R1 final JSON parsing; missing/truncated output is not negative evidence."""
import json

def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON key')
        out[key]=value
    return out

def parse_final(text, *, eos_observed, stopped_at_cap, code_map, patient_ids):
    if not eos_observed or stopped_at_cap: raise ValueError('truncated_or_no_native_EOS')
    body=text.strip()
    opens=body.count('<think>'); closes=body.count('</think>')
    if opens or closes:
        if opens!=1 or closes!=1 or not body.startswith('<think>'): raise ValueError('malformed_reasoning_boundary')
        _,body=body.split('</think>',1)
        body=body.strip()
    obj=json.loads(body,object_pairs_hook=unique)
    if not isinstance(obj,dict) or set(obj)!=set(code_map): raise ValueError('incomplete_or_unknown_candidate_keys')
    patients=set(patient_ids)
    for value in obj.values():
        if not isinstance(value,list) or any(type(x) is not str or x not in patients for x in value) or len(value)!=len(set(value)):
            raise ValueError('invalid_patient_list')
    return obj

def tests():
    good='{"C000":["P1"],"C001":[]}'
    common=dict(eos_observed=True,stopped_at_cap=False,code_map={'C000':'HP:0000001','C001':'HP:0000002'},patient_ids=['P1'])
    assert parse_final(good,**common)=={'C000':['P1'],'C001':[]}
    assert parse_final('<think>unscored synthetic reasoning</think>\n'+good,**common)=={'C000':['P1'],'C001':[]}
    negatives=[(good,{'eos_observed':False}),(good,{'stopped_at_cap':True}),('<think>'+good,{}),('</think>'+good,{}),('explanation '+good,{}),('```json\n'+good+'\n```',{}),('{"C000":[]}',{}),('{"C000":[],"C001":[],"C002":[]}',{}),('{"C000":["P9"],"C001":[]}',{}),('{"C000":["P1","P1"],"C001":[]}',{}),('{"C000":[],"C000":[],"C001":[]}',{}),('{"C000":"P1","C001":[]}',{})]
    for text,change in negatives:
        try:parse_final(text,**(common|change))
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid synthetic final accepted')
    return dict(valid_cases=2,rejected_cases=len(negatives),no_model_outputs_or_gold=True)
