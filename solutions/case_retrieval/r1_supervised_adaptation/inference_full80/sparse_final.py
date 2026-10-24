"""Predeclared sparse evidence policy; missing keys are abstention, never fabricated negatives."""
import json
from strict_final import unique,parse_final

def parse_sparse(text, *, eos_observed, stopped_at_cap, code_map, patient_ids):
    if not eos_observed or stopped_at_cap:raise ValueError('truncated_or_no_native_EOS')
    body=text.strip();opens=body.count('<think>');closes=body.count('</think>')
    if opens or closes:
        if opens!=1 or closes!=1 or not body.startswith('<think>'):raise ValueError('malformed_reasoning_boundary')
        _,body=body.split('</think>',1);body=body.strip()
    obj=json.loads(body,object_pairs_hook=unique)
    if not isinstance(obj,dict) or not set(obj).issubset(set(code_map)):raise ValueError('nonobject_or_unknown_candidate_keys')
    patients=set(patient_ids)
    for values in obj.values():
        if not isinstance(values,list) or any(type(x) is not str or x not in patients for x in values) or len(values)!=len(set(values)):raise ValueError('invalid_patient_list')
    return {'provided_mapping':obj,'abstained_CIDs':sorted(set(code_map)-set(obj)),'strict_every_CID':set(obj)==set(code_map),'missing_CIDs_are_negative':False}

def tests():
    common=dict(eos_observed=True,stopped_at_cap=False,code_map={'C000':'HP:1','C001':'HP:2'},patient_ids=['P1'])
    full='{"C000":["P1"],"C001":[]}';sparse='{"C000":["P1"]}'
    assert parse_sparse(full,**common)['provided_mapping']==parse_final(full,**common)
    r=parse_sparse('<think>synthetic</think>\n'+sparse,**common);assert r['provided_mapping']=={'C000':['P1']} and r['abstained_CIDs']==['C001'] and r['strict_every_CID'] is False and 'C001' not in r['provided_mapping']
    r=parse_sparse('{}',**common);assert r['provided_mapping']=={} and r['abstained_CIDs']==['C000','C001']
    bad=[(sparse,{'eos_observed':False}),(sparse,{'stopped_at_cap':True}),('<think>'+sparse,{}),('</think>'+sparse,{}),('```json\n'+sparse+'\n```',{}),('prose '+sparse,{}),('{"C009":[]}',{}),('{"C000":["P9"]}',{}),('{"C000":["P1","P1"]}',{}),('{"C000":[],"C000":[]}',{}),('{"C000":"P1"}',{}),('[]',{})]
    for text,change in bad:
        try:parse_sparse(text,**(common|change))
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid sparse evidence accepted')
    return {'valid_cases':3,'rejected_cases':len(bad),'no_model_gold_or_answer_repair':True}
