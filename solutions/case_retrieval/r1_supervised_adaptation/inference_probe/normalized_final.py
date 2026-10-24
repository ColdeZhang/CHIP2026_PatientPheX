"""Fixed reversible surface codec: whole JSON fence and exact integer P-ID aliases only."""
import json
from strict_final import unique

def reject_constant(value):raise ValueError('nonfinite_JSON_constant:'+value)
def parse_normalized(text, *, eos_observed, stopped_at_cap, code_map, patient_ids):
    if not eos_observed or stopped_at_cap:raise ValueError('truncated_or_no_native_EOS')
    if any(type(x) is not str for x in patient_ids) or len(patient_ids)!=len(set(patient_ids)):raise ValueError('invalid_or_ambiguous_allowed_patient_list')
    body=text.strip();opens=body.count('<think>');closes=body.count('</think>')
    if opens or closes:
        if opens!=1 or closes!=1 or not body.startswith('<think>'):raise ValueError('malformed_reasoning_boundary')
        _,body=body.split('</think>',1);body=body.strip()
    removed=False;language=None
    if body.startswith('```'):
        lines=body.splitlines()
        if len(lines)<3 or lines[0] not in ['```json','```'] or lines[-1]!='```' or body.count('```')!=2:raise ValueError('not_one_whole_allowed_JSON_fence')
        language='json' if lines[0]=='```json' else '';body='\n'.join(lines[1:-1]);removed=True
    obj=json.loads(body,object_pairs_hook=unique,parse_constant=reject_constant)
    if not isinstance(obj,dict) or not set(obj).issubset(set(code_map)):raise ValueError('nonobject_or_unknown_candidate_keys')
    patients=set(patient_ids);mapped={};aliases=[]
    for cid,values in obj.items():
        if not isinstance(values,list):raise ValueError('patient_value_not_list')
        normalized=[]
        for i,value in enumerate(values):
            if type(value) is str:
                if value not in patients:raise ValueError('unknown_patient_string')
                target=value
            elif type(value) is int:
                if value<0:raise ValueError('negative_patient_integer')
                target='P'+str(value)
                if target not in patients:raise ValueError('no_exact_canonical_P_decimal_alias')
                aliases.append({'CID':cid,'position':i,'original_integer':value,'exact_allowed_PID':target})
            else:raise ValueError('patient_type_not_string_or_integer')
            if target in normalized:raise ValueError('duplicate_patient_after_normalization')
            normalized.append(target)
        mapped[cid]=normalized
    missing=sorted(set(code_map)-set(mapped))
    return {'provided_mapping':mapped,'abstained_CIDs':missing,'normalized_every_CID':not missing,'missing_CIDs_are_negative':False,'whole_outer_fence_removed':removed,'fence_language':language,'integer_PID_aliases_applied':aliases,'semantic_answer_repair_performed':False}

def tests():
    common=dict(eos_observed=True,stopped_at_cap=False,code_map={'C000':'HP:1','C001':'HP:2'},patient_ids=['P1','P2'])
    cases=['{"C000":["P1"],"C001":[]}', '<think>synthetic</think>\n```json\n{"C000":[1],"C001":[]}\n```','```\n{"C000":[2]}\n```','{}']
    out=[parse_normalized(x,**common) for x in cases];assert out[1]['provided_mapping']=={'C000':['P1'],'C001':[]} and out[1]['normalized_every_CID'];assert out[2]['abstained_CIDs']==['C001'] and 'C001' not in out[2]['provided_mapping'];assert out[3]['provided_mapping']=={} and out[3]['abstained_CIDs']==['C000','C001']
    bad=[('{"C000":[1]}',{'eos_observed':False}),('{"C000":[1]}',{'stopped_at_cap':True}),('<think>{"C000":[]}',{}),('</think>{"C000":[]}',{}),('prose\n```json\n{"C000":[]}\n```',{}),('```json\n{"C000":[]}\n```\nprose',{}),('```JSON\n{"C000":[]}\n```',{}),('```python\n{"C000":[]}\n```',{}),('```json\n{"C000":[]}\n```\n```\n{}\n```',{}),('{"C000":[],"C000":[]}',{}),('{"C009":[]}',{}),('{"C000":[true]}',{}),('{"C000":[1.0]}',{}),('{"C000":[-1]}',{}),('{"C000":[9]}',{}),('{"C000":["P01"]}',{}),('{"C000":["1"]}',{}),('{"C000":["P1",1]}',{}),('{"C000":[1,1]}',{}),('{"C000":1}',{}),('{"C000":[NaN]}',{}),('{"C000":[1]}',{'patient_ids':['P01']}),('{"C000":[1]}',{'patient_ids':['P1','P1']})]
    for text,change in bad:
        try:parse_normalized(text,**(common|change))
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid normalized evidence accepted')
    # Canonical integer zero is allowed only when exactly P0 is explicitly listed; no padding alias.
    assert parse_normalized('{"C000":[0]}',**(common|{'patient_ids':['P0']}))['provided_mapping']=={'C000':['P0']}
    return {'valid_cases':len(cases)+1,'rejected_cases':len(bad),'no_gold_model_or_zero_fill':True}
