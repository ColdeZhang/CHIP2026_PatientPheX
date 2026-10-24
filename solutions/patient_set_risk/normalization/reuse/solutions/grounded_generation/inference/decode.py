"""Deterministic raw-response interpretation; invalid items abstain, never repair."""
import json
from solutions.grounded_generation.schema import grounded, units


def reject_constant(value):
    raise ValueError('Non-JSON constant: '+value)


def interpret(raw, doc, ended_with_eos, cap_reached):
    if not isinstance(raw,str):
        raise TypeError('Raw response must be exact decoded text')
    result=dict(status='abstain',anchors=[],rejected=[],raw_not_repaired=True)
    if not ended_with_eos or cap_reached:
        result['reason']='non_eos_or_token_cap';return result
    try:
        parsed=json.loads(raw,parse_constant=reject_constant)
    except (json.JSONDecodeError,ValueError):
        result['reason']='not_complete_strict_JSON';return result
    if not isinstance(parsed,list):
        result['reason']='not_an_array';return result
    by={unit['unit_id']:unit for unit in units(doc)};seen=set()
    for index,value in enumerate(parsed):
        try:
            anchor=grounded(value,by)
        except (ValueError,TypeError) as error:
            result['rejected'].append(dict(index=index,reason=str(error)));continue
        key=(anchor['offset'],anchor['length'],anchor['quote'])
        if key in seen:
            result['rejected'].append(dict(index=index,reason='duplicate_exact_anchor'));continue
        seen.add(key);result['anchors'].append(anchor)
    result.update(status='complete',reason='itemwise_exact_grounding',generated_items=len(parsed))
    return result


def contracts():
    doc=dict(full_text=[dict(offset=10,text='café café')])
    value=interpret('[["U0000","café",1],["U0000","cafe",0],["U0000","café",1]]',doc,True,False)
    assert len(value['anchors'])==1 and value['anchors'][0]['offset']==15 and len(value['rejected'])==2
    assert interpret('[["U0000","café",true]]',doc,True,False)['anchors']==[]
    for raw in ['```json\n[]\n```','[] trailing','{"items":[]}','[NaN]']:
        assert interpret(raw,doc,True,False)['status']=='abstain'
    assert interpret('[]',doc,False,False)['reason']=='non_eos_or_token_cap'
    assert interpret('[]',doc,True,True)['reason']=='non_eos_or_token_cap'
    assert interpret('[]',doc,True,False)['status']=='complete'
    return dict(Unicode_exact=True,partial_invalid_items_abstain=True,boolean_occurrence_rejected=True,
                no_markdown_or_trailing_repair=True,NaN_rejected=True,EOS_and_cap_gates=True,empty_list_valid=True)
