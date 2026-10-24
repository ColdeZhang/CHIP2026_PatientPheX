"""Small discriminating checks: literal preservation, labels and sample identities."""
import json
import tempfile
from pathlib import Path
from common import append_relations, gold_set, ref, sampled_intersection

with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);u={'d1':{'P1'},'d2':{'P1'}}
    base=[{'pmc_id':p,'entities':[{'text':'a\\"association'}],'association':[{'patient_id':'P1','phenotype':['HP:0000001']}]} for p in u]
    (d/'base').write_text('\n'.join(json.dumps(x,ensure_ascii=False,indent=None) for x in base)+'\n')
    result,counts=append_relations(ref(d/'base'),u,{('d1','P1','HP:0000002'),('d1','P1','HP:0000001')})
    assert counts=={'base_relations':2,'added_relations':1,'entities':2}
    docs=list(map(json.loads,result.splitlines()));assert docs[1]['association'][0]['phenotype']==['HP:0000001']
    old=(d/'base').read_bytes();assert b'"entities": [{"text": "a' in result
    gold=[{'pmc_id':'d1','association':[{'patient_id':'P1','phenotype':['HP:0000001']},{'patient_id':'P1','phenotype':['HP:0000002;HP:0000003','natural;text']}]},{'pmc_id':'d2','association':[]}]
    (d/'gold').write_text('\n'.join(map(json.dumps,gold)))
    assert gold_set(ref(d/'gold'),u)=={('d1','P1','HP:0000002'),('d1','P1','HP:0000003'),('d1','P1','natural;text')}
    raw=[]
    for stream in ('S1','S2'):
        for pmc in u:
            raw.append({'stream':stream,'row_id':pmc,'pmc_id':pmc,'strict_complete':True,'actual_natural_eos':True,'fresh':True,'actual_hit_token_limit':False,'actual_has_think_marker':False,'whole_row_abstention':False,'parser_error':None,'relations':[[pmc,'P1','HP:0000001']]})
    (d/'diag').write_text('\n'.join(map(json.dumps,raw)));assert len(sampled_intersection(ref(d/'diag'),u,2)[0])==2
    raw[-1]['pmc_id']='d1';(d/'bad').write_text('\n'.join(map(json.dumps,raw)))
    try:sampled_intersection(ref(d/'bad'),u,2)
    except AssertionError:pass
    else:raise AssertionError('mismatched row identities accepted')
print('PASS: document-scoped union, literal nonassociation preservation, compound/last-assignment gold, sampled row identity')
