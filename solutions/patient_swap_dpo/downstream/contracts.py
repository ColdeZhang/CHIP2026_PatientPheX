"""Synthetic strict parsing, vote replacement, byte preservation and full80 checks."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from codec import parse_complete
from common import ref
from quality import assemble,atoms,current_q,pairs,select
from literal_helpers import nonrelation_bytes


def main(output):
    started=time.monotonic();count=0
    def check(value):
        nonlocal count
        assert value;count+=1
    row={'code_map':{'C000':'HP:0001000','C001':'HP:0001001'},'patient_ids':['A','B']}
    check(parse_complete('{"C000":["A","B"],"C001":[]}',row)=={'C000':['A','B'],'C001':[]})
    for bad in ['[]','{"C000":[]}','{"C000":[],"C001":[],"X":[]}',
                '{"C000":["X"],"C001":[]}','{"C000":[1],"C001":[]}',
                '{"C000":["A","A"],"C001":[]}','{"C000":null,"C001":[]}',
                '{"C000":[],"C000":["A"],"C001":[]}']:
        try:parse_complete(bad,row)
        except (ValueError,TypeError):count+=1
        else:raise AssertionError('Malformed JSON was accepted')
    protected={('d','A','HP:0000001')};one=('d','A','HP:0000002');two=('d','B','HP:0000003')
    old={'G':{one,two},'S':set(),'Q':{one},'C':{two}}
    check(select(protected,old)==protected|{one,two})
    new={**old,'Q':set()};check(select(protected,new)==protected|{two})
    check(select(protected,{'G':{one},'S':set(),'Q':set(),'C':set()})==protected)
    line=' { "pmc_id":"d", "entities":[], "patient":[{"patient_id":"A"},{"patient_id":"B"}], "association":[{"patient_id":"A","phenotype":["HP:0000001;HP:0000004","exact NIL"]},{"patient_id":"B","phenotype":[]}], "extra":"keep" }\n'
    base=pairs([json.loads(line)]);desired=base|{one,two};changed=assemble([line],desired)
    check(pairs([json.loads(changed[0])])==desired)
    check(nonrelation_bytes(changed)==nonrelation_bytes([line]))
    check(json.loads(changed[0])['association'][0]['phenotype'][:2]==['HP:0000001;HP:0000004','exact NIL'])
    check(atoms('HP:0000001;HP:0000002')==['HP:0000001','HP:0000002'])
    check(atoms('NIL;literal')==['NIL;literal'])
    with tempfile.TemporaryDirectory(prefix='dpo-downstream-contract-') as tmp:
        native=Path(tmp)
        for d in ['generated-ids','raw-responses','canonical-records']:(native/d).mkdir()
        prompts=[];encoded=[];raw=[];canonical=[];patients={str(i):{'A','B'} for i in range(16)}
        def persist(i,r,c,tokens):
            identity={k:r[k] for k in ['row_id','pmc_id','original_index','natural_eos','hit_token_limit']}
            (native/'generated-ids'/f'{i:03d}.json').write_text(json.dumps({**identity,'generated_ids':tokens,'configured_eos_ids':[2]}))
            (native/'raw-responses'/f'{i:03d}.json').write_text(json.dumps(r))
            (native/'canonical-records'/f'{i:03d}.json').write_text(json.dumps(c))
        for i in range(80):
            size=8 if i<27 else 7
            p={'row_id':str(i),'pmc_id':str(i//5),'patient_ids':['A','B'],'code_map':{f'C{j:03d}':f'HP:{1000+i*8+j:07d}' for j in range(size)}}
            value={k:(['A'] if k=='C000' else []) for k in p['code_map']}
            identity={'row_id':p['row_id'],'pmc_id':p['pmc_id'],'original_index':i,'natural_eos':True,'hit_token_limit':False}
            r={**identity,'provenance':'fresh','response':json.dumps(value),'input_tokens':3,'output_tokens':2}
            c={**identity,'provenance':'fresh','strict_complete':True,'abstention':None,'candidate_patients':value}
            e={**p,'input_ids':[1,3,4]}
            prompts.append(p);encoded.append(e);raw.append(r);canonical.append(c);persist(i,r,c,[7,2])
        got,strict,artifacts=current_q(prompts,encoded,raw,canonical,native,patients)
        check(len(got)==strict==80 and len(artifacts)==240)
        r=copy.deepcopy(raw[0]);c=copy.deepcopy(canonical[0]);r['response']='{}';c.update(strict_complete=False,candidate_patients={},abstention='ValueError: Expected exactly the supplied candidate keys');raw[0]=r;canonical[0]=c;persist(0,r,c,[7,2])
        got,strict,_=current_q(prompts,encoded,raw,canonical,native,patients);check(len(got)==strict==79)
        r=copy.deepcopy(raw[1]);c=copy.deepcopy(canonical[1]);r.update(natural_eos=False,hit_token_limit=True,output_tokens=1600);c.update(natural_eos=False,hit_token_limit=True,strict_complete=False,candidate_patients={},abstention='ValueError: No natural EOS or generation reached cap');raw[1]=r;canonical[1]=c;persist(1,r,c,[7]*1600)
        got,strict,_=current_q(prompts,encoded,raw,canonical,native,patients);check(len(got)==strict==78)
        bad=copy.deepcopy(raw);bad[2]['provenance']='cache'
        try:current_q(prompts,encoded,bad,canonical,native,patients)
        except AssertionError:count+=1
        else:raise AssertionError('Cached response accepted as fresh')
    old=Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations')
    def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in ['unique_object','parse_complete']:check(functions(Path(__file__).parent/'codec.py')[name]==functions(old/'native.py')[name])
    def inference_loop(p):
        main=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        return next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['_','_','index'])
    check(ast.dump(inference_loop(Path(__file__).parent/'infer.py'),include_attributes=False)==ast.dump(inference_loop(old/'f0_infer.py'),include_attributes=False))
    result={'status':'PASS_SYNTHETIC_DOWNSTREAM_CONTRACTS','checks':count,'seconds':time.monotonic()-started,
            'original_strict_codec_AST_exact':True,'original_f0_raw_first_generation_loop_AST_exact':True,
            'synthetic_full80_587CID_complete_partial_abstention_cap_cache_cases':True,
            'held_gold_reads':0,'remote_calls':0,'GPU_calls':0,'model_calls':0,'quality_calls':0,
            'sources':{p.name:ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
