"""Reconstruct the existing B4 formula from frozen pure streams, no gold/model."""
import argparse
import collections
import copy
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        if '.gold.jsonl' in os.fsdecode(args[0]):
            raise PermissionError('Held gold is outside this read-only formula check')


def ref(path):
    p=Path(path).resolve();b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def checked(r):
    assert ref(r['path'])==r
    return Path(r['path'])


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def atoms(x):return x.split(';') if re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+',x) else [x]
def pairs(ds):return {(str(d['pmc_id']),a['patient_id'],v) for d in ds for a in d['association'] for x in a['phenotype'] for v in atoms(x)}


def main(args):
    sys.addaudithook(guard)
    p=json.loads(args.source_plan.read_text())
    b=rows(args.protected_base);old=rows(checked(p['refs']['baseline']))
    original=pairs(b);target=pairs(old)
    assert len(original)==471 and len(target)==524
    patients={str(d['pmc_id']):{a['patient_id'] for a in d['association']} for d in b}
    pure={k:pairs(rows(checked(v))) for k,v in p['pure_sources'].items()}
    streams={'S1':set(),'S2':set()};seen={'S1':{},'S2':{}}
    for item in rows(checked(p['sampled_diagnostics'])):
        stream=item['stream']
        if stream=='G':continue
        assert stream in streams and item['row_id'] not in seen[stream]
        assert item['strict_complete'] and item['actual_natural_eos'] and item['fresh']
        assert not any(item[k] for k in ['actual_hit_token_limit','actual_has_think_marker','whole_row_abstention','parser_error'])
        seen[stream][item['row_id']]=str(item['pmc_id'])
        for doc,pid,value in item['relations']:
            assert str(doc) in patients and pid in patients[str(doc)]
            streams[stream].update((str(doc),pid,a) for a in atoms(value))
    assert seen['S1']==seen['S2'] and len(seen['S1'])==80
    pure['S']=streams['S1']&streams['S2']
    assert set(pure)==set('GSQC')
    counts=collections.Counter(k for stream in pure.values() for k in stream)
    expected=original|{k for k,v in counts.items() if v>=2}
    assert expected==target
    # Rebuild from protected B471 and compare semantic objects and prefixes.
    reconstructed=copy.deepcopy(b)
    for d in reconstructed:
        for a in d['association']:
            a['phenotype'].extend(sorted(v for doc,pid,v in expected-original if doc==str(d['pmc_id']) and pid==a['patient_id']))
    assert pairs(reconstructed)==target
    for bb,oo in zip(b,old):
        assert {k:v for k,v in bb.items() if k!='association'}=={k:v for k,v in oo.items() if k!='association'}
        for ba,oa in zip(bb['association'],oo['association']):
            assert {k:v for k,v in ba.items() if k!='phenotype'}=={k:v for k,v in oa.items() if k!='phenotype'}
            assert oa['phenotype'][:len(ba['phenotype'])]==ba['phenotype']
    payload=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in reconstructed).encode()
    result=dict(status='PASS_EXISTING_B4_EQUALS_B471_UNION_TWO_OF_FOUR_PURE_STREAMS',
                at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sources={k:ref(v['path']) for k,v in p['pure_sources'].items()},
                sampled_diagnostics=p['sampled_diagnostics'],protected_base=ref(args.protected_base),
                old_B4=p['refs']['baseline'],source_plan=ref(args.source_plan),source=ref(__file__),
                source_counts={k:len(v) for k,v in pure.items()},base_count=len(original),final_count=len(target),
                all_nonassociation_and_patient_objects_exact=True,protected_base_prefix_exact=True,
                rebuilt_relation_sets_exact=True,rebuilt_full_objects_equal=reconstructed==old,
                rebuilt_bytes_exact=payload==checked(p['refs']['baseline']).read_bytes(),
                serialization_policy='No new prediction saved. Future zero semantic change inherits exact existing B4 file; do not claim arbitrary reconstruction byte equivalence.',
                gold_reads=0,model_calls=0,fit_calls=0,validator_calls=0,evaluator_calls=0)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','sampled_diagnostics','protected_base','old_B4','source_plan','source']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-plan',required=True,type=Path);p.add_argument('--protected-base',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    main(p.parse_args())
