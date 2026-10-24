"""Exactly one full B4 union adaptedG intersection S/Q/C; no old-G support."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from common import checked, deny_gold, module, pairs, primary, publish, read, ref, save

S=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--normalization',required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args()
    deny_gold();e=read(S/'evidence/preparation.json');p=read(checked(e['preparation']));nm=read(a.normalization)
    assert nm['schema']=='adapted-ordinary-G-normalization-complete-v1' and nm['source']==ref(S/'finalize.py')
    assert nm['preparation']==e['preparation'] and nm['original_strict_decisions']==p['strict_decisions']
    assert nm['original_extraction']==p['original_G']['extraction_manifest'] and nm['policy']==ref(S/'policy_v1.json')
    assert nm['documents']==16 and nm['patients']==39 and nm['anchors']==573 and nm['necessary_queries']==254
    assert nm['no_threshold_changes'] and nm['no_quality'] and nm['original_itemwise_parser_unchanged']
    normalized=[json.loads(x) for x in checked(nm['normalizations']).read_text().splitlines()]
    quotes=[json.loads(x) for x in checked(p['original_G']['quotes']).read_text().splitlines()]
    assert len(normalized)==len(quotes)==573
    assert all(all(row[k]==v for k,v in q.items()) for row,q in zip(normalized,quotes))
    assert all(row['normalization_policy']=='grounded-adapted-sap-conservative-v1' for row in normalized)
    sys.path.insert(0,str(Path(p['original_support']['path']).parent))
    support=module('support',p['original_support']);assembler=module('_adapted_original_relation_assembler',p['original_relation_assembler'])
    strict=module('_adapted_final_strict',p['sources']['strict.py']);hpo=strict.HPO(checked(p['hpo']))
    schema=module('_adapted_grounded_schema',p['grounded_schema'])
    inputs=[json.loads(x) for x in checked(p['input']).read_text().splitlines()]
    class BoundStrict:
        HPO=hpo
        verify_span=staticmethod(strict.verify_span)
    selected,_,counts=assembler.new_relations(normalized,inputs,BoundStrict,schema,nm['policy']['sha256'])
    new={(did,pid,key[1]) for (did,pid),keys in selected.items() for key in keys}
    sys.path.insert(0,str(Path(p['evaluator']['path']).parent))
    ev=module('_adapted_final_evaluator_atoms',p['evaluator']);validator=module('_adapted_final_validator',p['validator'])
    refs=p['support']['refs'];streams={k:[json.loads(x) for x in checked(v).read_text().splitlines()] for k,v in refs.items()}
    sets={k:pairs(v,ev) for k,v in streams.items()};base=sets['B4'];oldlines,base_rows,ids=support.rows(refs['B4']['path'])
    assert len(base)==524 and len(base_rows)==16 and sum(len(d['entities']) for d in base_rows)==1189
    assert sum(len(d['association']) for d in base_rows)==39 and ids==[str(d['pmc_id']) for d in inputs]
    assert sets['inherited_B']<=base and sets['inherited_B']<=sets['S']
    final=primary(base,new,sets['S'],sets['Q'],sets['C']);added=final-base
    assert final==primary(base,new,sets['S']-sets['inherited_B'],sets['Q'],sets['C'])
    spans,mask=support.byte_helpers(p['byte_helper']);rendered=[]
    for line,d in zip(oldlines,base_rows):
        replacement={row['patient_id']:row['phenotype']+sorted(h for did,pid,h in added if did==str(d['pmc_id']) and pid==row['patient_id']) for row in d['association']}
        rendered.append(support.replace_phenotypes(line,replacement,spans,mask))
    raw=''.join(rendered).encode();pred=[json.loads(x) for x in rendered]
    assert pairs(pred,ev)==final and all(mask(a)==mask(b) for a,b in zip(oldlines,rendered))
    for old,newdoc in zip(base_rows,pred):
        for before,after in zip(old['association'],newdoc['association']):assert after['phenotype'][:len(before['phenotype'])]==before['phenotype']
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    publish(out/'prediction.jsonl',raw)
    fmt=validator.validate_predictions(inputs,pred,validator.read_hpo(p['hpo']['path']));save(out/'format.json',fmt);assert fmt['valid']
    pure=copy.deepcopy(base_rows)
    for d in pure:
        for row in d['association']:row['phenotype']=sorted(h for did,pid,h in new if did==str(d['pmc_id']) and pid==row['patient_id'])
    publish(out/'adapted-G-unscored.jsonl',b''.join((json.dumps(d,ensure_ascii=False)+'\n').encode() for d in pure))
    save(out/'freeze.json',dict(schema='adapted-ordinary-G-single-primary-freeze-v1',at_unix=time.time(),source=ref(__file__),
        decision=ref(S/'decision.json'),preparation=e['preparation'],normalization=ref(a.normalization),
        prediction=ref(out/'prediction.jsonl'),pure_unscored=ref(out/'adapted-G-unscored.jsonl'),format=ref(out/'format.json'),
        baseline=refs['B4'],baseline_relations=524,documents=16,patients=39,entities=1189,new_G_relations=len(new),
        formula='B4 UNION (G_adapted_normalizer INTERSECT (S UNION Q UNION C))',old_G_in_support=False,
        support_refs={k:refs[k] for k in ['S','Q','C']},support_union=len(sets['S']|sets['Q']|sets['C']),added=len(added),added_pairs=sorted(added),
        removed=0,nonphenotype_bytes_exact=True,baseline_prefixes_exact=True,normalization_counts=counts,
        byteexact_to_baseline=raw==checked(refs['B4']).read_bytes(),no_gold_opened=True,new_v2=0,old_v2=0))
    print(json.dumps(ref(out/'freeze.json')))


if __name__=='__main__':main()
