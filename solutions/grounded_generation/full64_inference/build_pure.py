"""Whole36 normalized output -> pure relation supply, no quality or normalization."""
import argparse,json
from pathlib import Path
from actual import H,verify_actual,new_relations

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=H.read(a.plan);gate=H.gold_guard(p)
 assert p['schema']=='grounded-dev36-complete-pure-relation-supply-v1' and p['source']==H.ref(__file__)
 for item in list(p['references'].values())+list(p['actual'].values()):H.checked(item)
 r=p['references'];lines,base,ids=H.rows(r['baseline']['path']);_,inputs,iids=H.rows(r['dev_input']['path'])
 assert ids==iids and len(ids)==16 and sum(len(d['entities']) for d in base)==1061
 assert sum(len(d['association']) for d in base)==sum(len(d['patient']) for d in inputs)==36
 strict=H.load_module('_grounded_F0_strict',r['strict']);hpo=strict.HPO(r['hpo']['path']);schema=H.load_module('_grounded_F0_schema',r['schema'])
 normalized,coverage=verify_actual(p,inputs)
 class BoundStrict:
  HPO=hpo
  verify_span=staticmethod(strict.verify_span)
 selected,decisions,counts=new_relations(normalized,inputs,BoundStrict,schema,r['normalization_policy']['sha256'])
 spans,mask=H.byte_helpers(r['byte_helper']);rendered=[]
 for line,doc,inp in zip(lines,base,inputs):
  assert [x['patient_id'] for x in doc['association']]==[x['patient_id'] for x in inp['patient']]
  values={a['patient_id']:H.canonical_values([key[1] for key in sorted(selected[(str(doc['pmc_id']),a['patient_id'])])],hpo) for a in doc['association']}
  updated=H.replace_phenotypes(line,values,spans,mask);assert mask(updated)==mask(line);rendered.append(updated)
 out=Path(p['output_directory']);out.mkdir(parents=True,exist_ok=False)
 H.publish(out/'new-only.jsonl',''.join(rendered).encode());H.save(out/'consumption.json',dict(counts=counts,coverage=coverage,records=decisions))
 validator=H.load_module('validate_predictions',r['validator']);pred=[json.loads(x) for x in rendered];v=validator.validate_predictions(inputs,pred,validator.read_hpo(r['hpo']['path']));H.save(out/'validation.json',v);assert v['valid']
 keys=sorted((did,pid,*key) for (did,pid),items in selected.items() for key in items);H.save(out/'pure-relation-keys.json',keys)
 assert not gate['enabled'];H.save(out/'freeze.json',dict(plan=H.ref(a.plan),source=H.ref(__file__),predictions={'new-only':H.ref(out/'new-only.jsonl')},pure_keys=H.ref(out/'pure-relation-keys.json'),consumption=H.ref(out/'consumption.json'),validation=H.ref(out/'validation.json'),relation_count=len(keys),all36_complete=True,documents=16,patients=36,entities=1061,nonrelation_bytes_exact=True,no_gold_no_score=True,baseline_relationships_excluded=True,model_decode_parser_normalizer_reexecuted=False))
 print(json.dumps(H.ref(out/'freeze.json')))
if __name__=='__main__':main()
