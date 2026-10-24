#!/usr/bin/env python3
"""Read only verified train32 and public HPO; mechanical evidence coverage, not entailment."""
import argparse, collections, hashlib, importlib.util, json, re, time
from pathlib import Path

def ref(p):
 p=Path(p).resolve(); b=p.read_bytes(); return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def load(p): return json.loads(Path(p).read_text())
def put(p,x):
 with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def sentence_ranges(text):
 start=0
 for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9(])|\n+',text):
  if m.start()>start: yield start,m.start()
  start=m.end()
 if start<len(text): yield start,len(text)
def containing(doc,off,length):
 hits=[(i,p,off-p['offset']) for i,p in enumerate(doc['full_text']) if 0<=off-p['offset'] and off-p['offset']+length<=len(p['text'])]
 assert len(hits)==1
 return hits[0]
def gap(e,m): return max(0,e['offset']-m['offset']-m['length'],m['offset']-e['offset']-e['length'])
def main():
 a=argparse.ArgumentParser();a.add_argument('--old-plan',required=True);a.add_argument('--output',required=True);args=a.parse_args()
 start=time.monotonic(); plan=load(args.old_plan); refs=plan['references'];out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 # Deliberately never open the held_input reference, only exact train/HPO/provenance.
 for k in ('train','hpo','split_provenance'): assert ref(refs[k]['path'])==refs[k]
 src=plan['sources']['normalization/strict.py'];assert ref(src['path'])==src
 spec=importlib.util.spec_from_file_location('verified_rationale_hpo',src['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 hpo=mod.HPO(refs['hpo']['path']);docs=mod.read_jsonl(refs['train']['path']);prov=load(refs['split_provenance']['path'])
 ids={str(d['pmc_id']) for d in docs};assert len(ids)==32 and ids==set(prov['training_document_ids']) and not ids & set(prov['excluded_document_ids'])
 totals=collections.Counter(); records=[];invalid=[];schema=collections.Counter();types=collections.Counter()
 for d in docs:
  schema.update(d.keys());patients={p['patient_id']:p for p in d['patient']};totals['patients']+=len(patients)
  by=collections.defaultdict(list)
  for i,e in enumerate(d['entities']):
   types[e['type']]+=1
   try:
    assert e['type']=='Phenotype' and e['length']==len(e['text']);u,p,s=containing(d,e['offset'],e['length']);assert p['text'][s:s+e['length']]==e['text']
    keys=hpo.entity_keys(e['identifier'],e['text'])
   except (ValueError,AssertionError): continue
   for key in keys:by[key].append((i,e,u,p,s))
  for p in patients.values():
   for m in p['mention']:
    _,u,s=containing(d,m['offset'],m['length']);assert u['text'][s:s+m['length']]==m['text']
  for a in d['association']:
   pid=a['patient_id'];keys=set()
   for v in a['phenotype']:
    try:keys.update(hpo.relation_keys(v))
    except ValueError:invalid.append(dict(pmc_id=d['pmc_id'],patient_id=pid,label=v))
   for key in sorted(keys):
    choices=[]
    for ei,e,ui,u,s in by.get(key,[]):
     ranges=[(lo,hi) for lo,hi in sentence_ranges(u['text']) if lo<=s and s+e['length']<=hi]
     # A span crossing the heuristic sentence boundary has no sentence evidence.
     sentence=ranges[0] if len(ranges)==1 else None
     spids=set();upids=set()
     for other in patients.values():
      for m in other['mention']:
       z=m['offset']-u['offset']
       if 0<=z and z+m['length']<=len(u['text']):
        upids.add(other['patient_id'])
        if sentence and sentence[0]<=z and z+m['length']<=sentence[1]:spids.add(other['patient_id'])
     own=min(gap(e,m) for m in patients[pid]['mention']);foreign=[min(gap(e,m) for m in q['mention']) for q in patients.values() if q['patient_id']!=pid]
     choices.append(dict(entity_index=ei,offset=e['offset'],length=e['length'],unit=ui,sentence=sentence,
       sentence_patient_ids=sorted(spids),unit_patient_ids=sorted(upids),distance_target=own,
       nearer_foreign=bool(foreign and min(foreign)<own),text_exact=True))
    rec=dict(pmc_id=str(d['pmc_id']),patient_id=pid,key=list(key),patients=len(patients),choices=choices,
      any_same_sentence_target=any(pid in c['sentence_patient_ids'] for c in choices),
      any_target_only_sentence=any(c['sentence_patient_ids']==[pid] for c in choices),
      any_same_unit_target=any(pid in c['unit_patient_ids'] for c in choices),
      all_mentions_nearer_foreign=bool(choices) and all(c['nearer_foreign'] for c in choices))
    records.append(rec);totals['atomic_positive_relations']+=1
    totals['locatable']+=bool(choices);totals['multiple_mentions']+=len(choices)>1
    for k in ('any_same_sentence_target','any_target_only_sentence','any_same_unit_target','all_mentions_nearer_foreign'):totals[k]+=rec[k]
    if len(patients)>1:totals['multi_patient_positive_relations']+=1
 # Evidence supply harder than G's nearest choice: every lexical occurrence closer to another patient.
 hard=[r for r in records if r['all_mentions_nearer_foreign']]
 per_doc=collections.defaultdict(list)
 for r in hard:per_doc[r['pmc_id']].append(r)
 selected=[]
 for pmc in sorted(per_doc):
  selected.append(min(per_doc[pmc],key=lambda r:hashlib.sha256(json.dumps([r['pmc_id'],r['patient_id'],r['key'],20261024]).encode()).hexdigest()))
 put(out/'train-evidence-inventory.json',records);put(out/'hard-probe-selection.json',selected);put(out/'invalid-training-labels.json',invalid)
 result=dict(status='TRAIN_ONLY_MECHANICAL_EVIDENCE_CENSUS_NOT_SEMANTIC_VALIDATION',seconds=time.monotonic()-start,source=ref(__file__),
  references={k:refs[k] for k in ('train','hpo','split_provenance')},normalizer_source=src,documents=len(ids),counts=dict(totals),
  document_fields=dict(schema),entity_types=dict(types),gold_relation_schema=list(docs[0]['association'][0]),
  hard_documents=len(per_doc),proposed_probe_rows=len(selected),invalid_relation_labels=len(invalid),
  sentence_rule=r'(?<=[.!?])\s+(?=[A-Z0-9(])|\n+',sentence_is_heuristic_not_annotation=True,
  relation_specific_human_evidence_field_present=False,selection_is_train_label_dependent=True,
  artifacts={p.name:ref(p) for p in out.iterdir()},held_gold_reads=0,quality_calls=0,model_calls=0,fit_calls=0)
 put(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
