import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,re,hashlib,statistics
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,'solutions/case_retrieval')
from prepare_rag import mark_patients
from run import load_hpo
ROOT=Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/patient-set-feasibility')
TRAIN=Path('/workspace/patientphex-data/round2-inner4/fold-0/train48.jsonl')
HPO=Path('/workspace/patientphex-data/hp.obo')
SFT=Path('/workspace/patientphex-runs/round2-20261023/sequence_crf/relation-sft-fold0/train.jsonl')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
docs=read(TRAIN);terms,valid,canonical,_=load_hpo(HPO)
atom=lambda value:str(value).split(';') if re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})*',str(value)) else [str(value)]
SYSTEM='Read the entire biomedical report for the single TARGET patient. Return a JSON object with hpo_ids (distinct HPO identifiers) and unmapped (verbatim phenotype strings with no equivalent HPO concept). Include only observed findings belonging to TARGET; exclude negated findings, other patients, and general syndrome descriptions. The inline PATIENT markers identify supplied patients. Do not infer a symptom from a diagnosis. Return empty lists if no findings are supported. No explanation.'
hdoc=defaultdict(set);paragraphs=defaultdict(set);nonliteral=[]
counts=Counter();patient_rel=[];docchars=[];patient_ids=defaultdict(set);phrase_ids=Counter();doc_rows=[];token_rows=[];all_relations=set();entitymapped=set();sftcovered=set();references_with_gold=[]
for row in read(SFT):
 answer=json.loads(row['answer'])
 for code,pids in answer.items():
  for pid in pids:sftcovered.add((str(row['pmc_id']),str(pid),row['code_map'][code]))
for d in docs:
 for para in d['full_text']:
  normalized=' '.join(para['text'].lower().split())
  if len(normalized)>=150:paragraphs[normalized].add(str(d['pmc_id']))
 did=str(d['pmc_id']);counts['documents']+=1;counts['patients']+=len(d['patient']);counts['single_patient_documents']+=len(d['patient'])==1
 gold={str(p['patient_id']):set(h for h0 in p['phenotype'] for h in atom(h0)) for p in d['association']}
 assert set(gold)=={str(p['patient_id']) for p in d['patient']}
 positives=defaultdict(list)
 for e in d['entities']:
  if str(e.get('note','')).upper()=='NO':continue
  for hp in ([e['text']] if str(e['identifier'])=='-1' else str(e['identifier']).split(';')):positives[hp].append(e)
 mapping={str(p['patient_id']):f'P{i+1:03d}' for i,p in enumerate(d['patient'])}
 stripped={k:v for k,v in d.items() if k not in ['entities','association']}
 marked=dict(stripped,patient=[dict(p,patient_id=mapping[str(p['patient_id'])]) for p in stripped['patient']])
 text='\n\n'.join(f"SECTION {p.get('section_type','')} / {p.get('type','')}; OFFSET {p['offset']}\n"+mark_patients(p['text'],int(p['offset']),marked) for p in marked['full_text'])
 chars=sum(len(p['text']) for p in d['full_text']);docchars.append(chars)
 for p in d['patient']:
  pid=str(p['patient_id']);ids=gold[pid];patient_rel.append(len(ids));counts['empty_patients']+=not ids;counts['patient_mentions']+=len(p['mention']);patient_ids[pid].add(did)
  hp=[];unmapped=[]
  for concept in sorted(ids):
   all_relations.add((did,pid,concept));phrase_ids[concept]+=1;hdoc[concept].add(did)
   counts['atomic_relations']+=1
   is_hpo=bool(re.fullmatch(r'HP:[0-9]{7}',concept));counts['hpo_relations' if is_hpo else 'text_relations']+=1
   if is_hpo:
    hp.append(concept);counts['hpo_outside_valid_branch']+=concept not in canonical
   else:
    unmapped.append(concept);literal=any(concept in para['text'] for para in d['full_text']);counts['text_relation_not_literal_in_fulltext']+=not literal
    if not literal:nonliteral.append(concept)
   if concept in positives:entitymapped.add((did,pid,concept))
  messages=[{'role':'system','content':SYSTEM},{'role':'user','content':'TARGET: '+mapping[pid]+'\nSUPPLIED PATIENTS: '+', '.join(mapping.values())+'\n\nFULL REPORT:\n'+text}]
  # Budget probe only. No predictions or training run are produced.
  token_rows.append({'row_id':did+':patient-set-budget:'+mapping[pid],'pmc_id':did,'patient_id':pid,'messages':messages,'answer':json.dumps({'hpo_ids':hp,'unmapped':unmapped},separators=(',',':'))})
 doc_rows.append({'pmc_id':did,'characters':chars,'patients':len(d['patient']),'relations':sum(map(len,gold.values()))})
report={'scope':'train48-only aggregate feasibility; no validation/dev gold; no training or prediction','counts':dict(counts),'relation_set_size':len(all_relations),'unique_relation_items':len(phrase_ids),'unique_hpo_items':sum(bool(re.fullmatch(r'HP:[0-9]{7}',x)) for x in phrase_ids),'relations_with_any_positive_gold_entity_same_document':len(entitymapped),'relations_without_positive_gold_entity_same_document':len(all_relations-entitymapped),'entity_match_policy':'Existing sequence association_keys: NIL by text; other entity IDs split by semicolon. This leaves the malformed compound training relation unresolved.', 'oof_candidate_relationships':len(all_relations&sftcovered),'oof_candidate_missing_relationships':len(all_relations-sftcovered),'missed_but_positive_entity_in_document':len((all_relations-sftcovered)&entitymapped),'document_chars':{'min':min(docchars),'median':statistics.median(docchars),'max':max(docchars),'sum':sum(docchars)},'patient_relation_counts':{'min':min(patient_rel),'median':statistics.median(patient_rel),'max':max(patient_rel),'mean':statistics.mean(patient_rel)},'patient_id_strings_reused_across_documents':sum(len(v)>1 for v in patient_ids.values()),'patients_affected_by_reused_patient_id_strings':sum(len(v) for v in patient_ids.values() if len(v)>1),'document_rows':doc_rows,'files':{'train':sha(TRAIN),'hpo':sha(HPO),'sft':sha(SFT),'source':sha(__file__)},'positive_entity_coverage_is_only_a_proxy':'Matching concept somewhere in document is not proof the local occurrence belongs to target patient; no gold chosen evidence or local target labels are generated.'}
report['rare_item_counts']={'items_occurring_in_one_patient':sum(v==1 for v in phrase_ids.values()),'items_occurring_in_one_document':sum(len(v)==1 for v in hdoc.values())}
report['duplicate_input_screen']={'policy':'normalized whole passages >=150 chars, train48 only','cross_document_exact_duplicate_passages':sum(len(v)>1 for v in paragraphs.values()),'limitation':'Not evidence of absence of same real person across publications; no global patient identifier exists.'}
report['train_edge_cases']={'out_of_branch_hpo_relations':counts['hpo_outside_valid_branch'],'malformed_hpo_compound_relations':sum(v.startswith('HP:') for v in nonliteral),'genuine_non_hpo_literal_relations':counts['text_relations']-len(nonliteral),'nonliteral_values':nonliteral,'policy':'No guessing missing digits; no automatic gold repair in this study.'}
(ROOT/'train-feasibility.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(ROOT/'token-budget-probes.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in token_rows))
print(json.dumps({k:v for k,v in report.items() if k!='document_rows'},ensure_ascii=False))
