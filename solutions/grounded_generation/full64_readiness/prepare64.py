"""Conditional full64 CPU-only targets and pure dev36; exact existing target policy."""
import argparse,hashlib,json,os,signal,sys,time
from collections import Counter
from pathlib import Path
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S))
from prepare import construct,ref,put
from schema import INPUT_KEYS,contracts,messages,units
from normalization.strict import HPO,StrictNormalizer,read_jsonl

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.monotonic();p=json.loads(Path(a.plan).read_text())
 assert p['cpu_budget_seconds']==60 and not p['training_authorized'] and not p['quality_authorized']
 forbidden=set(map(os.path.realpath,p['forbidden_gold_paths']))
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0])) in forbidden:raise PermissionError('Original dev or held gold forbidden in conditional readiness')
 sys.addaudithook(audit);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU60')));signal.alarm(60)
 for group in ['sources','references']:
  for k,v in p[group].items():assert ref(v['path'])==v,k
 assert p['sources']['prepare64.py']==ref(__file__)
 r=p['references'];train=read_jsonl(r['train']['path']);held=read_jsonl(r['held_input']['path']);prov=json.loads(Path(r['split_provenance']['path']).read_text())
 tids=[str(x['pmc_id']) for x in train];hids=[str(x['pmc_id']) for x in held]
 assert len(tids)==len(set(tids))==64 and len(hids)==len(set(hids))==16 and not set(tids)&set(hids)
 assert sorted(tids)==prov['training_document_ids'] and sorted(hids)==prov['excluded_document_ids']
 assert r['train']['sha256']==prov['train64_sha256'] and r['held_input']['sha256']==prov['excluded_dev_input_sha256']
 assert all(set(x)==INPUT_KEYS for x in held)
 assert sum(len(x['patient']) for x in train)==173 and sum(len(x['patient']) for x in held)==36
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);put(out/'intent.json',dict(plan=ref(a.plan),conditional_only=True,no_held_or_original_dev_gold=True))
 hpo=HPO(r['hpo']['path']);rows=[];coverage=[];excluded=[]
 for doc in train:
  rr,cc,ee=construct(doc,hpo);rows.extend(rr);coverage.extend(cc);excluded.extend(dict(pmc_id=str(doc['pmc_id']),**e) for e in ee)
 heldrows=[dict(row_id=f"{d['pmc_id']}:{q['patient_id']}:grounded",pmc_id=str(d['pmc_id']),target_patient_id=q['patient_id'],messages=messages(d,q['patient_id']),units=units(d)) for d in held for q in d['patient']]
 assert len(coverage)==173 and len(heldrows)==36 and all('answer' not in x for x in heldrows)
 counts=Counter()
 for c in coverage:
  for name in ['original_relation_literals','valid_atomic_relation_keys','supported_atomic_keys','distinct_evidence_targets']:counts[name]+=c[name]
  for name in ['original_gold_empty','included_in_SFT','incomplete_positive_target']:counts[name]+=int(c[name])
  counts['unsupported_atomic_keys']+=len(c['unsupported']);counts['invalid_relation_labels']+=len(c['invalid_relation_labels'])
  for choice in c['selected_weak_evidence']:
   for name in ['nearer_other_patient','equally_near_other_patient']:counts[name]+=int(choice[name])
   counts['multi_candidate_relation_keys']+=int(choice['candidate_entities']>1)
   counts['chosen_entity_has_other_gold_atoms']+=bool(choice['chosen_entity_atoms_not_in_target_gold'])
 normalizer=StrictNormalizer(r['hpo']['path'],r['train']['path'],expected_train_sha256=r['train']['sha256']);normalizer.assert_disjoint_input(r['held_input']['path'])
 # Stable alias identity, no encoder/bank/model or query scoring.
 alias_table=[dict(text=text,options=[dict(ids=list(ids),evidence=evidence) for ids,evidence in sorted(options.items())]) for text,options in sorted(normalizer.exact.items())]
 alias_sha=hashlib.sha256(json.dumps(alias_table,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 norm=dict(normalizer.manifest,exact_alias_content_sha256=alias_sha,policy=r['normalizer_policy'],no_threshold_change=True,public41099_bank_reused_without_encoding=True,no_detector_or_encoder_fit=True)
 artifacts=dict(train=put(out/'train.jsonl',rows,True),held_input=put(out/'dev.input36.jsonl',heldrows,True),training_coverage=put(out/'train-coverage.json',coverage),invalid_training_entities=put(out/'invalid-training-entities.json',excluded),schema_contract=put(out/'schema-contract.json',contracts()),normalizer_train64=put(out/'normalizer-train64-manifest.json',norm))
 result=dict(status='COMPLETE_CPU_WEAK_TARGETS_NO_MODEL_NO_QUALITY',plan=ref(a.plan),sources=p['sources'],references=r,artifacts=artifacts,train_documents=64,all_train_patients_in_coverage=173,actual_train_rows=len(rows),held_documents=16,held_rows=36,train_ids=sorted(tids),held_ids=sorted(hids),counts=dict(counts),invalid_training_entities=len(excluded),same_construct_schema_and_quote_policy=True,targets_are_weak_patient_span_links=True,input_contains_no_gold_entities_HPO_candidates_or_relations=True,no_input_truncation=True,no_held_or_original_dev_gold_access=True,no_model_no_GPU_no_fit_no_quality=True,conditional_on_F0_gate_only=True,training_authorized=False,seconds=time.monotonic()-start)
 put(out/'manifest.json',result)
 parent=out.resolve()
 while True:
  fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if parent==parent.parent:break
  parent=parent.parent
 print(json.dumps(ref(out/'manifest.json')))
if __name__=='__main__':main()
