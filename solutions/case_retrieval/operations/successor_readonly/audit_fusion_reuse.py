"""Read existing split/model/candidate provenance only; no model fit or gold evaluation."""
from pathlib import Path
import json,hashlib
R=Path('/workspace/patientphex-runs/round2-20261023');Q=R/'case_retrieval/qwen35-preparation';O=R/'ontology_rules';S=R/'sequence_crf/relation-sft-fold0';D=Path('/workspace/patientphex-data/round2-inner4')
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def ids(p):return {str(json.loads(l)['pmc_id']) for l in p.read_text().splitlines()}
folds={i:ids(D/f'fold-{i}/validation16.input.jsonl') for i in range(4)}
assert all(len(v)==16 for v in folds.values()) and len(set.union(*folds.values()))==64
refs=[ref(D/f'fold-{i}/validation16.input.jsonl') for i in range(4)]
pools=[]
for i in [1,2,3]:
 train=O/f'ce-oof-control/generator{i}/train16.jsonl';assert ids(train)==folds[i];refs.append(ref(train))
 for j in [1,2,3]:
  if i==j:continue
  b=O/f'ce-oof-control/pool{i}-to{j}';m=json.loads((b/'manifest.json').read_text());assert ids(b/'baseline.jsonl')==folds[j] and m['inputs']['train']==ref(train)['sha256'];assert not folds[i]&folds[0] and not folds[i]&folds[j]
  merged=Q/f'strict-meta-calibration-run/pool{i}-to{j}/candidates.jsonl'
  pools.append({'generator_fold':i,'target_fold':j,'generator_docs':16,'target_docs':16,'outer0_excluded':True,'same_target_excluded':True,'semantic_pool_manifest':ref(b/'manifest.json'),'semantic_predictions':ref(b/'baseline.jsonl'),'assembled_CRF_plus_Sap_exists':merged.exists(),'assembled_CRF_plus_Sap':ref(merged) if merged.exists() else None})
held=[]
for i in [1,2,3]:
 train=O/f'nested-outer0-inner{i}/train32.jsonl';assert ids(train)==set.union(*(folds[j] for j in [1,2,3] if j!=i));p=S/f'inner{i}/prompts.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];assert {str(r['pmc_id']) for r in rows}==folds[i] and all('answer' not in r for r in rows)
 held.append({'heldout_inner':i,'train32':ref(train),'prompts':ref(p),'candidate_manifest':ref(S/f'inner{i}/candidates.manifest.json'),'prompts_count':len(rows),'candidate_patient_pairs':sum(len(r['code_map'])*len(r['patient_ids']) for r in rows)})
models=[]
for p in sorted(Q.rglob('checkpoint.json')):
 if p.parent.name=='epoch-002':
  d=json.loads(p.read_text());fp=d.get('fingerprint',{});models.append({'checkpoint':ref(p),'training_docs':len(fp.get('training_document_ids',[])),'trained_inner1':bool(set(fp.get('training_document_ids',[]))&folds[1]),'excluded_outer0':not bool(set(fp.get('training_document_ids',[]))&folds[0])})
base=Q/'llm-tree-base204/base204.responses.manifest.json';m=json.loads(base.read_text());assert m['fingerprint']['configuration']['adapter'] is None and m['base_parameters']==m['total_parameters']==8953803264
train=Q/'relation-training-preparation/formal2ep/training_complete.json';seconds=json.loads(train.read_text())['elapsed_seconds']
d={'status':'read_only_reuse_inventory_no_training_or_new_quality','scope':'Only existing train-side candidate documents/manifests and completed checkpoint metadata; outer0 input-onlyIDs. No outer/originaldev gold.','base204_was_public_base_not_SFT':True,'base204_manifest':ref(base),'candidate_pools':pools,'existing_inner_holdout_inputs':held,'existing_Qwen35_epoch002_checkpoints':models,'suitable_inner_train32_SFT_checkpoints_found':0,'full_three_inner_crossfit_missing_SFT_models':3,'remaining_candidate_work':'Six16-to16 semantic pools exist. Two CRF+Sap assembled directions2to3/3to2 exist. Four other directions need CRF predictions/assembly; generator2/3 CRF models can potentially be reused and generator1 requires CPU fit; exact run contract still needed.','smaller_single_meta_fold':'Train one freshSFT32 on2+3 using already assembled2to3/3to2 pools; predict existing80 F1 prompts. A fixed regularized fusion can train only onF1 (16 docs), then use existing SFT48 outer0 outputs plus independently fit LR48. This is single-meta-fold, not full48 OOF or unbiased meta-validation.','LR_requirement':'Any learned LR feature must also be trained without scored innerfold and outer0; reuse feature source and keys, but raw cached features alone do not establish out-of-fold LR probability. Freeze meta model/threshold, no outergold tuning.','cost_reference':{'old48_two_epoch_seconds':seconds,'old48_docs':48,'simple32_doc_scaled_seconds_per_model':seconds*32/48,'three_models_scaled_seconds':seconds*2,'warning':'Document-linear proxy only; new training examples/tokens not built. Not actual32 runtime or a GPU budget. Existing total204 base inference759.096s is another proxy, not measuredSFT32 inference.'},'references':refs+[ref(train),ref(Path(__file__))]}
p=Q/'successor-readonly-review/fusion-reuse-inventory.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print('semantic pools',len(pools),'assembled',sum(x['assembled_CRF_plus_Sap_exists'] for x in pools),'heldout',[(x['heldout_inner'],x['prompts_count'],x['candidate_patient_pairs']) for x in held],'models',[(x['training_docs'],x['excluded_outer0']) for x in models])
