from pathlib import Path
import hashlib,json,time
from datetime import datetime,timezone
R=Path('/workspace/patientphex-runs/round2-20261023');B=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion';O=R/'ontology_rules';D=Path('/workspace/patientphex-data/round2-inner4');Q=R/'case_retrieval/qwen35-preparation';OUT=R/'research/binary-complement-design'
refs={}
def read(path,name):
 b=path.read_bytes();refs[name]={'path':str(path),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return json.loads(b)
def ids(path,name):
 b=path.read_bytes();refs[name]={'path':str(path),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 rows=[json.loads(l) for l in b.splitlines() if l];assert all('entities' not in x and 'association' not in x for x in rows);z={str(x['pmc_id']) for x in rows};assert len(z)==len(rows)==16;return z
start=time.monotonic();folds=[ids(D/f'fold-{i}/validation16.input.jsonl',f'F{i}_input_only') for i in range(4)]
assert len(set.union(*folds))==64
prep=read(B/'preparation-manifest.json','SFT32_training_preparation');assert set(prep['training_document_ids'])==folds[2]|folds[3];assert set(prep['excluded_document_ids'])==folds[0]|folds[1]
for x in prep['provenance']:
 assert set(x['generator_ids'])==folds[x['generator_fold']]
 assert set(x['target_ids'])==folds[x['target_fold']]
 assert not (set(x['generator_ids'])|set(x['target_ids'])) & (folds[0]|folds[1])
 assert x['generator_fold'] in (2,3) and x['target_fold'] in (2,3) and x['generator_fold']!=x['target_fold']
cm=read(R/'sequence_crf/relation-sft-fold0/inner1/candidates.manifest.json','F1_entity_assembly');sem=read(O/'nested-outer0-inner1/manifest.json','F1_Sap_generator');crf=read(O/'nested-crf/inner1/work/pred-manifest.json','F1_CRF_generator');pm=read(R/'sequence_crf/relation-sft-fold0/inner1/prompts.manifest.json','F1_prompts')
train32sha=prep['sources']['train32']['sha256'];assert cm['inputs']['train']==sem['inputs']['train']==crf['train_sha256']==train32sha
assert cm['inputs']['input']==sem['inputs']['input']==crf['input_sha256']==pm['files']['input']==refs['F1_input_only']['sha256']
assert cm['inputs']['candidate_manifest']==refs['F1_Sap_generator']['sha256'];assert cm['inputs']['base']==crf['output_sha256'];assert cm['output_sha256']==pm['files']['candidates'];assert pm['documents']==16 and pm['prompts']==80
fusion=read(B/'fusion-plan.json','prior_SFT_LR_HGB_protocol');assert fusion['references']['meta_prompts']['sha256']==pm['files']['output'];assert fusion['only_difference']=='four original SFT facts';assert fusion['same_common_LR_features']
slot=read(B/'decision-target-context-design/fixed-slot-formal-v2/run-plan.json','binary32_training_protocol');assert 'F2+F3' in slot['protocol']['same_pairs'];assert 'no temperature/cut search' in slot['protocol']['threshold']
old48=read(Q/'relation-training-preparation/formal2ep/epoch-002/checkpoint.json','SFT48_checkpoint_fingerprint')['fingerprint'];assert set(old48['training_document_ids'])==folds[1]|folds[2]|folds[3];assert set(old48['excluded_document_ids'])==folds[0]
f0cm=read(R/'sequence_crf/crf-semantic-fold0/lr.pred.manifest.json','F0_entity_assembly');f0crf=read(R/'sequence_crf/baseline-fold0/pred-manifest.json','F0_CRF_generator');f0sem=read(O/'frozen-fold0-sapbert/manifest.json','F0_Sap_generator')
assert f0cm['inputs']['train']==f0crf['train_sha256']==f0sem['inputs']['train'];assert f0cm['inputs']['input']==f0crf['input_sha256']==f0sem['inputs']['input']==refs['F0_input_only']['sha256']
prior=read(B/'fusion-inputs/preparation.json','prior_LR_preparation_with_ID_lists');roles={x['role']:x for x in prior['roles']};assert set(roles['meta_F1']['train_ids'])==folds[2]|folds[3];assert set(roles['outer_F0']['train_ids'])==folds[1]|folds[2]|folds[3];assert f0cm['inputs']['train']==roles['outer_F0']['sources']['train']['sha256']
report=dict(status='PASS metadata/source isolation; no model or quality experiment',created_at=datetime.now(timezone.utc).isoformat(),checks={'F1_candidates_train32_exact_F2_F3':True,'F1_CRF_Sap_same_train32_SHA':True,'F1_80_prompts_bound_to_candidate_output':True,'both_relation_models_train32_excludes_F0_F1':True,'SFT32_training_generators_only_F2_to_F3_F3_to_F2':True,'F0_SFT48_train_exact_F1_F2_F3':True,'F0_CRF_Sap_train48_excludes_F0':True},counts={'F1_metatrain_documents':16,'F1_base_model_training_documents':32,'F1_candidate_generator_training_documents':32,'F0_old_SFT_training_documents':48,'F0_candidate_generator_training_documents':48},references=refs,limitations=['No F0/originaldev gold read, no 1557 pair probabilities read, no fit/inference/quality scoring.','Checks bind existing source manifests and recorded training-ID lists; no neural weights/old predictors reloaded.','F1 is a known development fold; future F1-only meta fitting can exclude F0 but F0 was previously researched and cannot be claimed blind.','SFT32-to-SFT48 and candidate generator32-to48 are transfer changes requiring disclosure.'],seconds=time.monotonic()-start)
report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (OUT/'source-check.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:report[k] for k in ['status','checks','counts','seconds','script_sha256']}))
