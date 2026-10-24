"""Freeze source/data identities only; never parse candidate rows, gold, or count spaces."""
from pathlib import Path
import hashlib,json,datetime
R=Path('/workspace/patientphex-runs/round2-20261023');S=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion';D=R/'research/relation-candidate-space-followup';W=Path(__file__).parent

def ref(p):
 p=Path(p).resolve();assert '.gold.' not in str(p) and p.name not in ('train64.jsonl','train48.jsonl','train32.jsonl','missing-candidate-keys.json'),str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(Path(p).read_text())
def checked(r):
 actual=ref(r['path']);assert actual['sha256']==r['sha256']
 if 'bytes' in r:assert actual['bytes']==r['bytes']
 return actual

lgbroot=R/'research/lightgbm-selector-readiness/formal-control';lgbfreeze=read(lgbroot/'complete-freeze.json');prep=read(R/'ontology_rules/public-biobert-selector-control/prepared/manifest.json')
folds=[]
for fold,name,sha_prefix in [(0,'public-new-relations-outer0','50bea617'),(1,'public-new-relations','f0088357')]:
 path=S/name/'run-plan.json';plan=read(path);rp=ref(path);assert rp['sha256'].startswith(sha_prefix)
 r=plan['references'];record=dict(fold=fold,public_run_plan=rp)
 for key in ['input','old_entities','old_prompts','prompts','input_only_difference_freeze','new_entities_evidence']:record[key]=checked(r[key])
 freeze=read(r['input_only_difference_freeze']['path']);record['public_difference_concepts']=checked(freeze['artifacts']['union']['concepts']);record['public_preparation_source']=checked(freeze['sources']['source'])
 assert record['prompts']['sha256']==freeze['artifacts']['union']['prompts']['sha256']
 assert record['old_entities']['sha256']==freeze['sources']['old_candidates']['sha256']
 assert record['input']['sha256']==freeze['sources']['input']['sha256']
 for key in ['pubmed_manifest','pubmed_canonical','bioformer_manifest','bioformer_canonical']:record[key]=checked(freeze['sources'][key])
 lg=lgbfreeze['folds'][fold];record['Task1_selected_entities_file']=checked(lg['new_prediction']);record['Task1_scores_source_flags']=checked(lg['scores']);record['Task1_fitted_manifest']=checked(lg['manifest']);record['Task1_input_pool']=checked(prep['folds'][fold]['input'])
 record['read_contract']='For Task1 read only entities. Existing association is copied old baseline, not the current relationship prediction. Legal relationship space comes from old+public prompt code_map and patient_ids.'
 record['current_relation_prediction_for_FN']=None
 record['gold_unread']=dict(path=plan['gold_path'],historical_sha256_from_existing_plan=plan.get('historical_gold_sha256',plan.get('historical_F1_gold_sha256')),actual_bytes_not_read_or_hashed=True)
 folds.append(record)
common={
 'HPO':ref('/workspace/patientphex-data/hp.obo'),
 'Task1_full_freeze':ref(lgbroot/'complete-freeze.json'),
 'Task1_prepared_manifest':ref(R/'ontology_rules/public-biobert-selector-control/prepared/manifest.json'),
 'Task1_prepare_source':ref('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/prepare_pool.py'),
 'Task1_decode_source':ref('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/lightgbm_selector_control/execute_control.py'),
 'old_prompt_builder':ref(S/'prepare_pair_rag.py'),
 'old_association_keys_and_HPO_parser':ref(S/'run.py'),
 'evaluator_v2_normalization_only':ref('/workspace/CHIP2026_PatientPheX/scripts/evaluate.py'),
 'F1_prior_missing_diagnostic_source_only':ref('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/operations/sft_meta1/diagnose_training_errors.py'),
 'prior_public_LGB_support_control_source':ref('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/public_new_lgb_support/control.py'),
 'prior_public_LGB_support_control_plan':ref('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/public_new_lgb_support/plan.json')}
for kind,folder,terminal in [('pubmed','public-phenotagger-train64-cache','native-train64-cache'),('bioformer','public-phenotagger-bioformer-train64-cache','native-train64-final-cache'),('biobert','public-phenotagger-biobert-train64-cache','native-train64-final-cache')]:
 c=R/'research'/folder/'terminal'/terminal;m=read(c/'manifest.json');assert m['fixed_policy']['threshold_strictly_greater_than']==.95 and m['public_model_training_on_task'] is False
 common[kind+'_cache_manifest']=ref(c/'manifest.json')
 for key,filename in [('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]:
  item=ref(c/filename);assert item['sha256']==m['outputs'][filename]['sha256'];common[kind+'_'+key]=item
 prior=json.loads((W/'count-definitions.json').read_text())
result=dict(status='INPUT_SOURCE_IDENTITIES_AND_NORMALIZATION_INVENTORY_ONLY_NO_COUNTS',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),definition=ref(W/'count-definitions.json'),folds=folds,common=common,history_search=dict(scopes=['F1 public-candidate-coverage source/input-only-freeze','F0 public-new-relations-outer0/candidate-difference source/input-only-freeze','fixed LightGBM formal-control source/complete-freeze','public_new_lgb_support source/plan','route markdown and report filenames for candidate-space/coverage'],exact_same_control_found=False,qualified_conclusion='No identical LGB-selected canonical concepts versus old+public2 actual prompt space attribution/FN audit found in these searched sources; not a claim of exhaustive global absence.',closest_prior=[dict(name='F1 old public2 candidate coverage',source=folds[1]['public_preparation_source'],difference='Uses original oldpool and public2 union/single-source; includes existing missing-gold oracle later in source. No current LGB-selected concepts or BioBERT source attribution. Existing report body and missing keys were not read.'),dict(name='F0 public2 input difference',source=folds[0]['public_preparation_source'],difference='Only oldpool versus public2 native canonical input difference; no LGB/BioBERT-selected missing-space audit.'),dict(name='two-fold public-new LGB support replacement',source=common['prior_public_LGB_support_control_source'],difference='Filters already generated public-new positive relationships by LGB selected doc-HPO support; cannot test never-prompted concepts or create their outcomes.')]),findings=['Native>.95 is a frozen neural decoder policy, not a second filter on canonical hybrid rows. Both F0/F1 public sources select all canonical rows.','Task1 selected file association must not be used as current relationship FN baseline.','Candidate-origin flags at exact mention key and document-HPO presence are different levels; report both, do not call every BioBERT-bearing row BioBERT-only.','Raw prompt identifiers and posthoc canonical HPO comparison are distinct; verify typed oldpool↔prompt equivalence before coverage.'],blocked_dependent_work=['Gold/FN counting awaits root-bound current full relation prediction refs for both folds.'],no_candidate_or_prompt_rows_parsed=True,no_new_space_counts=True,no_gold_read=True,no_NN_fit_predict_or_quality=True,no_remote_action=True)
for path in [D/'input-source-inventory.json',W/'input-source-inventory.json']:
 with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(output=ref(D/'input-source-inventory.json'),folds=[dict(fold=x['fold'],Task1=x['Task1_selected_entities_file']['sha256'],old=x['old_prompts']['sha256'],public=x['prompts']['sha256']) for x in folds])))
