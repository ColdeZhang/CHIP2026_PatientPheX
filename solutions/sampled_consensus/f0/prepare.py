"""F0 no-gold provenance and exact reused-input construction contracts."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys

def ref(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--data',required=True);ap.add_argument('--lineage',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();R=Path(a.runs);D=Path(a.data);F=R/'research/qwen35-sft32-full-report-feasibility/f0-preparation';out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 refs={};folds={}
 for i in range(4):
  p=D/f'round2-inner4/fold-{i}/validation16.input.jsonl';ds=rows(p);assert all(not {'entities','association'}&d.keys() for d in ds);folds[i]={str(d['pmc_id']) for d in ds};assert len(folds[i])==16;refs[f'fold{i}_input']=ref(p)
 assert len(set.union(*folds.values()))==64
 prep_path=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion/preparation-manifest.json';ck_path=prep_path.parent/'formal2ep/epoch-002/checkpoint.json';prep=read(prep_path);ck=read(ck_path);fp=ck['fingerprint'];train32=folds[2]|folds[3];exclude=folds[0]|folds[1]
 assert set(prep['training_document_ids'])==set(fp['training_document_ids'])==train32
 assert set(prep['excluded_document_ids'])==set(fp['excluded_document_ids'])==exclude
 assert fp['meta32_preparation_sha256']==ref(prep_path)['sha256'] and fp['train_sha256']==prep['training_sha256']
 assert ck['completed_epochs']==2 and prep['fresh_base_required'] and prep['old_adapter_continuation_forbidden'] and prep['gold_added_candidates']==0 and prep['gold_in_prompts'] is False
 refs['generation_preparation']=ref(prep_path);refs['generation_checkpoint']=ref(ck_path)
 ancestors=[]
 for item in prep['provenance']:
  g=set(item['generator_ids']);t=set(item['target_ids']);assert g==folds[item['generator_fold']] and t==folds[item['target_fold']] and not g&t and not (g|t)&exclude
  ancestors.append({'generator_ids':sorted(g),'target_ids':sorted(t),'F0_intersection':[]})
  for key in ['assembly_manifest','semantic_manifest','CRF_manifest']:
   r=item[key];assert ref(r['path'])==r;refs[f'{item["generator_fold"]}-to-{item["target_fold"]}-{key}']=r
 lineage=read(a.lineage);fold0=next(f for f in lineage['folds'] if f['fold']==0);train48=set.union(folds[1],folds[2],folds[3]);assert set(fold0['train48_ids'])==train48 and set(fold0['held16_ids'])==folds[0];trainsha=fold0['train48_sha256'];refs['strong_B0_reviewed_lineage']=ref(a.lineage)
 for label,name in [('CRF','sequence_crf/baseline-fold0/pred-manifest.json'),('semantic','ontology_rules/frozen-fold0-sapbert/manifest.json'),('gate','sequence_crf/crf-semantic-fold0/lr.pred.manifest.json')]:
  p=R/name;m=read(p);actual=m.get('train_sha256',m.get('inputs',{}).get('train'));assert actual==trainsha;assert m.get('input_sha256',m.get('inputs',{}).get('input'))==refs['fold0_input']['sha256'];assert m['input_annotations_used'] is False;refs[label]=ref(p)
 short=R/'sequence_crf/qwen-pair-fold0/prompts.jsonl';short_manifest=short.with_name('prompts.manifest.json');m=read(short_manifest);refs['original_short_prompts']=ref(short);refs['original_prompt_manifest']=ref(short_manifest)
 assert m['files']['input']==refs['fold0_input']['sha256'] and m['files']['output']==ref(short)['sha256']
 candidates=ref(m['configuration']['candidates']);assert candidates['sha256']==m['files']['candidates']==read(R/'sequence_crf/crf-semantic-fold0/lr.pred.manifest.json')['output_sha256'];refs['original_prompt_candidates']=candidates
 # Reconstruct every original report without tokenizer, labels, or model loading.
 sys.path.insert(0,str(F/'input-cpu'));import construct
 old=rows(short);docs=rows(refs['fold0_input']['path']);extended,docaudit,rowaudit=construct.append_reports(old,docs,training=False);prompts=rows(F/'expanded58.jsonl');assert extended==prompts;assert len(prompts)==58
 audit=read(F/'full58-v1/row-audit.json');assert len(audit)==58
 for i,(row,t) in enumerate(zip(prompts,audit)):
  assert t['index']==i and row['row_id']==t['row_id'] and construct.jsha(row)==t['expanded_row_sha256'];assert row['patient_ids']==[str(p['patient_id']) for p in next(d for d in docs if str(d['pmc_id'])==str(row['pmc_id']))['patient']]
 assert sum(t['expanded']['tokens'] for t in audit)==505090 and max(t['expanded']['tokens'] for t in audit)==16245
 assert max(range(58),key=lambda i:audit[i]['expanded']['tokens'])==30
 for k,p in [('prompts',F/'expanded58.jsonl'),('audit',F/'full58-v1/row-audit.json'),('old_actual_token_contract',F/'input-cpu/actual-result.json'),('constructor',F/'input-cpu/construct.py'),('legacy_constructor',F/'input-cpu/legacy_full_context.py')]:refs[k]=ref(p)
 result={'status':'PASS_F0_ANCESTRY_AND_EXACT_FULL58_NO_GOLD_NO_MODEL','F0_ids':sorted(folds[0]),'generation_train32':sorted(train32),'generation_excluded32':sorted(exclude),'generation_training_candidate_ancestors':ancestors,'held_candidate_and_gate_train48':sorted(train48),'held_candidate_train_sha256':trainsha,'F0_intersection_all_task_fitted_ancestors':[],'generation_adapter_sha256':'4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff','same_SFT32_model_as_F1':True,'crossfold_disclosure':'F1 held candidate generators trained32; reused F0 held candidate generators trained48, all exclude F0. Same generation SFT32. This is a development crossfold recipe replication, not a pure equal-training-volume causal comparison.','rows':58,'documents':16,'patients':sum(len(d['patient']) for d in docs),'candidate_codes':sum(len(r['code_map']) for r in prompts),'input_tokens_per_pass':505090,'fresh_input_tokens_two_passes':1010180,'max_input_tokens':16245,'longest_original_index':30,'input_limit':16384,'all_fulltext_original_reconstruction_exact':True,'existing_native_token_audit_reused_no_new_tokenization':True,'gold_read':False,'refs':refs,'document_geometry':docaudit}
 (out/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['refs','document_geometry','generation_training_candidate_ancestors','generation_train32','generation_excluded32','held_candidate_and_gate_train48','F0_ids']}))
if __name__=='__main__':main()
