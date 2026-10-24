"""Conditional deployment inputs/provenance only; no model, gold, or quality."""
from pathlib import Path
import json,hashlib,sys,argparse

def ref(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();R=Path(a.runs);D=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 pure=D/'day1/dev16.input.jsonl';docs=rows(pure);dev={str(d['pmc_id']) for d in docs};assert len(dev)==16 and all(not {'entities','association'}&d.keys() for d in docs)
 train=set().union(*[{str(d['pmc_id']) for d in rows(D/f'round2-inner4/fold-{i}/validation16.input.jsonl')} for i in range(4)]);assert len(train)==64 and not train&dev
 root=R/'case_retrieval/qwen35-preparation';prep_path=root/'full64-relation-cpu-preparation/preparation-manifest.json';prep=read(prep_path);ck_path=prep_path.parent/'formal2ep/epoch-002/checkpoint.json';ck=read(ck_path);fp=ck['fingerprint'];assert set(prep['training_document_ids'])==set(fp['training_document_ids'])==train and set(prep['excluded_document_ids'])==set(fp['excluded_document_ids'])==dev
 assert fp['full64_preparation_sha256']==ref(prep_path)['sha256'] and fp['train_sha256']==prep['training_sha256'];assert prep['fresh_base_required'] and prep['old_adapter_continuation_forbidden'] and prep['gold_added_candidates']==0 and prep['gold_in_prompts'] is False and ck['completed_epochs']==2
 for fold in prep['provenance']:
  tr=set(fold['train_ids']);held=set(fold['input_ids']);assert len(tr)==48 and len(held)==16 and not tr&held and (tr|held)==train and not (tr|held)&dev
 refs={'pure_input':ref(pure),'generation_preparation':ref(prep_path),'generation_checkpoint':ref(ck_path)}
 short=root/'dev-relation-transfer-preparation/prompts.jsonl';pm=read(short.with_name('prompts.manifest.json'));assert pm['files']['input']==ref(pure)['sha256'] and pm['files']['output']==ref(short)['sha256'];assert ref(pm['configuration']['candidates'])['sha256']==pm['files']['candidates'];refs['original_short_prompts']=ref(short);refs['short_prompt_manifest']=ref(short.with_name('prompts.manifest.json'))
 manifest_paths=['sequence_crf/gliner-dev-transfer/assembled/manifest.json','sequence_crf/gliner-dev-transfer/link-manifest.json','sequence_crf/crf-semantic-dev/lr.pred.manifest.json','ontology_rules/dev16-sapbert/manifest.json']
 for n in manifest_paths:
  p=R/n;m=read(p);assert m['inputs']['train']==prep['train64_sha256'] and m['inputs']['input']==ref(pure)['sha256'];refs[n]=ref(p)
  if 'training_document_ids' in m:assert set(m['training_document_ids'])==train
  if 'span_training_provenance' in m:assert set(m['span_training_provenance']['document_ids'])<=train and not set(m['span_training_provenance']['document_ids'])&dev
 F=R/'research/qwen35-sft32-full-report-feasibility/f0-preparation/input-cpu';sys.path.insert(0,str(F));import construct
 original=rows(short);expanded,docaudit,rowaudit=construct.append_reports(original,docs,training=False);assert len(expanded)==64
 payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in expanded);(out/'expanded64.jsonl').write_text(payload)
 for k,p in [('constructor',F/'construct.py'),('legacy_constructor',F/'legacy_full_context.py')]:refs[k]=ref(p)
 result={'status':'CONDITIONAL_FULL64_ANCESTRY_INPUT_PREPARATION_NO_GPU_NO_GOLD','generation_train_ids':sorted(train),'excluded_originaldev_ids':sorted(dev),'all_task_fitted_ancestors_exclude_originaldev':True,'generation_adapter':{'weights':ck['files']['adapter/adapter_model.safetensors'],'config':ck['files']['adapter/adapter_config.json']},'rows':64,'documents':16,'patients':sum(len(d['patient']) for d in docs),'candidate_codes':sum(len(r['code_map']) for r in original),'candidate_patient_pairs':sum(len(r['code_map'])*len(r['patient_ids']) for r in original),'short_prompt_SHA':ref(short)['sha256'],'short_prompt_is_NOT_full_report':True,'expanded':ref(out/'expanded64.jsonl'),'full_report_append_all64_exact':True,'all_original_metadata_and_text_restored_exact':True,'document_geometry':docaudit,'row_geometry':rowaudit,'refs':refs,'conditions':'No originaldev GPU or quality authorized. Only consider two fixed seeds on this fulltext input if root retains F0 independent or fused route. Preserve B901 downstream in root fusion.','new_fit_calls':0,'GPU_calls':0,'gold_read':False,'quality_calls':0}
 (out/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['document_geometry','row_geometry','refs','generation_train_ids','excluded_originaldev_ids']}))
if __name__=='__main__':main()
