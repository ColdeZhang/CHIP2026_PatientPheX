"""Metadata-only binding of completed fixed B artifacts; never runs the pool or models."""
import argparse
import hashlib
import json
from pathlib import Path

B_INPUT='878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
ROOT_CONTRACT='2c1690660c58337235e9e11d96633109ac3ac3ffd62fa5e0600980c4b8a2ba21'
TRAIN64='cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2'
POOL_SOURCE='f80669fb7e75305462ae3a15f8153812c82c6ba4ebf5e97d49aa526e9cade39b'
WEIGHTS={'pubmed':'d58af55a68840a24b052a775cf0cb32b8993a1fa4823c76aad10d3e6299d7ec3','bioformer':'867819ac932e11250c42ca13c49eba3711a7f24b170e24bfb31b65589d0024a1','biobert':'1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def ref(path):
 p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(path):return json.loads(Path(path).read_text())
def checked(r):
 assert ref(r['path'])==r,r['path'];return Path(r['path'])
def hashes(value):
 """Inspect manifest identity links only, never candidate rows or training records."""
 if isinstance(value,str):return {value} if len(value)==64 and all(c in '0123456789abcdef' for c in value) else set()
 if isinstance(value,dict):return set().union(*(hashes(x) for x in value.values())) if value else set()
 if isinstance(value,list):return set().union(*(hashes(x) for x in value)) if value else set()
 return set()

def main():
 p=argparse.ArgumentParser()
 for name in ['root-contract','template','source','base-manifest','crf-manifest','sap-lexical-manifest','sap-scored-manifest','sap-scored-candidates','batch-plan','pubmed-cache','bioformer-cache','biobert-cache','output-plan','output-dir']:p.add_argument('--'+name,required=True)
 a=p.parse_args();contract=read(a.root_contract);assert sha(a.root_contract)==ROOT_CONTRACT and contract['input']['sha256']==B_INPUT and contract['documents']==100
 checked(contract['input']);checked(contract['hpo']);assert sha(a.source)==POOL_SOURCE
 plan=read(a.template);assert plan['execution_ready'] is False and plan['source_sha256']==POOL_SOURCE and plan['mode']=='official_B'
 base=read(a.base_manifest);assert base['inputs']['train']['sha256']==TRAIN64 and base['inputs']['b_input']['sha256']==B_INPUT and base['checks']['dev_bytes_exact'] is True
 assert base['outputs']['dev.base.jsonl']['sha256']=='5a994261965f7887f2bfef384401380d49581940435bbe46d19604f6e4868e4b'
 base_ref=base['outputs']['B.base.jsonl'];checked(base_ref)
 crf=read(a.crf_manifest);assert crf['training_source_sha256']==TRAIN64 and crf['inputs']['input']['sha256']==B_INPUT and crf['training_called'] is False
 assert crf['inputs']['pipeline']['sha256']=='8753c5f2b719094dea4bdcbb6d34ebb48a4ffb97b61508ac52e80f8592f75b01' and crf['inputs']['crf_model']['sha256']=='ee0f34ba65ae1fbaf181c01a1600c8afb2768410413582bcae71ed77e7c846e7'
 crf_ref=crf['outputs']['crf.pred.jsonl'];checked(crf_ref)
 lexical=read(a.sap_lexical_manifest);assert lexical['inputs']['train']['sha256']==TRAIN64 and lexical['inputs']['b_input']['sha256']==B_INPUT and lexical['checks']['all_dev_candidates_exact_original_fields'] is True
 assert lexical['inputs']['detector']['sha256']=='adb8ff8c8f395d74e2d76dc02995b5c51bf6e0fde818de7a7f4e8667e8b6b626' and lexical['gpu_query_contract']['bank_sha256']=='4fc0e062ea7940365222aad637058aadfb37dc57681e444eb2091577b591ae03'
 scored=read(a.sap_scored_manifest);sap_ref=ref(a.sap_scored_candidates);links=hashes(scored)
 assert sap_ref['sha256'] in links,'scored manifest must bind actual final scored candidate output SHA'
 raw_ref=lexical['outputs']['B.raw-candidates.jsonl'];query_ref=lexical['outputs']['B.query-requests.jsonl']
 assert B_INPUT in links or (raw_ref['sha256'] in links and query_ref['sha256'] in links),'scored manifest must bind root B or exact B raw+query inputs'
 assert TRAIN64 in links or sha(a.sap_lexical_manifest) in links or lexical['gpu_query_contract']['bank_sha256'] in links,'scored manifest must link frozen train64 lexical manifest or bank'
 assert not any(scored.get(k) is True for k in ('fit','training_called','fit_detector_called','learned_model_fit','quality_evaluation'))
 batch=read(a.batch_plan);assert batch['input_sha256']==B_INPUT and batch['input_contract_sha256']==ROOT_CONTRACT and batch['order']==['pubmedbert','bioformer','biobert']
 caches={};cache_checks={}
 for kind in ('pubmed','bioformer','biobert'):
  directory=Path(getattr(a,kind+'_cache'));cm=read(directory/'manifest.json');modelkind='pubmedbert' if kind=='pubmed' else kind;sub=next(x for x in batch['models'] if x['model']==modelkind)
  assert cm['input_sha256']==B_INPUT and cm['model_weight_sha256']==WEIGHTS[kind] and cm['public_model_training_on_task'] is False and cm['counts']['documents']==100
  assert len(cm['input_ids'])==len(set(cm['input_ids']))==100 and hashlib.sha256('\n'.join(cm['input_ids']).encode()).hexdigest()==contract['ordered_document_ids_sha256']
  assert cm['plan_sha256']==sub['plan_sha256']
  caches[kind]={role:ref(directory/name) for role,name in [('manifest','manifest.json'),('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]}
  for role,name in [('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]:assert caches[kind][role]['sha256']==cm['outputs'][name]['sha256'] and caches[kind][role]['bytes']==cm['outputs'][name]['bytes']
  policy=cm['fixed_policy'];assert policy['ngram_per_sentence_min']==2 and policy['ngram_per_sentence_max']==8 and policy['threshold_strictly_greater_than']==.95
  cache_checks[kind]=dict(model_weight_sha256=WEIGHTS[kind],native_plan_sha256=sub['plan_sha256'],ordered100_IDs_exact=True)
 out=Path(a.output_plan).resolve();assert not out.exists() and not Path(a.output_dir).exists()
 plan.update(execution_ready=True,input=contract['input'],hpo=contract['hpo'],input_contract=ref(a.root_contract),base=base_ref,crf=crf_ref,sap_candidates=sap_ref,caches=caches,output_dir=str(Path(a.output_dir).resolve()),additional_sources=[ref(x) for x in [a.base_manifest,a.crf_manifest,a.sap_lexical_manifest,a.sap_scored_manifest,a.batch_plan,__file__]])
 plan.pop('binding_notes',None);plan['metadata_binding']=dict(source=ref(__file__),source_pool=ref(a.source),base_train64_exact_dev_proof=True,crf_saved_original64_identity=True,sap_scored_output_and_pureinput_links=True,cache_checks=cache_checks,actual_candidate_reconstruction_executed=False,no_gold_read=True,no_model_fit_predict_or_quality=True)
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
 print(json.dumps(ref(out)))
if __name__=='__main__':main()
