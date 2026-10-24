"""Freeze the existing 39-input CPU receipt and exact adapter replica on A30.

No tokenizer/model/quality calls. The original A800 deployment remains unstarted.
"""
import copy,json
from pathlib import Path
from prepare import S,D,P,ref,ident,save

T=S/'a30'
ROOT=P+'/grounded-sampled-consensus-v1/infer-F1-A30-v1'

def main():
 old=json.loads((S/'runtime/infer-plan.json').read_text())
 host_ref=ref(D/'a30-host-inventory-v1.json');host=json.loads(Path(host_ref['path']).read_text())
 actual=host['actual'];assert actual['status']=='PASS_SAME_NATIVE_A30_ASSETS_NO_GPU_NO_TOKENIZATION'
 assert actual['base_stats_same_as_prior_SHA_certificate'] and actual['no_GPU_model_or_tokenization']
 assert actual['versions']==dict(torch='2.7.1+cu126',transformers='5.20.2',peft='0.21.2',tokenizers='0.23.1')
 assert actual['runtime_sha256']==dict(qwen_modeling='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3',generation_utils='7944240ff1ee27a7b3acc16571b7a480cc14f16e7222aeb1a79d289ebc96e3ea')
 copy_ref=ref(D/'copy-A30-v1/copy-complete.json');transfer=json.loads(Path(copy_ref['path']).read_text());h=transfer['ack']
 original_copy=json.loads(Path(old['input_sources']['copy_complete']['path']).read_text())['ack']
 assert h['files']==original_copy['files'] and h['source_directory']==old['checkpoint_directory']
 assert h['destination']==P+'/grounded-sampled-consensus-v1/model-cache/epoch-002'
 assert h['source_fsync_and_reopen_SHA'] and h['destination_all_files_fsync_and_reopen_SHA'] and h['destination_ancestor_directories_fsynced']
 assert h['original_source_retained'] and not h['optimizer_state_copied'] and h['GPU_or_model_calls']==0
 refs=copy.deepcopy(old['payload_sources'])
 for name in ['run.py','raw_store.py','sampling.py','infer_driver.py']:refs[name]=ref(T/name)
 refs.update({'host-cache-binding.json':copy_ref,'host-runtime.json':host_ref})
 for item in refs.values():assert ref(item['path'])==item
 assets={k:dict(path=ROOT+'/'+Path(v['path']).name,**ident(refs[Path(v['path']).name])) for k,v in old['assets'].items()}
 assets.update(host_cache_binding=dict(path=ROOT+'/host-cache-binding.json',**ident(copy_ref)),host_runtime=dict(path=ROOT+'/host-runtime.json',**ident(host_ref)))
 p=copy.deepcopy(old);p.update(authorization='Root explicit prelaunch A30 migration, same model/input/sampler; one full78 native run3600+TERM30, no selective retry',budget_seconds=3600,authorized_budget_seconds=3600,
   python='/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',model=P+'/case-retrieval/qwen35-preparation/models/Qwen3.5-9B',
   gpu_uuid='GPU-126293b9-6677-8464-cee7-01e693efbf29',source_inference_cache_directory=old['checkpoint_directory'],
   checkpoint_directory=h['destination'],adapter=h['destination']+'/adapter',assets=assets,payload_sources=refs,
   files={ROOT+'/'+name:ident(item) for name,item in refs.items()},output_dir=ROOT+'/output',
   operations_source=ref(T/'remote.py'),collection_source=ref(T/'collect.py'),binding_source=ref(__file__),reuse_packages=[],reuse_actual_artifacts={},
   actual_row_order='each seed: descending input_tokens then original row_id; row-local RNG independent of order')
 p['argv']=[p['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json']
 p['environment']['CUDA_VISIBLE_DEVICES']=p['gpu_uuid']
 p['environment']['PYTHONPATH']=P+'/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:'+P+'/case-retrieval/round2/runtime'
 p['input_sources'].update(original_unstarted_A800_plan=ref(S/'runtime/infer-plan.json'),host_inventory=host_ref,host_cache_copy=copy_ref,host_budget_revision=ref(S/'decision-host-budget-v3.json'))
 p['budget_basis']=dict(cap_seconds=3600,grace_seconds=30,prelaunch_revision=ref(S/'decision-host-budget-v3.json'),prior_A30_rows116_seconds=805.858144,prior_A30_rows128_seconds=851.003390,new78_outputtoken_proxy_seconds=2369,margin_factor=1.45,additional_seconds=150,not_a_guarantee=True)
 p['inference_runtime']=dict(versions=old['inference_runtime'],host='.100 A30 GPU0',native_CPU_reused_from_A800=True,cross_GPU_bitwise_equivalence_claimed=False)
 assert len(refs)==19 and p['seeds']==[20261024,20261025] and p['presence_penalty']==0
 assert not (D/'infer-F1-v1/start-intent.json').exists()
 save(T/'infer-plan.json',p);print(json.dumps(ref(T/'infer-plan.json')))

if __name__=='__main__':main()
