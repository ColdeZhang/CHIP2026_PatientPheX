"""Bind actual A100 native47 identity to the unchanged verified rank32 mean adapter."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from synthesize import ref,save
S=Path(__file__).resolve().parent;R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F0';F1=R/'research/grounded-weight-average-v1/F1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/grounded-weight-average-v1/F0/infer-v1';CPU=P+'/grounded-weight-average-v1/F0/native-cpu-v1';OLD=P+'/grounded-weight-average-v1/F1/infer-v1'
old=json.loads((S.parent/'inference/infer-plan.json').read_text());cp=json.loads((S/'cpu-plan.json').read_text());col=json.loads((D/'native-cpu-v1/collection.json').read_text());assert col['CPU_reported_success'] and col['all_actual_SHA_verified']
result=ref(D/'native-cpu-v1/terminal/actual/result.json');rv=json.loads(Path(result['path']).read_text());assert rv['rows']==47 and rv['all47_local_native_actual_IDs_rendering_decode_exact'] and rv['input_tokens']==289420 and rv['max_input_tokens']==9935
p={k:v for k,v in old.items() if k not in ['assets','files','payload_sources','reuse_actual_artifacts','reuse_packages','input_sources','binding_source']}
p.update(schema='grounded-meanweights-held47-inference-v1',authorization='Root fixed F0 after strict positive F1; same synthesized delta32+CPU FP32 query; complete47 primary B4 union mean/original only',budget_seconds=3000,authorized_budget_seconds=3000,expected_rows=47,output_dir=ROOT+'/output')
p['argv']=[p['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'];p['operations_source']=ref(S/'remote.py');p['collection_source']=ref(S/'direct_collect.py');p['binding_source']=ref(__file__)
p['assets']={};p['payload_sources']={};p['reuse_actual_artifacts']={};p['reuse_packages']=[]
for key in ['raw_store','model_manifest','policy','normalization_policy','synthesis']:
 item=old['assets'][key];name=Path(item['path']).name;origin=old['payload_sources'][name];assert ref(origin['path'])==origin
 p['assets'][key]=dict(item,path=ROOT+'/'+name);p['payload_sources'][name]=origin;p['reuse_actual_artifacts'][name]=dict(item,original_root=OLD,original_member=name)
for key,name,originname in [('prompts','held-prompts.jsonl','held.input.jsonl'),('held_ids','held-actual-ids.json.gz','held-local-ids.json.gz'),('held_audit','held-audit.json','held-audit.json')]:
 origin=cp['payload_sources'][originname];assert ref(origin['path'])==origin;p['assets'][key]=dict(origin,path=ROOT+'/'+name);p['payload_sources'][name]=origin;p['reuse_actual_artifacts'][name]=dict(origin,path=CPU+'/'+originname,original_root=CPU,original_member=originname)
name='native-CPU-result.json';p['assets']['native_cpu_result']=dict(result,path=ROOT+'/'+name);p['payload_sources'][name]=result;p['reuse_actual_artifacts'][name]=dict(result,path=CPU+'/actual/result.json',original_root=CPU,original_member='actual/result.json')
p['assets']['runner']=dict(ref(S/'run.py'),path=ROOT+'/run.py');p['payload_sources']['run.py']=ref(S/'run.py');p['payload_sources']['infer_driver.py']=ref(S/'infer_driver.py')
p['files']={ROOT+'/'+name:{k:v[k] for k in ['bytes','sha256']} for name,v in p['payload_sources'].items()}
p['input_sources']=dict(F1_actual_complete=ref(F1/'infer-v1/terminal/output/complete.json'),F1_collection=ref(F1/'infer-v1/collection.json'),native_CPU_collection=ref(D/'native-cpu-v1/collection.json'),native_CPU_result=result,execution_plan=ref(S/'execution-plan.json'),ancestry=ref(S.parent/'evidence/F0-ancestry.json'),synthesis=ref(F1/'cpu-v1/collected/output/complete.json'))
save(S/'infer-plan.json',p)
checks=dict(status='PASS_FIXED_F0_SOURCE_BINDING',rows=47,documents=16,input_tokens=289420,max_input_tokens=9935,model_parameters=9040359936,same_weight_native_generation_policy=True,same_CPU_FP32_normalization=True,actual_CPU=result,cost_seconds=json.loads((S/'execution-plan.json').read_text())['cost_seconds'],budget_seconds=3000,grace_seconds=30,no_GPU_or_quality=True)
save(S.parent/'evidence/F0-source-contract.json',checks);print(json.dumps(checks))
