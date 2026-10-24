"""Bind only the actual synthesized delta average to unchanged native held39 inputs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from synthesize import ref,save
S=Path(__file__).resolve().parent;G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/inference')
R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F1';CPU=D/'cpu-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/grounded-weight-average-v1/F1/infer-v1';OLD=P+'/grounded-generation-v1/infer-F1-v1'
obs=json.loads(sorted(CPU.glob('observation-*.json'))[-1].read_text());actual=obs['files'];wrapper=actual['wrapper-complete.json']
assert wrapper['returncode']==0 and not wrapper['timed_out']
collected=CPU/'metadata';collected.mkdir(exist_ok=True)
for key in ['output/complete.json','output/tiny-peft.json']:
 target=collected/Path(key).name;raw=(json.dumps(actual[key],indent=2,allow_nan=False)+'\n').encode()
 if target.exists():assert target.read_bytes()==raw
 else:target.write_bytes(raw)
expected=wrapper['complete'];local=ref(collected/'complete.json');assert (local['bytes'],local['sha256'])==(expected['bytes'],expected['sha256'])
old=json.loads((G/'infer-plan.json').read_text());p={k:v for k,v in old.items() if k not in ['assets','files','payload_sources','reuse_actual_artifacts','reuse_packages','checkpoint_directory','binding_source']}
p.update(schema='grounded-meanweights-held39-inference-v1',authorization='root fixed two-epoch effective-delta average, complete39, only B4 union mean/original intersection',budget_seconds=1800,authorized_budget_seconds=1800,output_dir=ROOT+'/output',adapter=str(Path(actual['output/complete.json']['new_adapter']['path']).parent))
p['argv']=[p['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'];p['operations_source']=ref(S/'remote.py');p['collection_source']=ref(S/'collect.py');p['binding_source']=ref(__file__)
p['assets']={};p['payload_sources']={};p['reuse_actual_artifacts']={};p['reuse_packages']=[]
keep=['raw_store','prompts','held_ids','held_audit','model_manifest','policy','normalization_policy','native_cpu_result']
for key in keep:
 item=old['assets'][key];name=Path(item['path']).name
 p['assets'][key]=dict(item,path=ROOT+'/'+name)
 origin=old['payload_sources'][name];assert ref(origin['path'])==origin
 p['payload_sources'][name]=origin
 p['reuse_actual_artifacts'][name]=dict(item,original_root=OLD,original_member=name)
for key,name,localpath in [('runner','run.py',S/'run.py'),('synthesis','synthesis-complete.json',collected/'complete.json')]:
 item=ref(localpath);p['payload_sources'][name]=item;p['assets'][key]=dict(item,path=ROOT+'/'+name)
p['payload_sources']['infer_driver.py']=ref(S/'infer_driver.py')
p['files']={ROOT+'/'+name:{k:v[k] for k in ['bytes','sha256']} for name,v in p['payload_sources'].items()}
p['input_sources']=dict(old_input_sources=old['input_sources'],synthesis_observation=ref(sorted(CPU.glob('observation-*.json'))[-1]),CPU_source_plan=ref(S.parent/'cpu-plan.json'),execution_plan=ref(S.parent/'execution-plan.json'))
save(S/'infer-plan.json',p)
checks=dict(unchanged_payloads=keep,complete_rows=39,unchanged_native_ID_sha256=p['assets']['held_ids']['sha256'],new_model_parameters=9040359936,new_adapter=actual['output/complete.json']['new_adapter'],CPU_seconds=wrapper['seconds'],mean_provenance_not_epoch2=True,no_quality=True)
save(S.parent/'evidence/CPU-synthesis.json',dict(complete=local,tiny=ref(collected/'tiny-peft.json'),checks=checks))
print(json.dumps(checks,indent=2))
