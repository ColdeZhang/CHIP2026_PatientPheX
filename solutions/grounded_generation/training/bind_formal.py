"""Bind complete actual engineering and an explicitly assigned formal budget."""
import argparse,hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=PROJECT+'/grounded-generation-v1/formal-v1'

def ref(p):p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def identity(x):return {k:x[k] for k in ['bytes','sha256']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--budget-seconds',required=True,type=int);ap.add_argument('--authorization-note',required=True);a=ap.parse_args()
 assert 0<a.budget_seconds<=3600 and a.authorization_note.strip()
 interface=json.loads((S/'interface.json').read_text());sources=interface['asset_bindings_or_unbound_actual_slots']
 c=B/'probe-v1';collection=json.loads((c/'collection.json').read_text());assert collection['probe_reported_success'] and collection['all_actual_SHA_verified']
 for item in collection['files'].values():assert ref(item['path'])==item
 for name,file in [('probe_complete','complete.json'),('probe_step','step.json'),('probe_manifest','run-manifest.json')]:sources[name]=ref(c/'terminal'/file)
 complete=json.loads(Path(sources['probe_complete']['path']).read_text());step=json.loads(Path(sources['probe_step']['path']).read_text())
 assert complete['actual_backwards']==step['actual_backwards']==14 and complete['actual_steps']==1 and step['all_three_Pareto_completed']
 assert complete['step_sha256']==sources['probe_step']['sha256'] and complete['manifest_sha256']==sources['probe_manifest']['sha256']
 names={'trainer':'train.py','encoded':'native-encoded.json.gz','train_audit':'train-audit.json','target_manifest':'target-manifest.json','model_manifest':'model-source-manifest.json','cpu_result':'CPU-result.json','cpu_runtime':'CPU-runtime.json','cpu_plan':'CPU-plan.json','native_cpu_source':'native_cpu.py','probe_complete':'probe-complete.json','probe_step':'probe-step.json','probe_manifest':'probe-manifest.json'}
 for name in ['indexed_head','slot_model','qwen35_runner','qwen35_relation_cost','qwen35_smoke']:names[name]=name+'.py'
 assert set(names)==set(sources) and len(names)==17 and len(set(names.values()))==17
 for item in sources.values():assert ref(item['path'])==item
 payload={names[k]:v for k,v in sources.items()};payload['formal_driver.py']=ref(S/'formal_driver.py')
 assets={k:dict(path=ROOT+'/'+names[k],**identity(v)) for k,v in sources.items()}
 probeplan=json.loads((S.parent/'runtime/probe-plan.json').read_text());env=probeplan['environment']
 reuse={}
 cpu=B/'native-cpu-v2';cpuresult=json.loads((cpu/'terminal/actual/result.json').read_text())
 for key,remote in [('encoded',cpuresult['encoded']['path']),('cpu_runtime',cpuresult['runtime']['path'])]+[(k,PROJECT+'/grounded-generation-v1/probe-v1/'+f) for k,f in [('probe_complete','complete.json'),('probe_step','step.json'),('probe_manifest','run-manifest.json')]]:
  origin=PROJECT+'/grounded-generation-v1/'+('native-cpu-v2' if key in ['encoded','cpu_runtime'] else 'probe-v1')
  reuse[names[key]]=dict(path=remote,original_root=origin,original_member=str(Path(remote).relative_to(origin)),**identity(sources[key]),actual_collection=ref((cpu if key in ['encoded','cpu_runtime'] else c)/'collection.json'))
 plan=dict(schema='grounded-whole-report-formal87-v1',phase='train',formal_training_authorized=True,authorization_note=a.authorization_note,
  budget_seconds=a.budget_seconds,authorized_budget_seconds=a.budget_seconds,grace_seconds=30,configuration=interface['configuration'],
  assets=assets,payload_sources=payload,files={ROOT+'/'+n:identity(v) for n,v in payload.items()},trainer_sha256=sources['trainer']['sha256'],
  python=probeplan['python'],model=probeplan['model'],gpu_uuid=probeplan['gpu_uuid'],environment=env,storage_parent=PROJECT,output_dir=ROOT+'/output',
  argv=[probeplan['python'],'-u',ROOT+'/train.py','--plan',ROOT+'/formal-plan.json'],
  operations_source=ref(S/'formal_remote.py'),collection_source=ref(S/'formal_collect.py'),binding_source=ref(__file__),
  input_sources=dict(native_CPU=ref(cpu/'collection.json'),actual_probe=ref(c/'collection.json'),target_manifest=sources['target_manifest']),
  reuse_packages=[ref(c/'package.json'),ref(cpu/'package.json')],reuse_actual_artifacts=reuse,
  fresh_base_and_adapter=True,probe_state_discarded=True,no_quality_or_inference_in_formal=True,
  expected_complete_epochs=2,actual_fullstate_durable_ACK_required_each_epoch=True,readback_SHA_fsync_required=True)
 path=S/'formal-plan.json';assert not path.exists();path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(path)))

if __name__=='__main__':main()
