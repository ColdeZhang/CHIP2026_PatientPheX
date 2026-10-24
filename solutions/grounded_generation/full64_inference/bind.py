"""Bind actual full64 final checkpoint and both durable ACKs, without generation."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;BASE=S.parent
R=Path('/workspace/patientphex-runs/round2-20261023')
B=R/'research/grounded-generation-full64-v1'
READY=R/'research/grounded-generation-full64-readiness-v1'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
ROOT=P+'/grounded-generation-full64-v1/infer-dev36-v1'
CPU=P+'/grounded-generation-full64-v1/native-cpu-v1'

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def read(p):return json.loads(Path(p).read_text())

def main():
 formal=B/'formal-v1';native=B/'native-cpu-v1'
 collected=read(formal/'collection.json');cpu_collection=read(native/'collection.json')
 assert collected['formal_reported_success'] and cpu_collection['CPU_reported_success']
 for c in [collected,cpu_collection]:
  assert c['all_actual_SHA_verified']
  for item in c['files'].values():assert ref(item['path'])==item
 campaign=read(formal/'campaign-complete.json');controller=read(formal/'controller-plan.json')
 assert campaign['status']=='TWO_COMPLETE_EPOCHS_DURABLE_ACKS_VERIFIED'
 checkpoint=formal/'terminal/output/epoch-002/checkpoint.json';metadata=read(checkpoint)
 assert metadata['completed_epochs']==2 and metadata['global_step']==44
 assert metadata['fingerprint']['route']=='grounded-full64-v1'
 cpu=read(native/'terminal/actual/result.json');assert cpu['train_rows']==173 and cpu['held_rows']==36
 ack1=read(formal/'epoch-001-export-ack.json');ack2=read(formal/'epoch-002-export-ack.json')
 for epoch,ack in [(1,ack1),(2,ack2)]:
  assert campaign['epoch%d_ack_sha256'%epoch]==ref(formal/('epoch-%03d-export-ack.json'%epoch))['sha256']
  assert ack['status']=='persistent_checkpoint_verified' and ack['global_step']==22*epoch
 assert ack2['checkpoint_sha256']==ref(checkpoint)['sha256']
 cp=controller['output_dir']+'/epoch-002'
 assert ack2['source_checkpoint_directory']==cp and ack2['persistent_directory']==controller['persistent_checkpoint_root']+'/epoch-002'
 actual=collected['actual_external_checkpoints']['output/epoch-002/adapter/adapter_model.safetensors']
 assert actual['sha256']==metadata['files']['adapter/adapter_model.safetensors']==ack2['files']['adapter/adapter_model.safetensors']['sha256']
 assert actual['persistent_path']==ack2['persistent_directory']+'/adapter/adapter_model.safetensors'
 inputs={
  'runner':('run.py',S/'run.py'),'raw_store':('raw_store.py',BASE/'inference/raw_store.py'),
  'trainer':('train.py',formal/'train.py'),'checkpoint':('checkpoint.json',checkpoint),
  'training_complete':('training_complete.json',formal/'terminal/output/training_complete.json'),
  'campaign_complete':('campaign-complete.json',formal/'campaign-complete.json'),
  'epoch1_ack':('epoch-001-export-ack.json',formal/'epoch-001-export-ack.json'),
  'epoch2_ack':('epoch-002-export-ack.json',formal/'epoch-002-export-ack.json'),
  'prompts':('held-prompts.jsonl',READY/'targets-v2/dev.input36.jsonl'),
  'held_ids':('held-actual-ids.json.gz',native/'terminal/actual/encoded.json.gz'),
  'held_audit':('held-audit.json',READY/'token-census-v1/dev-input-audit.json'),
  'model_manifest':('model-source-manifest.json',formal/'model-source-manifest.json'),
  'policy':('policy.json',S/'policy.json'),'normalization_policy':('normalization-policy.json',BASE/'normalization/policy_v1.json'),
  'normalizer_train64':('normalizer-train64-manifest.json',READY/'targets-v2/normalizer-train64-manifest.json'),
  'native_cpu_result':('native-CPU-result.json',native/'terminal/actual/result.json')}
 refs={k:ref(v[1]) for k,v in inputs.items()}
 assets={k:dict(path=ROOT+'/'+inputs[k][0],**ident(v)) for k,v in refs.items()}
 payload={inputs[k][0]:v for k,v in refs.items()};payload['infer_driver.py']=ref(S/'infer_driver.py')
 reuse={}
 for target,member,key in [('held-actual-ids.json.gz','actual/encoded.json.gz','held_ids'),('native-CPU-result.json','actual/result.json','native_cpu_result')]:
  reuse[target]=dict(path=CPU+'/'+member,original_root=CPU,original_member=member,**ident(refs[key]),actual_collection=ref(native/'collection.json'))
 env={**controller['env'],'PATIENTPHEX_DURABILITY_ROOT':P}
 p=dict(schema='grounded-dev36-full64-inference-v1',phase='infer',inference_authorized=True,
  authorization='Root approved full64 complete two durable ACKs then originaldev pure36 native greedy1600; no local gold/quality/official action',
  budget_seconds=1200,authorized_budget_seconds=1200,grace_seconds=30,expected_rows=36,all_rows_fresh=True,max_new_tokens=1600,
  assets=assets,payload_sources=payload,files={ROOT+'/'+k:ident(v) for k,v in payload.items()},
  python=controller['python'],model=controller['model'],gpu_uuid=controller['gpu_uuid'],environment=env,storage_parent=P,output_dir=ROOT+'/output',
  checkpoint_directory=cp,adapter=cp+'/adapter',training_output_dir=controller['output_dir'],persistent_checkpoint_root=controller['persistent_checkpoint_root'],
  actual_adapter_sha256=actual['sha256'],argv=[controller['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'],
  operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),binding_source=ref(__file__),
  input_sources=dict(formal_collection=ref(formal/'collection.json'),campaign=ref(formal/'campaign-complete.json'),native_CPU_collection=ref(native/'collection.json'),
   targets=ref(READY/'targets-v2/manifest.json'),normalizer_train64=refs['normalizer_train64'],fixed_policy=refs['normalization_policy']),
  reuse_packages=[ref(native/'package.json')],reuse_actual_artifacts=reuse,
  runtime='Torch2.7.1+cu126/TF5.20.2/tokenizers.23.1/PEFT.21.2; original native, no FLA',
  budget_basis=dict(authoritative_seconds=1200,rows=36,input_tokens=163925,max_input_tokens=8858,max_new_tokens=1600,
    old_F0_47_same_A800_native_seconds=593.56223599,old_F1_39_A100_native_seconds=1103.215470307,
    new_model_output_length_unknown=True,no_old_response_reuse=True),
  quality_policy='Only full new-only and all normalized quote supply. Root owns final originaldev complete comparison freezes and evaluators; no gold opened here.')
 target=S/'infer-plan.json';assert not target.exists();target.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(target)))

if __name__=='__main__':main()
