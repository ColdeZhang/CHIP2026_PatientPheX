"""Bind the actual one-epoch persistent checkpoint to the prelocked held47 completion."""
from pathlib import Path
import hashlib,json
S=Path(__file__).resolve().parent;Q=S.parent
R=Path('/workspace/patientphex-runs/round2-20261023');N=R/'research/quote-set-completion-v1'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/quote-set-completion-v2/infer-F0-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def identity(v):return {k:v[k] for k in ['bytes','sha256']}
def read(p):return json.loads(Path(p).read_text())
def main():
 f=N/'formal-v1';h=N/'F0-input-preparation-v1';collected=read(f/'collection.json')
 assert collected['formal_reported_success'] and collected['all_actual_SHA_verified']
 for item in collected['files'].values():assert ref(item['path'])==item
 campaign=read(f/'campaign-complete.json');controller=read(f/'controller-plan.json');ack=read(f/'epoch-001-export-ack.json')
 assert campaign['status']=='ONE_COMPLETE_EPOCH_DURABLE_ACK_VERIFIED' and campaign['epoch1_ack_sha256']==ref(f/'epoch-001-export-ack.json')['sha256']
 assert ack['status']=='persistent_checkpoint_verified' and ack['completed_epochs']==1 and ack['global_step']==33
 checkpoint=f/'terminal/output/epoch-001/checkpoint.json';metadata=read(checkpoint)
 assert metadata['completed_epochs']==1 and metadata['global_step']==33 and metadata['fingerprint']['route']=='quote-set-completion-v2'
 assert ack['checkpoint_sha256']==ref(checkpoint)['sha256']
 cp=controller['output_dir']+'/epoch-001';assert ack['source_checkpoint_directory']==cp and ack['persistent_directory']==controller['persistent_checkpoint_root']+'/epoch-001'
 actual=collected['actual_external_checkpoints']['output/epoch-001/adapter/adapter_model.safetensors']
 assert actual['sha256']==metadata['files']['adapter/adapter_model.safetensors']==ack['files']['adapter/adapter_model.safetensors']['sha256']
 assert actual['persistent_path']==ack['persistent_directory']+'/adapter/adapter_model.safetensors'
 prep=read(h/'manifest.json');assert prep['rows']==47 and prep['known_triples']==204 and prep['no_held_gold']
 for item in prep['files'].values():assert ref(item['path'])==item
 inputs={
 'runner':('run.py',S/'run.py'),'raw_store':('raw_store.py',G/'inference/raw_store.py'),
 'trainer':('train.py',f/'train.py'),'checkpoint':('checkpoint.json',checkpoint),
 'training_complete':('training_complete.json',f/'terminal/output/training_complete.json'),
 'campaign_complete':('campaign-complete.json',f/'campaign-complete.json'),
 'epoch1_ack':('epoch-001-export-ack.json',f/'epoch-001-export-ack.json'),
 'prompts':('held-prompts.jsonl',h/'prompts.jsonl'),'held_ids':('held-ids.json.gz',h/'encoded.json.gz'),
 'held_audit':('held-audit.json',h/'audit.json'),'held_preparation':('held-preparation.json',h/'manifest.json'),
 'known':('known.json',h/'known.json'),'policy':('policy.json',S/'policy.json'),
 'model_manifest':('model-source-manifest.json',f/'model-source-manifest.json'),
 'normalization_policy':('normalization-policy.json',G/'normalization/policy_v1.json')}
 refs={k:ref(v[1]) for k,v in inputs.items()};assets={k:dict(path=ROOT+'/'+inputs[k][0],**identity(v)) for k,v in refs.items()}
 payload={inputs[k][0]:v for k,v in refs.items()};payload['infer_driver.py']=ref(S/'infer_driver.py')
 env={**controller['env'],'PATIENTPHEX_DURABILITY_ROOT':P}
 plan=dict(schema='quote-completion-F0-fresh47-v2',phase='infer',inference_authorized=True,
 authorization='Root fixed one261warm epoch durableACK then47 onecompletionpass1800+30; originalG32acceptedknown204 only; fixed B4union endpoint separately.',
 budget_seconds=1800,authorized_budget_seconds=1800,grace_seconds=30,admission_window_seconds=300,expected_rows=47,all_rows_fresh=True,max_new_tokens=1600,
 assets=assets,payload_sources=payload,files={ROOT+'/'+k:identity(v) for k,v in payload.items()},
 python=controller['python'],model=controller['model'],gpu_uuid=controller['gpu_uuid'],environment=env,storage_parent=P,output_dir=ROOT+'/output',
 checkpoint_directory=cp,adapter=cp+'/adapter',training_output_dir=controller['output_dir'],persistent_checkpoint_root=controller['persistent_checkpoint_root'],
 actual_adapter_sha256=actual['sha256'],argv=[controller['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'],
 operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),binding_source=ref(__file__),
 input_sources=dict(formal_collection=ref(f/'collection.json'),campaign=ref(f/'campaign-complete.json'),pureheld=ref(h/'manifest.json'),known=refs['known'],fixed_policy=refs['normalization_policy'],quality_policy=refs['policy']),
 reuse_packages=[],reuse_actual_artifacts={},runtime='Exact sameA800 Torch2.7.1 TF5.20.2 tokenizers.23.1 PEFT.21.2 native; complete47HF/native IDs replay beforemodel load',
 budget_basis=dict(rows=47,input_tokens=296853,max_input_tokens=10262,max_new_tokens=1600,actual_completion_F1_wrapper_seconds=123.47922615427524,actual_completion_F1_input_tokens=211943,actual_completion_F1_generated_tokens=1159,F1_empty_rows=33,F1_input_max10854_bound=True,new_completion_output_unknown=True,seconds=1800,no_partial_quality=True),
 quality_policy='Exactly B4union all legal normalized newcompletion relations; all947entities/47patients/oldrelations/nonrelationship bytes preserved; fixedfreeze1v2old0, noaddedbyteexact0v2.')
 target=S/'infer-plan.json';assert not target.exists();target.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(target)))
if __name__=='__main__':main()
