"""Bind actual complete two-epoch durable model to fresh whole-F1 input39.

Only metadata and published input IDs are read here; no model/gold/evaluator.
"""
import gzip,hashlib,json,os
from pathlib import Path
S=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
ROOT=PROJECT+'/grounded-generation-v1/infer-F1-v1'
FORMAL=PROJECT+'/grounded-generation-v1/formal-v1'

def ref(p):p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def ident(v):return {k:v[k] for k in ['bytes','sha256']}
def main():
 formal=B/'formal-v1';collection=json.loads((formal/'collection.json').read_text())
 assert collection['all_actual_SHA_verified']
 wrapper=json.loads((formal/'terminal/wrapper-complete.json').read_text())
 assert wrapper['returncode']==0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper and wrapper['all_epochs_persistent_and_readback_verified']
 for item in collection['files'].values():assert ref(item['path'])==item
 trainplan=json.loads((S.parent/'training/formal-plan.json').read_text());assert wrapper['training_complete_sha256']==ref(formal/'terminal/output/training_complete.json')['sha256']
 for epoch in [1,2]:
  ack=json.loads((formal/('terminal/output/durability-epoch-%03d.json'%epoch)).read_text());ck=json.loads((formal/('terminal/output/epoch-%03d/checkpoint.json'%epoch)).read_text())
  assert ack['completed_epochs']==epoch and ack['global_step']==11*epoch and ack['all_members_reopened_and_SHA_verified'] and ack['reopened_SHA_all_exact'] and ack['file_directory_fsync']
  assert ack['checkpoint_metadata_sha256']==ref(formal/('terminal/output/epoch-%03d/checkpoint.json'%epoch))['sha256']
  assert ack==wrapper['durability'][str(epoch)]['receipt'] and ck['fingerprint']['source_sha256']==trainplan['trainer_sha256']
 native=B/'native-cpu-v2';cpu=json.loads((native/'terminal/actual/result.json').read_text());assert cpu['all_local_native_actual_IDs_labels_rendering_exact'] and cpu['held_rows']==39
 enc=json.loads(gzip.decompress((native/'terminal/actual/encoded.json.gz').read_bytes()))['held']
 ids=B/'inference-preparation/held-actual-ids.json.gz';assert json.loads(gzip.decompress(ids.read_bytes()))==enc
 names={'runner':'run.py','raw_store':'raw_store.py','trainer':'train.py','checkpoint':'checkpoint.json','durability':'durability-epoch-002.json','training_complete':'training_complete.json','training_wrapper':'training-wrapper-complete.json','prompts':'held-prompts.jsonl','held_ids':'held-actual-ids.json.gz','held_audit':'held-audit.json','model_manifest':'model-source-manifest.json','policy':'policy.json','normalization_policy':'normalization-policy.json','native_cpu_result':'native-CPU-result.json'}
 inputs={'runner':S/'run.py','raw_store':S/'raw_store.py','trainer':S.parent/'training/train.py','checkpoint':formal/'terminal/output/epoch-002/checkpoint.json','durability':formal/'terminal/output/durability-epoch-002.json','training_complete':formal/'terminal/output/training_complete.json','training_wrapper':formal/'terminal/wrapper-complete.json','prompts':B/'targets-v2/F1.input.jsonl','held_ids':ids,'held_audit':B/'token-census-v1/F1-input-audit.json','model_manifest':Path(trainplan['payload_sources']['model-source-manifest.json']['path']),'policy':S/'policy.json','normalization_policy':S.parent/'normalization/policy_v1.json','native_cpu_result':native/'terminal/actual/result.json'}
 assert set(inputs)==set(names)
 refs={k:ref(v) for k,v in inputs.items()};assert refs['normalization_policy']['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
 assets={k:dict(path=ROOT+'/'+names[k],**ident(v)) for k,v in refs.items()};payload={names[k]:v for k,v in refs.items()};payload['infer_driver.py']=ref(S/'infer_driver.py')
 reuse={}
 for key,remote in [('checkpoint',FORMAL+'/output/epoch-002/checkpoint.json'),('durability',FORMAL+'/output/durability-epoch-002.json'),('training_complete',FORMAL+'/output/training_complete.json'),('training_wrapper',FORMAL+'/wrapper-complete.json')]:
  reuse[names[key]]=dict(path=remote,original_root=FORMAL,original_member=str(Path(remote).relative_to(FORMAL)),**ident(refs[key]),actual_collection=ref(formal/'collection.json'))
 plan=dict(schema='grounded-held39-inference-v1',phase='infer',inference_authorized=True,authorization='Parent /root complete39 fresh after both durable ACK; independent1200+TERM30; two fixed quality endpoints separately',budget_seconds=1200,authorized_budget_seconds=1200,grace_seconds=30,expected_rows=39,all_rows_fresh=True,max_new_tokens=1600,
  assets=assets,payload_sources=payload,files={ROOT+'/'+k:ident(v) for k,v in payload.items()},python=trainplan['python'],model=trainplan['model'],gpu_uuid=trainplan['gpu_uuid'],environment=trainplan['environment'],storage_parent=PROJECT,output_dir=ROOT+'/output',checkpoint_directory=FORMAL+'/output/epoch-002',adapter=FORMAL+'/output/epoch-002/adapter',argv=[trainplan['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'],operations_source=ref(S/'remote.py'),collection_source=ref(S/'collect.py'),binding_source=ref(__file__),input_sources=dict(formal_collection=ref(formal/'collection.json'),native_CPU_collection=ref(native/'collection.json'),train_target=ref(B/'targets-v2/manifest.json'),normalizer_policy=refs['normalization_policy'],normalizer_sources={p.name:ref(p) for p in [S.parent/'normalization/strict.py',S.parent/'normalization/semantic.py',S.parent/'normalization/public_bank.py']}),reuse_packages=[ref(formal/'package.json'),ref(native/'package.json')],reuse_actual_artifacts=reuse)
 path=S/'infer-plan.json';assert not path.exists()
 with path.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(ref(path)))
if __name__=='__main__':main()
