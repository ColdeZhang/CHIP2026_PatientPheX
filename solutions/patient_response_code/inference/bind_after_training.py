"""Bind completed two-epoch coded adapter and native held80; never load model or gold."""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',type=Path,required=True);ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();C=a.controller;D=a.directory;S=Path(__file__).resolve().parent
 cp=json.loads((C/'controller-plan.json').read_text());done=json.loads((C/'campaign-complete.json').read_text());assert done['status']=='TWO_COMPLETE_EPOCHS_32_STEPS_DURABLE_ACKS_VERIFIED';assert done['campaign']['plan_sha256']==sha(C/'controller-plan.json');assert done['at_epoch']<=done['campaign']['absolute_deadline_unix']
 acks=[]
 for e in [1,2]:
  p=C/f'epoch-{e:03d}-export-ack.json';v=json.loads(p.read_text());assert v['status']=='persistent_checkpoint_verified' and v['completed_epochs']==e and v['global_step']==16*e
  assert v['fsync_all_files'] and v['reopened_final_filename_SHA_all_files'] and v['destination_directories_fsynced'];assert sha(p)==done[f'epoch{e}_ack_sha256']
  verified=json.loads((C/f'epoch-{e:03d}-export-verification.json').read_text());assert verified['all_files_reopened_SHA'] and verified['ack']==v
  assert json.loads((C/f'epoch-{e:03d}-source-ack-copy.json').read_text())['sha256']==sha(p)
  acks.append(v)
 assert acks[1]['trainer_sha256']==sha(S.parent/'formal_train/train.py')==cp['trainer_sha256'];assert acks[1]['files']==json.loads((C/'epoch-002-server-ready.json').read_text())['manifest']['files']
 Q=Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-readiness-20261024');R=Q.parent/'patient-response-code-v1';audit=R/'inference-preparation/row-audit.json';held=Q/'held.response-code.jsonl'
 assert sha(held)=='54c8cb8c3e273dca62d6972cf24fc520d921fde091f35626931adf5de2676e9b'
 native=json.loads((Q/'native-terminal/native-row-audit.json').read_text());expected=[x for x in native if x['group']=='held' and x['arm']=='response_code'];assert len(expected)==80 and json.loads(audit.read_text())==expected
 D.mkdir(exist_ok=False);sources={n:S/n for n in ['infer_full80.py','probe_runner.py','legacy_runner.py','launch.py','response_codec.py','model-source-manifest.json']};sources.update({'held.response-code.jsonl':held,'row-audit.json':audit,'training-campaign-complete.json':C/'campaign-complete.json','training-epoch2-persistent-ACK.json':C/'epoch-002-export-ack.json'})
 for n,p in sources.items():(D/n).write_bytes(p.read_bytes())
 plan=dict(schema='patient-response-code-short2ep-fulltext80-native-v1',status='TWO_ACTUAL_DURABLE_ACKS_BOUND_NO_GPU',mount_id=cp['mount_id'],gpu_uuid=cp['gpu_uuid'],python=cp['python'],model=cp['model'],remote='patient-response-code/F1-full80-v1',adapter=cp['output_dir']+'/epoch-002/adapter',adapter_files={n:acks[1]['files']['adapter/'+n]['sha256'] for n in ['adapter_config.json','adapter_model.safetensors']},env=cp['env'],cap_seconds=1200,term_grace_seconds=30,input_limit=14848,generation_limit=1600,cached_original_indices=[],fresh_rows=80,training_completed_epochs=2,training_global_step=32,configured_eos_ids=[248044,248046],expected_runtime=dict(torch='2.6.0+cu124',transformers='5.20.2',peft='0.21.2',tokenizers='0.23.1',actual_tokenizer_class='Qwen3_5Tokenizer',matmul_TF32=False,cudnn_TF32=True),files={n:{k:v for k,v in ref(D/n).items() if k!='path'} for n in sources},native_census=ref(Q/'native-terminal/native-result.json'),quality_policy=ref(S.parent/'quality-policy.json'),training_provenance=dict(controller=ref(C/'controller-plan.json'),campaign=ref(C/'campaign-complete.json'),ACKs=[ref(C/f'epoch-{e:03d}-export-ack.json') for e in [1,2]],epoch2_full_files=acks[1]['files']),raw_first=True,whole_row_abstention=True,no_repair=True,unique_quality_calls=2,old_quality_calls=0)
 save(D/'plan.json',plan);files={n:ref(D/n) for n in list(sources)+['plan.json']}
 with tarfile.open(D/'deployment.tar.gz','x:gz') as t:
  for n in files:t.add(D/n,arcname=n,recursive=False)
 with tarfile.open(D/'deployment.tar.gz') as t:
  assert set(t.getnames())==set(files)
  for n,r in files.items():b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 save(D/'deployment-manifest.json',dict(archive=ref(D/'deployment.tar.gz'),files=files))
 save(D/'entry.json',dict(plan=ref(D/'plan.json'),operations=ref(S/'operations.py'),deployment=ref(D/'deployment-manifest.json'),quality_plan=ref(S.parent/'quality-plan.json'),all80fresh=True,old_cache_reuse=False));print(json.dumps(ref(D/'entry.json')))
if __name__=='__main__':main()
