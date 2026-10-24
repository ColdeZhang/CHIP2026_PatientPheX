"""Local preparation of fixed four-epoch source/data payload; no remote or GPU calls."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,shutil,tarfile
S=Path(__file__).resolve().parent

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();R=a.run_root;B=a.output;B.mkdir(exist_ok=False);old=R/'research/patient-first-closed-set/formal-preparation';payload=json.loads((old/'payload.json').read_text())
 source_files={p.name:p for p in [S/'train.py',S/'runtime_contract.py',S/'patient_first.py',S/'final_parser.py',S/'qwen35_smoke.py',S/'qwen35_relation_cost.py',S/'inference/infer.py',*(S/'controller'/n for n in ['epoch_driver.py','io_driver.py','export_checkpoint_server.py','fetch_checkpoint.py'])]};source_files['inference_driver.py']=S/'inference/driver.py'
 names=['train.jsonl','excluded.input.jsonl','preparation-manifest.json','transposition-summary.json','train-token-audit.json','native-cpu-receipt.json','model-source-manifest.json','F1.input80.jsonl','F1-input-token-audit.json']
 provenance={};decompress={}
 for name in names:
  r=payload[name];actual=ref(r['path']);assert actual['bytes']==r['bytes'] and actual['sha256']==r['sha256'];provenance[name]=actual;raw=Path(r['path']).read_bytes()
  if name.endswith('.jsonl'):
   (B/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0));decompress[name+'.gz']={k:v for k,v in actual.items() if k!='path'}|dict(name=name)
  else:(B/name).write_bytes(raw)
 for name,path in source_files.items():(B/name).write_bytes(path.read_bytes())
 plan=json.loads((R/'research/qwen35-fulltext32-indexed-formal-v1/controller/controller-plan.json').read_text())
 relative='patient-schedule/fresh4-v1';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+relative;out='/dev/shm/patientphex-patient-schedule4-1045-20261024/formal4ep'
 plan.update(status='PREPARED_ONLY_GPU_NOT_AUTHORIZED',remote_relative=relative,remote_absolute=remote,persistent_remote_relative=relative,persistent_remote_absolute=remote,persistent_checkpoint_root=remote+'/checkpoints',output_dir=out,trainer_sha256=ref(S/'train.py')['sha256'],expanded_train_sha256=provenance['train.jsonl']['sha256'],fingerprint_schema='patientphex-qwen35-meta32-patient-first-four-epoch-v1',export_trainer_basename='train.py',export_run_files=['run_manifest.json','train.py','qwen35_smoke.py','qwen35_relation_cost.py','runtime_contract.py','patient_first.py','progress.json'],total_absolute_budget_seconds=5400,term_grace_seconds=30,export_seconds_each=600,no_inference_or_quality_here=True,no_probe_state_reuse=True,resume_policy='four fresh unified epochs; pause each epoch; next epoch only after all-files persistent ACK; restore full Adam/scheduler/all RNG/globalstep; one original5400 clock includes four exports',epochs=4,total_optimizer_steps=64,warmup_steps=4,decompress=decompress)
 plan.pop('source_training_plan',None);plan.pop('CPU_contract',None);plan['source_training_plan']=ref(S/'training-plan.json');plan['provenance']=provenance
 argv=[plan['python'],'-u',remote+'/train.py','--mode','train']
 for flag,name in [('train','train.jsonl'),('excluded-input','excluded.input.jsonl'),('data-manifest','preparation-manifest.json'),('transposition-manifest','transposition-summary.json'),('train-token-audit','train-token-audit.json'),('native-cpu-receipt','native-cpu-receipt.json'),('schema-helper','patient_first.py'),('model-source-manifest','model-source-manifest.json')]:argv+=['--'+flag,remote+'/'+name]
 argv+=['--model',plan['model'],'--output-dir',out,'--gpu-uuid',plan['gpu_uuid']];plan['train_argv']=argv
 plan['payload_files']={p.name:{k:v for k,v in ref(p).items() if k!='path'} for p in B.iterdir() if p.is_file()}
 plan['payload_files'].update({item['name']:{k:v for k,v in item.items() if k!='name'} for item in decompress.values()})
 save(B/'controller-plan.json',plan)
 files={p.name:{k:v for k,v in ref(p).items() if k!='path'} for p in B.iterdir() if p.is_file()}
 archive=B/'deployment.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for name in sorted(files):t.add(B/name,arcname=name)
 with tarfile.open(archive) as t:
  assert set(t.getnames())==set(files)
  for name,item in files.items():b=t.extractfile(name).read();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
 save(B/'deployment-manifest.json',dict(archive=ref(archive),files=files,no_remote=True))
 save(B/'GPU-AUTHORIZATION.template.json',dict(authorized=False,controller_plan_sha256=ref(B/'controller-plan.json')['sha256'],budget_seconds=5400,F1_inference_and_quality_authorized=False,note='Root scheduling decision required; this is a template, not an approval.'))
 print(json.dumps(dict(controller=ref(B/'controller-plan.json'),deployment=ref(archive),files=len(files),no_GPU_or_remote=True)))
if __name__=='__main__':main()
