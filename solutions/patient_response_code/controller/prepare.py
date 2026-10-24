"""Bind immutable formal core and the one actual pressure result, then package private data."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,tarfile

def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--probe-terminal',type=Path,required=True);ap.add_argument('--cost-decision',type=Path,required=True);a=ap.parse_args()
 C=Path(__file__).resolve().parent;F=C.parent/'formal_train';B=a.directory;B.mkdir(exist_ok=False);training=json.loads((F/'training-plan.json').read_text())
 terminal=json.loads(a.probe_terminal.read_text());decision=json.loads(a.cost_decision.read_text());assert terminal['returncode']==0 and not terminal['timed_out'] and terminal['all12_backwards_and_one_update_verified'] and terminal['target_has_no_process'];assert decision['formal_cost_fits_3600'] is True
 files=['train.py','runtime_contract.py','selected_loss.py','indexed_head.py','qwen35_smoke.py','qwen35_relation_cost.py','model-source-manifest.json','training-plan.json','CPU-contract-result.json']
 for n in files:shutil.copyfile(F/n,B/n)
 controls=['epoch_driver.py','io_driver.py','export_checkpoint_server.py','fetch_checkpoint.py']
 for n in controls:shutil.copyfile(C/n,B/n)
 data=[]
 for key,n in [('data_manifest','original-preparation-manifest.json'),('response_preparation','structural-result.json'),('native_receipt','native-result.json'),('native_row_audit','native-row-audit.json')]:
  item=training['data_arguments'][key];assert ref(item['path'])==item;shutil.copyfile(item['path'],B/n);data.append(n)
 decompress={}
 for key,n in [('train','train.response-code.jsonl'),('excluded_input','excluded.input.jsonl')]:
  item=training['data_arguments'][key];assert ref(item['path'])==item;b=Path(item['path']).read_bytes();gz=n+'.gz';(B/gz).write_bytes(gzip.compress(b,mtime=0));decompress[gz]=dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 project='/home/zhangyuheng/projects/CHIP2026_PatientPheX';relative='patient-response-code/normal2ep-v1';root=project+'/'+relative;python=training['python'];uuid='GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1';output=training['output_directory']
 allfiles=files+controls+data+list(decompress)
 payload={n:{k:v for k,v in ref(B/n).items() if k!='path'} for n in allfiles}
 for item in decompress.values():payload[item['name']]={k:item[k] for k in ['bytes','sha256']}
 argv=[python,'-u',root+'/train.py','--mode','train','--train',root+'/train.response-code.jsonl','--excluded-input',root+'/excluded.input.jsonl','--data-manifest',root+'/original-preparation-manifest.json','--response-preparation',root+'/structural-result.json','--native-receipt',root+'/native-result.json','--native-row-audit',root+'/native-row-audit.json','--model',training['model'],'--model-source-manifest',root+'/model-source-manifest.json','--output-dir',output,'--gpu-uuid',uuid]
 env=dict(CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=training['pythonpath'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 plan=dict(status='FROZEN_FRESH_SHORT124_CODED_2EPOCHS_32UPDATES_TWO_DURABLE_ACKS',mount_id='476f81cf-09e7-445b-a7f0-d0e12b12e50f',persistent_mount_id='5a9955da-5240-41cc-9e52-05d3d8da8a3c',source_owner='zhangyuheng',remote_relative=relative,remote_absolute=root,persistent_remote_relative=relative,persistent_remote_absolute=root,persistent_checkpoint_root=training['persistent_directory'],python=python,persistent_python='/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',source_private_IP='10.5.3.12',gpu_uuid=uuid,model=training['model'],output_dir=output,trainer_sha256=training['source_files']['train.py']['sha256'],train_sha256=training['train_sha256'],fingerprint_schema=training['fingerprint_schema'],export_trainer_basename='train.py',export_run_files=training['run_files'],total_absolute_budget_seconds=3600,term_grace_seconds=30,export_seconds_each=600,source_training_plan=ref(F/'training-plan.json'),CPU_contract=ref(F/'CPU-contract-result.json'),probe_terminal=ref(a.probe_terminal),cost_decision=ref(a.cost_decision),train_argv=argv,payload_files=payload,decompress=decompress,env=env,resume_policy=training['resume'],no_probe_state_reuse=True,no_inference_or_quality_here=True)
 save(B/'controller-plan.json',plan);allfiles+=['controller-plan.json'];m={n:{k:v for k,v in ref(B/n).items() if k!='path'} for n in allfiles}
 with tarfile.open(B/'deployment.tar.gz','w:gz') as t:
  for n in allfiles:t.add(B/n,arcname=n,recursive=False)
 save(B/'deployment-manifest.json',dict(archive=ref(B/'deployment.tar.gz'),files=m,controller_plan=ref(B/'controller-plan.json'),raw_inputs_verified_after_decompression=decompress))
 save(B/'GPU-AUTHORIZATION.json',dict(authorized=True,controller_plan_sha256=ref(B/'controller-plan.json')['sha256'],budget_seconds=3600,grace_seconds=30,source='root conditional authorization now backed by actual pressure PASS and measured cost estimate',no_fresh_probe_state=True))
 save(B/'entry.json',dict(plan=ref(B/'controller-plan.json'),operations=ref(C/'operations.py'),deployment_operation=ref(C/'deploy.py'),launcher=ref(C/'launch_watch.py'),watcher=ref(C/'watch_authorized_campaign.py'),deployment=ref(B/'deployment-manifest.json'),authorization=ref(B/'GPU-AUTHORIZATION.json'),unknown='Inspect original intent/job; never blindly repeat model start',budget_seconds=3600))
 print(json.dumps(ref(B/'entry.json')))
if __name__=='__main__':main()
