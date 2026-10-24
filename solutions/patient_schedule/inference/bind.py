"""Bind only completed epoch4 and four exact durable ACKs; never launch a model."""
from pathlib import Path
import argparse,json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quality'))
from common import ref,read,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);a=p.parse_args();b=a.controller;c=read(b/'controller-plan.json');campaign=read(b/'campaign-complete.json')
 assert campaign['status']=='FOUR_COMPLETE_EPOCHS_64_STEPS_DURABLE_ACKS_VERIFIED' and campaign['campaign']['plan_sha256']==ref(b/'controller-plan.json')['sha256']
 acknowledgements={}
 for ep in range(1,5):
  ack=read(b/f'epoch-{ep:03d}-export-ack.json');state=read(b/f'epoch-{ep:03d}-state.json')
  assert ack['completed_epochs']==ep and ack['global_step']==16*ep and ack['trainer_sha256']==c['trainer_sha256']
  assert ack['status']=='persistent_checkpoint_verified' and ack['fsync_all_files'] and ack['reopened_final_filename_SHA_all_files'] and ack['destination_directories_fsynced']
  assert state['wrapper']['returncode']==0 and not state['release']['target_compute_rows']
  assert ref(b/f'epoch-{ep:03d}-export-ack.json')['sha256']==campaign['epoch_ACKs'][str(ep)]
  assert ack['checkpoint_sha256']==state['wrapper']['checkpoint_sha256']
  acknowledgements[str(ep)]=ref(b/f'epoch-{ep:03d}-export-ack.json')
 root=Path(c['remote_absolute']);cp=Path(c['output_dir'])/'epoch-004';adapter=cp/'adapter';inference=root/'inference-output';ack4=read(b/'epoch-004-export-ack.json')
 proof=dict(four_ACKs_verified=True,ACKs=acknowledgements,campaign=ref(b/'campaign-complete.json'),trainer_sha256=c['trainer_sha256'],adapter_sha256=ack4['files']['adapter/adapter_model.safetensors']['sha256'],no_old_adapter_response_cache=True)
 save(b/'inference-proof.json',proof)
 files={str(root/n):r for n,r in c['payload_files'].items()}
 files[str(root/'inference-proof.json')]={k:v for k,v in ref(b/'inference-proof.json').items() if k!='path'}
 for n in ['adapter/adapter_model.safetensors','adapter/adapter_config.json','checkpoint.json'] : files[str(cp/n)]=ack4['files'][n]
 files[str(Path(c['output_dir'])/'training_complete.json')]=ack4['files']['_run/training_complete.json']
 argv=[c['python'],'-u',str(root/'infer.py'),'--model',c['model'],'--source-manifest',str(root/'model-source-manifest.json'),'--prompts',str(root/'F1.input80.jsonl'),'--output',str(inference/'responses.jsonl'),'--adapter',str(adapter),'--token-audit',str(root/'F1-input-token-audit.json'),'--training-checkpoint',str(cp/'checkpoint.json'),'--training-complete',str(Path(c['output_dir'])/'training_complete.json'),'--trainer-sha256',c['trainer_sha256'],'--campaign-proof',str(root/'inference-proof.json')]
 plan=dict(cap_seconds=1200,grace_seconds=30,root=str(root),output=str(inference/'responses.jsonl'),gpu_uuid=c['gpu_uuid'],argv=argv,files=files,controller_plan=ref(b/'controller-plan.json'),proof=ref(b/'inference-proof.json'),all80_fresh=True)
 save(b/'inference-plan.json',plan);print(ref(b/'inference-plan.json'))
if __name__=='__main__':main()
