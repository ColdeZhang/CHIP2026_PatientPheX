"""Fixed Med fresh2ep, two durable epoch exports, then one independent-budget80 inference.

No evaluator/gold imports. Partial, unknown, OOM or nonfinite stages do not advance.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W))
from run_paired_ack_paths import Run,sha,ref

class MedRun(Run):
 def export_epoch(self,epoch):
  p=self.plan;cp=p['training_output']+f'/epoch-{epoch:03d}';out=self.root/f'export-{epoch:03d}';out.mkdir()
  job=self.start_bounded(p['mount_id'],['python3','-u',p['remote_tools']+'/export_checkpoint_server.py','--checkpoint',cp],'.',{},out/'server-job.json')
  terminal=False;fetch=None
  try:
   ready=None
   while time.time()<self.deadline and ready is None:
    x=self.call(p['mount_id'],'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':40000})
    for line in x.get('content','').splitlines():
     if line.startswith('{'):ready=json.loads(line)
    if not ready:assert x['running'],'Exporter stopped before ready';time.sleep(2)
   assert ready and ready['ready'];self.save(out/'server-ready.json',ready)
   destination=p['persistent_root']+f'/epoch-{epoch:03d}'
   argv=['python3','-u',p['persistent_tools']+'/fetch_checkpoint.py','--url','http://10.5.4.7:'+str(ready['port']),'--manifest-sha256',ready['manifest_sha256'],'--destination',destination,'--deadline',str(self.deadline)]
   fetch=self.start_bounded(p['persistent_mount_id'],argv,'.',{},out/'fetch-job.json');self.watch(p['persistent_mount_id'],fetch,out/'fetch');terminal=True
   code="from pathlib import Path;import hashlib,json;p=Path("+repr(destination)+");a=json.loads((p/'export-ack.json').read_text());assert a['status']=='persistent_checkpoint_verified';assert all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size==r['bytes'] and hashlib.sha256((p/n).read_bytes()).hexdigest()==r['sha256'] for n,r in a['files'].items());print(json.dumps(a))"
   actual=self.command(p['persistent_mount_id'],code,60);self.save(out/'persistent-actual-readback.json',actual);ack=json.loads(actual['stdout'])
   assert ack['files']==ready['manifest']['files'] and ack['checkpoint_sha256']==ready['manifest']['checkpoint_sha256'] and ack['source_checkpoint_directory']==cp
   assert ack['completed_epochs']==epoch and ack['global_step']==16*epoch and ack['persistent_directory']==destination and ack['persistent_mount_id']==p['persistent_mount_id']
   self.save(out/'export-ack.json',ack);target=self.root/f'epoch-{epoch:03d}'
   for remote,local in [('checkpoint.json','checkpoint.json'),('_run/run_manifest.json','run_manifest.json'),('_run/progress.json','progress.json')]:self.get_file(p['persistent_mount_id'],destination+'/'+remote,target/local,ack['files'][remote])
   self.save(target/'export-ack.json',ack)
   if epoch==2:
    first=json.loads((self.root/'epoch-001/checkpoint.json').read_text());last=json.loads((target/'checkpoint.json').read_text());assert first['fingerprint']==last['fingerprint']
   return ack
  except BaseException as exc:
   self.save(out/'controller-interruption.json',{'error_type':type(exc).__name__,'error_text':str(exc),'known_server':job,'known_fetch':fetch,'fetch_terminal_confirmed':terminal,'absolute_deadline_unix':self.deadline,'action':'preserve_known_jobs_on_unknown_no_new_start'});raise
  finally:
   if terminal or time.time()>=self.deadline+30:self.save(out/'server-stopped.json',self.call(p['mount_id'],'stop_process',{'process_id':job['id']}))

 def verify_resume_ack(self):
  p=self.plan;ack=json.loads((self.root/'epoch-001/export-ack.json').read_text());meta=json.loads((self.root/'epoch-001/checkpoint.json').read_text())
  assert ack['completed_epochs']==1 and ack['global_step']==16 and ack['persistent_mount_id']==p['persistent_mount_id'] and ack['persistent_directory']==p['persistent_root']+'/epoch-001'
  assert ack['checkpoint_sha256']==sha(self.root/'epoch-001/checkpoint.json') and ack['run_manifest_sha256']==sha(self.root/'epoch-001/run_manifest.json') and not meta['smoke_only']
  # Independently reopen source own epoch after durable ACK. No stage reset or
  # new optimizer schedule: native trainer restores that epoch's complete state.
  code='from pathlib import Path;import hashlib,json;cp=Path('+repr(p['training_output']+'/epoch-001')+');r='+repr(ack['files'])+';assert all((cp/n).stat().st_size==v["bytes"] and hashlib.sha256((cp/n).read_bytes()).hexdigest()==v["sha256"] for n,v in r.items() if not n.startswith("_"));print(json.dumps(dict(status="same_complete_epoch1_source_after_durable_ACK",checkpoint_sha256=hashlib.sha256((cp/"checkpoint.json").read_bytes()).hexdigest())))'
  self.save(self.root/'epoch-001/source-resume-actual.json',self.command(p['mount_id'],code,30))

 def train_epoch(self,epoch):
  p=self.plan;out=self.root/f'train-{epoch:03d}';out.mkdir();self.gpu_empty(f'train-{epoch:03d}/gpu-preflight')
  if epoch==2:self.verify_resume_ack()
  argv=p['training_common_argv']+['--stop-after-epoch',str(epoch)]
  if epoch==2:argv+=['--resume',p['training_output']+'/epoch-001']
  job=self.start_bounded(p['mount_id'],argv,'.',p['env'],out/'job.json')
  try:self.watch(p['mount_id'],job,out)
  except BaseException as exc:
   self.save(out/'monitor-interruption.json',{'error_type':type(exc).__name__,'known_job':job,'absolute_deadline_unix':self.deadline,'action':'read_original_job_no_relaunch'});raise
  self.gpu_empty(f'train-{epoch:03d}/gpu-release')
  progress=self.shm_json(p['training_output']+'/progress.json',out/'progress.json');assert progress['completed_epochs']==epoch and progress['global_step']==16*epoch
  marker='paused.json' if epoch==1 else 'training_complete.json';result=self.shm_json(p['training_output']+'/'+marker,out/marker)
  if epoch==1:assert result['completed_epochs']==1 and result['global_step']==16
  else:assert result['epochs']==2 and result['global_step']==32 and not result['smoke_only'] and result['source_sha256']==p['trainer_sha256']
  return self.export_epoch(epoch)

 def infer(self,ack):
  p=self.plan;cp=self.root/'epoch-002/checkpoint.json';complete=self.root/'train-002/training_complete.json';adapter=p['training_output']+'/epoch-002/adapter'
  adapter_files={n:ack['files']['adapter/'+n]['sha256'] for n in ['adapter_config.json','adapter_model.safetensors']}
  ip={'checkpoint_sha256':sha(cp),'training_complete_sha256':sha(complete),'adapter_files':adapter_files,'adapter_directory':adapter,'prompts_sha256':p['prompts_sha256'],'runner_sha256':p['runner_sha256'],'model_source_sha256':p['model_source_sha256'],'rows':80,'no_F1_gold_opened':True,'pretraining_amendment_sha256':p['pretraining_amendment']['sha256']}
  self.save(self.root/'inference-plan.json',ip);self.gpu_empty('inference-gpu-preflight')
  started=time.time();self.deadline=started+600
  self.save(self.root/'inference-execution-state.json',{'status':'unique_original80_started','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_unix':started,'absolute_deadline_unix':self.deadline,'TERM_grace_seconds':30})
  remote=p['remote_inference_output'];argv=p['inference_common_argv']+['--output',remote,'--adapter',adapter]
  job=self.start_bounded(p['mount_id'],argv,'.',p['env'],self.root/'inference-job.json')
  try:self.watch(p['mount_id'],job,self.root/'inference')
  except BaseException as exc:
   self.save(self.root/'inference-monitor-interruption.json',{'error_type':type(exc).__name__,'known_job':job,'absolute_deadline_unix':self.deadline,'action':'read_original_job_no_relaunch'});raise
  self.gpu_empty('inference-gpu-release')
  for suffix in ['jsonl','manifest.json','checkpoint.json','loading.json']:self.get_file(p['mount_id'],remote[:-5]+suffix,self.root/('responses.'+suffix))
  rows=[json.loads(x) for x in (self.root/'responses.jsonl').read_text().splitlines()];meta=json.loads((self.root/'responses.manifest.json').read_text());fp=meta['fingerprint']
  assert len(rows)==len({x['row_id'] for x in rows})==meta['expected_rows']==80 and {x['row_id']:str(x['pmc_id']) for x in rows}==p['row_documents']
  assert meta['output_sha256']==sha(self.root/'responses.jsonl') and fp['adapter_files']==adapter_files and fp['source_sha256']==p['runner_sha256'] and fp['prompts_sha256']==p['prompts_sha256']
  self.save(self.root/'responses-freeze.json',{'plan_sha256':sha(self.root/'inference-plan.json'),'raw_sha256':sha(self.root/'responses.jsonl'),'manifest_sha256':sha(self.root/'responses.manifest.json'),'rows':80,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  self.save(self.root/'inference-complete.json',{'status':'complete_original80_before_quality','seconds':time.time()-started,'absolute_deadline_unix':self.deadline,'no_gold_no_quality_score':True})

 def execute(self):
  p=self.plan;assert sha(__file__)==p['controller_sha256'] and sha(W/'run_paired_ack_paths.py')==p['prior_controller_source_sha256'];assert p['training_cap_seconds']==1200 and p['inference_cap_seconds']==600 and p['grace_seconds']==30
  assert not (self.root/'execution-state.json').exists(),'Recover existing known jobs; do not duplicate'
  for r in p['references'].values():assert ref(r['path'])==r
  self.gpu_empty('initial-gpu-preflight')
  code='from pathlib import Path;import hashlib,json,os,shutil;refs='+repr(p['remote_source_sha256'])+';assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in refs.items());out=Path('+repr(p['training_output'])+');assert not out.exists();assert out.parent.stat().st_uid==os.getuid()==1045;assert not Path('+repr(p['remote_inference_output'])+').exists();assert shutil.disk_usage("/dev/shm").free>2*1024**3 and shutil.disk_usage(".").free>20*1024**2;print(json.dumps(dict(shm_free=shutil.disk_usage("/dev/shm").free,project_free=shutil.disk_usage(".").free)))'
  self.save(self.root/'source-and-storage-preflight.json',self.command(p['mount_id'],code,20))
  code='from pathlib import Path;import hashlib,json,shutil;root=Path('+repr(p['persistent_root'])+');assert not root.exists();assert shutil.disk_usage(".").free>2*1024**3;assert hashlib.sha256(Path('+repr(p['persistent_tools']+'/fetch_checkpoint.py')+').read_bytes()).hexdigest()=='+repr(p['fetcher_sha256'])+';print(json.dumps(dict(persistent_free=shutil.disk_usage(".").free)))'
  self.save(self.root/'persistent-storage-preflight.json',self.command(p['persistent_mount_id'],code,20))
  started=time.time();self.deadline=started+1200
  self.save(self.root/'execution-state.json',{'status':'fresh_two_epoch_stage_running','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_unix':started,'absolute_deadline_unix':self.deadline,'TERM_grace_seconds':30,'plan_sha256':sha(self.root/'run-plan.json')})
  for epoch in (1,2):ack=self.train_epoch(epoch)
  self.remaining()
  self.save(self.root/'training-campaign-complete.json',{'status':'complete_two_epochs_and_both_persistent_ACKs','epochs':2,'optimizer_calls':32,'exposures':248,'elapsed_seconds':time.time()-started,'absolute_deadline_unix':self.deadline,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_gold_no_quality_score':True})
  self.infer(ack)
  print('MED_FRESH2EP_DURABLE_AND_FIXED80_COMPLETE_NO_QUALITY',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);a=p.parse_args();MedRun(a.directory.resolve()).execute()
if __name__=='__main__':main()
