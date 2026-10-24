"""Exactly one declared epoch stage under the original absolute campaign deadline."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,signal,subprocess,time

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 directory=p.parent
 while True:
  fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if directory.parent==directory:break
  directory=directory.parent
def gpu(plan):
 csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 row=[s for s in csv.splitlines() if s.split(',')[1].strip()==plan['gpu_uuid']];assert len(row)==1
 return dict(at_utc=datetime.now(timezone.utc).isoformat(),target_csv=row[0],target_compute_rows=[s for s in apps.splitlines() if s.split(',')[0].strip()==plan['gpu_uuid']])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--epoch',type=int,choices=[1],required=True);a=ap.parse_args()
 plan=json.loads(a.plan.read_text());campaign=json.loads(a.campaign.read_text());epoch=a.epoch
 assert campaign['plan_sha256']==sha(a.plan) and campaign['cap_seconds']==plan['cap_seconds'] and campaign['grace_seconds']==30
 deadline=campaign['absolute_deadline_unix'];assert deadline==campaign['started_unix']+plan['cap_seconds']
 assert time.time()<deadline and os.getuid()==1045 and os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
 for name,item in plan['payload_files'].items():
  p=Path(plan['remote_absolute'])/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 control=Path(plan['remote_absolute'])/'control'/('epoch-%03d'%epoch);control.mkdir(parents=True,exist_ok=False)
 output=Path(plan['output_dir']);assert plan['formal_training_authorized'] is True
 assert output.parent==Path(plan['storage_parent']) and output.name=='output'
 assert output.parent.parent==Path('/dev/shm') and output.parent.name.startswith('patientphex-')
 assert epoch==1 and not output.exists()
 fresh=gpu(plan);index,uuid,used,free,total=[x.strip() for x in fresh['target_csv'].split(',')]
 assert index=='2' and int(used)<=100 and int(free)>=78000 and int(total)==81920 and not fresh['target_compute_rows']
 assert os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize>=4*1024**3
 save(control/'fresh-gpu.json',fresh)
 training_deadline=deadline-plan['persistent_ack_reserve_seconds']
 assert plan['persistent_ack_reserve_seconds']==120 and time.time()<training_deadline
 argv=plan['train_argv']+['--absolute-deadline-unix',str(training_deadline)]
 argv+=['--stop-after-epoch','1']
 remaining=int(training_deadline-time.time());assert remaining>0
 command=argv
 started=time.time();save(control/'driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),epoch=epoch,command=command,plan_sha256=sha(a.plan),campaign_sha256=sha(a.campaign),absolute_deadline_unix=training_deadline,campaign_deadline_unix=deadline,remaining_seconds=remaining,once_only=True))
 with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
  process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
  save(control/'child-started.json',dict(pid=process.pid,pgid=process.pid))
  timed_out=False
  try:rc=process.wait(timeout=max(.001,training_deadline-time.time()))
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(process.pid,signal.SIGTERM)
   try:rc=process.wait(timeout=max(.001,training_deadline+30-time.time()))
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
 report=dict(epoch=epoch,returncode=rc,timed_out=timed_out,elapsed_seconds=time.time()-started,absolute_deadline_unix=training_deadline,campaign_deadline_unix=deadline,no_budget_reset=True,at_utc=datetime.now(timezone.utc).isoformat())
 if rc==0 and not timed_out:
  try:
   cp=output/('epoch-%03d'%epoch);meta=json.loads((cp/'checkpoint.json').read_text());assert meta['completed_epochs']==epoch and meta['global_step']==plan['configuration']['steps_per_epoch']*epoch and not meta['smoke_only']
   assert meta['fingerprint']['source_sha256']==plan['trainer_sha256']
   files={}
   for n,digest in meta['files'].items():
    p=cp/n;assert p.is_file() and not p.is_symlink() and sha(p)==digest;files[n]=dict(bytes=p.stat().st_size,sha256=digest)
   report.update(checkpoint=str(cp),checkpoint_sha256=sha(cp/'checkpoint.json'),files=files,global_step=plan['configuration']['steps_per_epoch']*epoch,final_state_persistent_export_pending=True)
   complete=json.loads((output/'training_complete.json').read_text());assert complete['epochs']==1 and complete['global_step']==33;report['training_complete_sha256']=sha(output/'training_complete.json')
  except Exception as e:report['endpoint_error']=dict(type=type(e).__name__,message=str(e))
 save(control/'wrapper-complete.json',report)
 release=gpu(plan);save(control/'gpu-release.json',release)
 print(json.dumps(report),flush=True);raise SystemExit(rc or int(timed_out or 'endpoint_error' in report))
if __name__=='__main__':main()
