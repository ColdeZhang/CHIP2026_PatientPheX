"""Exactly one declared epoch stage under the original absolute campaign deadline."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,time

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def gpu(plan):
 csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 row=[s for s in csv.splitlines() if s.split(',')[1].strip()==plan['gpu_uuid']];assert len(row)==1
 return dict(at_utc=datetime.now(timezone.utc).isoformat(),target_csv=row[0],target_compute_rows=[s for s in apps.splitlines() if s.split(',')[0].strip()==plan['gpu_uuid']])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--epoch',type=int,choices=[1,2],required=True);a=ap.parse_args()
 plan=json.loads(a.plan.read_text());campaign=json.loads(a.campaign.read_text());epoch=a.epoch
 assert campaign['plan_sha256']==sha(a.plan) and campaign['cap_seconds']==8400 and campaign['grace_seconds']==30
 deadline=campaign['absolute_deadline_unix'];assert deadline==campaign['started_unix']+8400
 assert time.time()<deadline and os.getuid()==1045 and os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
 for name,item in plan['payload_files'].items():
  p=Path(plan['remote_absolute'])/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 control=Path(plan['remote_absolute'])/'control'/('epoch-%03d'%epoch);control.mkdir(parents=True,exist_ok=False)
 output=Path(plan['output_dir']);assert str(output).startswith('/dev/shm/patientphex-qwen35-fulltext32-1045-20261024/')
 if epoch==1:assert not output.exists()
 else:
  assert (output/'epoch-001/checkpoint.json').is_file() and not (output/'epoch-002').exists()
  ack=output/'export-ack-epoch-001.json';proof=json.loads(ack.read_text());assert proof['status']=='persistent_checkpoint_verified' and proof['completed_epochs']==1 and proof['global_step']==16
  assert proof['fsync_all_files'] and proof['reopened_final_filename_SHA_all_files'] and proof['destination_directories_fsynced']
  assert proof['checkpoint_sha256']==sha(output/'epoch-001/checkpoint.json')
 fresh=gpu(plan);index,uuid,used,free,total=[x.strip() for x in fresh['target_csv'].split(',')]
 assert index=='2' and int(used)<=100 and int(free)>=78000 and int(total)==81920 and not fresh['target_compute_rows']
 assert os.statvfs(output.parent).f_bavail*os.statvfs(output.parent).f_frsize>=4*1024**3
 save(control/'fresh-gpu.json',fresh)
 argv=plan['train_argv']+['--absolute-deadline-unix',str(deadline)]
 if epoch==1:argv+=['--stop-after-epoch','1']
 else:argv+=['--resume',str(output/'epoch-001'),'--resume-export-ack',str(output/'export-ack-epoch-001.json')]
 remaining=int(deadline-time.time());assert remaining>0
 command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s']+argv
 started=time.time();save(control/'driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),epoch=epoch,command=command,plan_sha256=sha(a.plan),campaign_sha256=sha(a.campaign),absolute_deadline_unix=deadline,remaining_seconds=remaining,once_only=True))
 with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
  process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
  save(control/'child-started.json',dict(pid=process.pid,pgid=process.pid));rc=process.wait()
 report=dict(epoch=epoch,returncode=rc,elapsed_seconds=time.time()-started,absolute_deadline_unix=deadline,no_budget_reset=True,at_utc=datetime.now(timezone.utc).isoformat())
 if rc==0:
  try:
   cp=output/('epoch-%03d'%epoch);meta=json.loads((cp/'checkpoint.json').read_text());assert meta['completed_epochs']==epoch and meta['global_step']==16*epoch and not meta['smoke_only']
   assert meta['fingerprint']['source_sha256']==plan['trainer_sha256']
   files={}
   for n,digest in meta['files'].items():
    p=cp/n;assert p.is_file() and not p.is_symlink() and sha(p)==digest;files[n]=dict(bytes=p.stat().st_size,sha256=digest)
   report.update(checkpoint=str(cp),checkpoint_sha256=sha(cp/'checkpoint.json'),files=files,global_step=16*epoch,final_state_persistent_export_pending=True)
   if epoch==1:
    paused=json.loads((output/'paused.json').read_text());assert paused['completed_epochs']==1 and paused['global_step']==16
   else:
    complete=json.loads((output/'training_complete.json').read_text());assert complete['epochs']==2 and complete['global_step']==32;report['training_complete_sha256']=sha(output/'training_complete.json')
  except Exception as e:report['endpoint_error']=dict(type=type(e).__name__,message=str(e))
 save(control/'wrapper-complete.json',report)
 release=gpu(plan);save(control/'gpu-release.json',release)
 print(json.dumps(report),flush=True);raise SystemExit(rc or int('endpoint_error' in report))
if __name__=='__main__':main()
