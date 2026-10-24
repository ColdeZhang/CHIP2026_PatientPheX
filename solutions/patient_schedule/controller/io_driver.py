"""One CPU checkpoint server/fetch stage; never extend the parent deadline."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(Path(p).parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args()
 plan=json.loads(a.plan.read_text());campaign=json.loads(a.campaign.read_text());entry=json.loads(a.entry.read_text());phase=entry['phase'];epoch=entry['epoch']
 assert phase in ['server','fetch'] and epoch in [1,2,3,4] and entry['plan_sha256']==sha(a.plan)
 assert campaign['plan_sha256']==sha(a.plan) and campaign['cap_seconds']==5400 and campaign['grace_seconds']==30
 deadline=entry['absolute_deadline_unix'];assert deadline<=campaign['absolute_deadline_unix'] and deadline<=entry['started_unix']+600 and time.time()<deadline
 assert not os.environ.get('CUDA_VISIBLE_DEVICES')
 root=Path(plan['remote_absolute'] if phase=='server' else plan['persistent_remote_absolute']);control=root/'control'/('export-%s-epoch-%03d'%(phase,epoch));control.mkdir(parents=True,exist_ok=False)
 if phase=='server':
  assert os.getuid()==1045;name='export_checkpoint_server.py';argv=[plan['python'],'-u',str(root/name),'--plan',str(a.plan),'--checkpoint',str(Path(plan['output_dir'])/('epoch-%03d'%epoch)),'--deadline',str(deadline)]
 else:
  assert os.getuid()==1006;name='fetch_checkpoint.py';argv=[plan['persistent_python'],'-u',str(root/name),'--plan',str(a.plan),'--destination',str(Path(plan['persistent_checkpoint_root'])/('epoch-%03d'%epoch)),'--url',entry['url'],'--manifest-sha256',entry['manifest_sha256'],'--deadline',str(deadline)]
 assert sha(root/name)==plan['payload_files'][name]['sha256']
 remaining=int(deadline-time.time());assert remaining>0
 command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s']+argv
 save(control/'driver-started.json',dict(entry_sha256=sha(a.entry),absolute_deadline_unix=deadline,command=command,pid=os.getpid(),once_only=True))
 started=time.time()
 with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err);save(control/'child-started.json',dict(pid=child.pid));rc=child.wait()
 save(control/'wrapper-complete.json',dict(returncode=rc,seconds=time.time()-started,absolute_deadline_unix=deadline,at_epoch=time.time(),phase=phase,epoch=epoch));raise SystemExit(rc)
if __name__=='__main__':main()
