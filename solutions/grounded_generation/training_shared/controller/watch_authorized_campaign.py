"""Local-only once-only scheduler for the already-authorized two-epoch campaign.

Calls the frozen operations controller; never changes runtime or retries an
unknown mutation. Original training/export deadlines remain authoritative.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
S=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();D=a.directory.resolve()
owner=(D/'monitor-owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):
 with (D/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
plan=json.loads((D/'controller-plan.json').read_text());campaign=json.loads((D/'campaign.json').read_text());deadline=campaign['absolute_deadline_unix']
assert deadline==campaign['started_unix']+plan['cap_seconds'] and campaign['plan_sha256']==hashlib.sha256((D/'controller-plan.json').read_bytes()).hexdigest()
assert plan['formal_training_authorized'] is True
assert (D/'epoch-001-job.json').exists() and not (D/'campaign-complete.json').exists()
ops=S/'operations.py';expected=''
source_hash=hashlib.sha256(ops.read_bytes()).hexdigest()
save('local-watch-intent-'+str(time.time_ns())+'.json',dict(at=now(),pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),exclusive_lock=True,operations_sha256=source_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),absolute_deadline_unix=deadline,no_remote_new_recipe=True,unknown_outcome_policy='stop_local_watcher_preserve_original_intent_no_retry',training_poll_seconds=180,export_poll_seconds=15))
def log(event,**data):
 x=dict(at=now(),event=event,**data)
 with (D/'local-watch-events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(x),flush=True)
def run(action,epoch):
 assert time.time()<deadline and hashlib.sha256(ops.read_bytes()).hexdigest()==source_hash
 result=subprocess.run([sys.executable,str(ops),action,'--directory',str(D),'--epoch',str(epoch)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=min(180,max(1,deadline-time.time())))
 if result.returncode:raise RuntimeError('Original operations failed, inspect saved original intent/tool receipt; '+action+': '+result.stderr[-1000:])
 log('original_operation_complete',action=action,epoch=epoch)
def intent_job_gate(stem):
 if (D/(stem+'-intent.json')).exists() and not (D/(stem+'-job.json')).exists():raise RuntimeError('Unknown prior mutation '+stem+'; no repeated start')
def nap(seconds):
 assert time.time()<deadline
 time.sleep(min(seconds,max(0,deadline-time.time())))
try:
 for epoch in [1,2]:
  tag='epoch-%03d'%epoch
  if epoch==2 and not (D/(tag+'-job.json')).exists():
   assert not (D/(tag+'-start-intent.json')).exists(),'Unknown epoch2 start; inspect original'
   run('start-epoch',epoch)
  while not (D/(tag+'-state.json')).exists():
   run('inspect',epoch)
   if not (D/(tag+'-state.json')).exists():nap(180)
  state=json.loads((D/(tag+'-state.json')).read_text());assert state['wrapper']['returncode']==0 and not state['wrapper'].get('timed_out') and 'endpoint_error' not in state['wrapper'] and not state['release']['target_compute_rows']
  log('epoch_finished',epoch=epoch,wrapper_seconds=state['wrapper'].get('seconds',state['wrapper'].get('wrapper_seconds')))
  if (D/(tag+'-export-ack.json')).exists():
   assert (D/(tag+'-source-ack-copy.json')).exists(),'Incomplete original ACK publication; do not retry'
   continue
  if not (D/(tag+'-server-job.json')).exists():
   assert not (D/(tag+'-export-intent.json')).exists(),'Unknown original server start; inspect'
   run('start-server',epoch)
  if not (D/(tag+'-fetch-job.json')).exists():
   assert not (D/(tag+'-fetch-intent.json')).exists(),'Unknown original fetch start; inspect'
   while not (D/(tag+'-fetch-job.json')).exists():
    run('start-fetch',epoch)
    if not (D/(tag+'-fetch-job.json')).exists():nap(15)
  while not (D/(tag+'-export-ack.json')).exists():
   run('collect-export',epoch)
   if not (D/(tag+'-export-ack.json')).exists():nap(15)
  assert (D/(tag+'-source-ack-copy.json')).exists()
  log('epoch_durable_ACK_complete',epoch=epoch)
 assert (D/'campaign-complete.json').exists()
 save('local-watch-complete.json',dict(at=now(),status='TWO_EPOCHS_DURABLE_ORIGINAL_CAMPAIGN_COMPLETE',no_inference_or_quality_started=True))
 log('complete_no_further_model_or_quality')
except BaseException as exc:
 save('local-watch-failure.json',dict(at=now(),error=repr(exc),policy='Original model/controller state preserved; no automatic rerun or deadline reset'))
 raise
