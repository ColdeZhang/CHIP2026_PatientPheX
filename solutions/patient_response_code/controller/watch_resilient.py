"""Local-only once-only scheduler for the already-authorized two-epoch campaign.

Calls the frozen operations controller; never changes runtime or retries an
unknown mutation. Original training/export deadlines remain authoritative.
"""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time
S=Path(__file__).parent
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--recovery',required=True);ap.add_argument('--first-inspect-not-before',type=float,required=True);a=ap.parse_args();D=a.directory;F=D/a.recovery;assert Path(a.recovery).name==a.recovery and F.is_dir()
lock=(D/'monitor-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x):
 with (F/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
plan=json.loads((D/'controller-plan.json').read_text());campaign=json.loads((D/'campaign.json').read_text());deadline=campaign['absolute_deadline_unix']
assert deadline==campaign['started_unix']+3600 and campaign['plan_sha256']==hashlib.sha256((D/'controller-plan.json').read_bytes()).hexdigest()
assert (D/'epoch-001-job.json').exists() and not (D/'campaign-complete.json').exists()
ops=S/'operations.py';expected=''
source_hash=hashlib.sha256(ops.read_bytes()).hexdigest()
save('local-watch-intent.json',dict(at=now(),pid=os.getpid(),operations_sha256=source_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),absolute_deadline_unix=deadline,no_remote_new_recipe=True,unknown_outcome_policy='stop_local_watcher_preserve_original_intent_no_retry',training_poll_seconds=120,export_poll_seconds=15))
def log(event,**data):
 x=dict(at=now(),event=event,**data)
 with (F/'local-watch-events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(x),flush=True)
def run(action,epoch):
 while True:
  assert time.time()<deadline and hashlib.sha256(ops.read_bytes()).hexdigest()==source_hash
  before=set(D.glob('inspect-*-tool.json'))
  result=subprocess.run([sys.executable,str(ops),action,'--directory',str(D),'--epoch',str(epoch)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=min(180,max(1,deadline-time.time())))
  if result.returncode:
   errors=[]
   if action=='inspect':
    for p in set(D.glob('inspect-*-tool.json'))-before:
     value=json.loads(p.read_text());kind='LUNA_OFFLINE' if 'LUNA_OFFLINE' in json.dumps(value) else 'UNKNOWN_OUTCOME' if 'UNKNOWN_OUTCOME' in json.dumps(value) else None
     if value.get('isError') and kind:errors.append(dict(receipt=str(p),kind=kind,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   if errors:
    not_before=time.time()+185;log('readonly_inspect_backoff',errors=errors,not_before_unix=not_before,remote_model_untouched=True)
    while time.time()<not_before:nap(min(30,not_before-time.time()))
    continue
   raise RuntimeError('Original operation failed; unknown mutation never repeated: '+action+': '+result.stderr[-1000:])
  log('original_operation_complete',action=action,epoch=epoch);return
def intent_job_gate(stem):
 if (D/(stem+'-intent.json')).exists() and not (D/(stem+'-job.json')).exists():raise RuntimeError('Unknown prior mutation '+stem+'; no repeated start')
def nap(seconds):
 assert time.time()<deadline
 time.sleep(min(seconds,max(0,deadline-time.time())))
try:
 log('recovery_ready',first_inspect_not_before_unix=a.first_inspect_not_before,remote_model_untouched=True)
 while time.time()<a.first_inspect_not_before:nap(min(30,a.first_inspect_not_before-time.time()))
 for epoch in [1,2]:
  tag='epoch-%03d'%epoch
  if epoch>1 and not (D/(tag+'-job.json')).exists():
   assert not (D/(tag+'-start-intent.json')).exists(),'Unknown epoch2 start; inspect original'
   run('start-epoch',epoch)
  while not (D/(tag+'-state.json')).exists():
   run('inspect',epoch)
   if not (D/(tag+'-state.json')).exists():nap(120)
  state=json.loads((D/(tag+'-state.json')).read_text());assert state['wrapper']['returncode']==0 and not state['release']['target_compute_rows']
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
