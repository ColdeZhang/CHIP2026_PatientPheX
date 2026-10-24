"""Bound only an already-running local read-only collector; never start a collector/model."""
import argparse,json,os,signal,time
from datetime import datetime,timezone
from pathlib import Path
from prepare import D,S,ref,save

def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,start_ticks=fields[19],state=fields[0],argv=(p/'cmdline').read_bytes().split(b'\0'),cwd=str((p/'cwd').resolve()))
 except (FileNotFoundError,ProcessLookupError):return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);ap.add_argument('--start-ticks',required=True);ap.add_argument('--deadline',required=True,type=float);a=ap.parse_args()
 directory=D/'infer-v1';status_path=directory/'full-raw-collection-status.json';accepted=ref(D/'accepted/manifest.json');assert accepted['sha256']=='a1242b31c29ff8bd35ff4cb3c5104cb62f8a744d28fbf76a1508c2d0b497a874'
 owner=dict(pid=a.pid,start_ticks=a.start_ticks,cwd=str(S));own=identity(os.getpid())
 save(directory/'readback-guardian-ready.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),guardian_pid=os.getpid(),guardian_start_ticks=own['start_ticks'],collector=owner,deadline_unix=a.deadline,source=ref(__file__),no_collector_restart=True,no_remote_calls=True,accepted_manifest_frozen=accepted))
 signal_sent=False;reason=None
 while True:
  if (directory/'collection.json').exists():
   c=json.loads((directory/'collection.json').read_text());assert c['all_actual_SHA_verified'] and c['job_reported_success'];assert c['remote_archive']['sha256']=='cb201713c40df3130236cd57a438783af5412e941dd8fc2dee41268a73e62ff7'
   status=dict(status='COMPLETE_ORIGINAL_RAW_ARCHIVE_LOCAL',collection=ref(directory/'collection.json'),archive=c['archive'],remote_archive=c['remote_archive'],members=len(c['files']),all_raw_locally_collected=True);break
  actual=identity(a.pid)
  if actual is None or actual['state']=='Z':reason='Original collector exited without full collection receipt';break
  if actual['start_ticks']!=a.start_ticks or actual['cwd']!=str(S) or b'collect.py' not in actual['argv'] or str(directory).encode() not in actual['argv']:
   reason='PID identity changed; no signal sent';break
  if time.time()>=a.deadline:
   os.kill(a.pid,signal.SIGTERM);signal_sent=True;reason='Read-only drain deadline reached; TERM only exact collector identity; full remote archive retained';break
  time.sleep(min(15,max(.1,a.deadline-time.time())))
 if reason is not None:status=dict(status='FULL_RAW_LOCAL_COLLECTION_PENDING',reason=reason,local_collector_TERM_sent=signal_sent,all_raw_locally_collected=False,remote_archive=json.loads((S/'essential-plan-v2.json').read_text())['original_full_archive'])
 assert ref(D/'accepted/manifest.json')==accepted
 save(status_path,dict(**status,at_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),guardian_ready=ref(directory/'readback-guardian-ready.json'),accepted_manifest_unchanged=accepted,no_model_or_remote_calls=True))
 print(json.dumps(ref(status_path)),flush=True)

if __name__=='__main__':main()
