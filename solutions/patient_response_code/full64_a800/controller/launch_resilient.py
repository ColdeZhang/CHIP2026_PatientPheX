"""Detach one monitor after prior local worker exited; never restarts epoch1."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time

def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',type=Path,required=True);ap.add_argument('--recovery',required=True);ap.add_argument('--not-before',type=float,required=True);a=ap.parse_args();C=a.controller;S=Path(__file__).resolve().parent;old=json.loads((C/'watch-job.json').read_text());p=Path('/proc')/str(old['pid'])/'stat';state=p.read_text().split(') ',1)[1].split()[0] if p.exists() else None;assert state in (None,'Z','X')
 active=[]
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:argv=(q/'cmdline').read_bytes().decode().split('\0')
  except (OSError,UnicodeDecodeError):continue
  if str(S/'watch_resilient.py') in argv or str(S/'watch_authorized_campaign.py') in argv:active.append(int(q.name))
 assert not active and (C/'local-watch-failure.json').is_file();assert Path(a.recovery).name==a.recovery;F=C/a.recovery;F.mkdir(exist_ok=False)
 save(F/'launch-intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_watch=ref(C/'watch-job.json'),original_failure=ref(C/'local-watch-failure.json'),original_process_state=state,other_watchers=active,first_inspect_not_before_unix=a.not_before,worker=ref(S/'watch_resilient.py'),original_operations=ref(S/'operations.py'),campaign=ref(C/'campaign.json'),no_remote_model_restart=True,read_only_transport_retry_seconds=185,unknown_mutation_retry=False))
 with (F/'stdout.log').open('xb') as out,(F/'stderr.log').open('xb') as err:
  child=subprocess.Popen([sys.executable,'-u',str(S/'watch_resilient.py'),'--directory',str(C),'--recovery',a.recovery,'--first-inspect-not-before',str(a.not_before)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
 save(F/'job.json',dict(pid=child.pid,session_id=child.pid,launcher_pid=os.getpid(),at_unix=time.time(),not_a_GPU_job=True));print(json.dumps(dict(pid=child.pid,recovery=str(F),original_epoch1_untouched=True)))
if __name__=='__main__':main()
