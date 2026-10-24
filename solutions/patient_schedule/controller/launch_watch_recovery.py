"""Detach only the recovery monitor from the short tool launcher; no model call."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--last-inspect',type=Path,required=True);ap.add_argument('--recovery',required=True);a=ap.parse_args();D=a.directory;S=Path(__file__).resolve().parent
 assert Path(a.recovery).name==a.recovery;old=json.loads((D/'local-watch-intent.json').read_text());assert not Path('/proc/'+str(old['pid'])).exists()
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().decode().split('\0')
  except (OSError,UnicodeDecodeError):continue
  if any(str(S/n) in argv for n in ['recover_watch.py','watch_authorized_campaign.py']):active.append(int(p.name))
 assert not active,'Another actual local watcher exists: '+repr(active)
 assert a.last_inspect.parent.resolve()==D.resolve() and a.last_inspect.is_file()
 stamp=int(a.last_inspect.name.removeprefix('inspect-').removesuffix('.json'))/1e9
 latest=json.loads(a.last_inspect.read_text());assert latest['1']['driver-started.json']['plan_sha256']==sha(D/'controller-plan.json')
 F=D/a.recovery;F.mkdir(exist_ok=False)
 save(F/'launch-intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_watch_pid=old['pid'],original_PID_absent=True,other_watchers=active,original_intent_sha256=sha(D/'local-watch-intent.json'),original_events_sha256=sha(D/'local-watch-events.jsonl'),last_root_inspect=dict(path=str(a.last_inspect),sha256=sha(a.last_inspect)),first_inspect_not_before_unix=max(time.time(),stamp+120),worker_source_sha256=sha(S/'recover_watch.py'),launcher_source_sha256=sha(__file__),old_exec73046_result='owner and root tools both returned Unknown process id; no failure/completion receipt exists',cause='Unknown; missing local process/session alone does not establish why it ended',no_remote_restart=True,no_deadline_reset=True,detach_policy='new session, stdin DEVNULL, dedicated stdout/stderr files, close_fds'))
 with (F/'stdout.log').open('xb') as out,(F/'stderr.log').open('xb') as err:
  child=subprocess.Popen([sys.executable,'-u',str(S/'recover_watch.py'),'--directory',str(D),'--recovery',a.recovery],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
 save(F/'job.json',dict(pid=child.pid,session_id=child.pid,launcher_pid=os.getpid(),launch_unix=time.time(),not_a_GPU_job=True))
 print(json.dumps(dict(pid=child.pid,directory=str(F),original_epoch1_untouched=True)))
if __name__=='__main__':main()
