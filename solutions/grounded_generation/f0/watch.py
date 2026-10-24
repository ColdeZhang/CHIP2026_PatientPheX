"""Single local read-only watcher of the original fresh47 inference intent; never starts NN."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent

def ref(p):p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 p=Path(p)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 d=p.parent
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args();out=Path(a.directory).resolve();assert (out/'start-intent.json').is_file()
 log=out/'watch';log.mkdir(exist_ok=True);lock=(log/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(log/('watcher-'+str(os.getpid())+'.json'),dict(pid=os.getpid(),source=ref(__file__),observe_source=ref(S/'remote.py'),cadence_seconds=180,readonly=True,remote_start_forbidden=True))
 observations=list(out.glob('observation-*.json'));last=max((p.stat().st_mtime for p in observations),default=time.time());next_query=last+180
 while True:
  backoffs=list((out/'transport').glob('offline-backoff-*.json'))
  next_query=max([next_query]+[json.loads(p.read_text())['not_before_unix'] for p in backoffs])
  while time.time()<next_query:time.sleep(min(30,next_query-time.time()))
  stamp=str(time.time_ns());began=time.time()
  try:
   got=subprocess.run([sys.executable,str(S/'remote.py'),'observe','--directory',str(out)],capture_output=True,text=True,timeout=100)
   save(log/(stamp+'.json'),dict(at_unix=time.time(),returncode=got.returncode,stdout=got.stdout,stderr=got.stderr,read_only=True))
  except subprocess.TimeoutExpired as error:
   save(log/(stamp+'.json'),dict(at_unix=time.time(),timeout=True,read_only=True,remote_job_unchanged=True));got=None
  candidates=sorted(out.glob('observation-*.json'))
  if candidates:
   latest=candidates[-1];v=json.loads(latest.read_text())
   if 'wrapper-complete.json' in v:
    save(log/'terminal-observed.json',dict(observation=ref(latest),terminal=v['wrapper-complete.json'],no_collection_or_new_model_started=True));print('TERMINAL_OBSERVED',flush=True);return
  next_query=max(began+180,time.time()+1)
if __name__=='__main__':main()
