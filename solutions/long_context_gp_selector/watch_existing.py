"""Single detached read-only observer. Never starts, kills, or changes a model job."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
S=Path(__file__).resolve().parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-gp-selector-readiness-v1/formal-v1')
sys.path.insert(0,str(S));from cpu_common import write_json

def main():
 lock=(D/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 latest=max(D.glob('observation-*.json'));value=json.loads(latest.read_text());start=value['job/started.json'];stop=start['KILL_unix']+360
 next_poll=int(latest.stem.split('-')[-1])/1e9+120
 while time.time()<stop:
  write_json(D/'watch-state.json',dict(pid=os.getpid(),read_only=True,next_poll_unix=next_poll,last_observation=str(latest),hard_stop_unix=stop))
  while time.time()<next_poll:time.sleep(min(20,next_poll-time.time()))
  with (D/'watch-errors.log').open('ab') as err:
   try:rc=subprocess.run([sys.executable,str(S/'operations.py'),'observe'],cwd=S.parents[1],stdout=subprocess.DEVNULL,stderr=err,timeout=100).returncode
   except subprocess.TimeoutExpired:rc=124
  latest=max(D.glob('observation-*.json'));value=json.loads(latest.read_text())
  if value.get('job/ended.json'):
   write_json(D/'watch-state.json',dict(pid=os.getpid(),read_only=True,status='ORIGINAL_JOB_TERMINAL_OBSERVED',last_observation=str(latest),terminal=value['job/ended.json']));return
  next_poll=time.time()+(120 if rc==0 else 185)
 write_json(D/'watch-state.json',dict(pid=os.getpid(),read_only=True,status='OBSERVER_STOPPED_WITHOUT_NEW_TERMINAL_NO_MODEL_ACTION',last_observation=str(latest)))
if __name__=='__main__':main()
