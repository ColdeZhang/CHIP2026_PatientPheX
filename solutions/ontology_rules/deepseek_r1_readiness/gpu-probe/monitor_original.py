"""Low frequency read-only monitoring of the original unique probe."""
from pathlib import Path
import json,subprocess,sys,time,fcntl
G=Path(__file__).parent;lock=(G/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=json.loads((G/'start-intent.json').read_text())['at']+270
while time.time()<deadline:
 begun=time.time()
 try:
  p=subprocess.run([sys.executable,str(G/'poll_original.py')],capture_output=True,text=True,timeout=90)
  if p.returncode:
   with (G/'monitor-errors.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,original_remote_job_unchanged=True))+'\n')
   print('READONLY_MONITOR_ERROR_STOP_REMOTE_JOB_UNCHANGED',flush=True);break
  x=json.loads(p.stdout);print(json.dumps(x),flush=True)
  if x['terminal']:break
 except Exception as e:
  with (G/'monitor-errors.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),error=repr(e),original_remote_job_unchanged=True))+'\n')
  print('READONLY_MONITOR_UNKNOWN_STOP_REMOTE_JOB_UNCHANGED',flush=True);break
 until=min(begun+30,deadline)
 while time.time()<until:time.sleep(min(15,until-time.time()))
