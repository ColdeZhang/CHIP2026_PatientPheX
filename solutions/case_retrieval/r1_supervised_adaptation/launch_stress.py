"""Remote timeout owner; preserve diagnostics and terminate only its own child."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());directory=Path(a.plan).parent
 assert not (directory/'launch-started.json').exists()
 assert plan['budget_seconds']==420 and plan['grace_seconds']==30
 start=time.time();save(directory/'launch-started.json',dict(start_epoch=start,deadline_epoch=start+420,pid=os.getpid(),plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()))
 with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
  child=subprocess.Popen([sys.executable,'-u',str(directory/'stress_probe.py'),'--plan',a.plan],stdout=stdout,stderr=stderr,start_new_session=True)
  save(directory/'child.json',dict(pid=child.pid,start_epoch=time.time()))
  expired=False
  try:rc=child.wait(timeout=420)
  except subprocess.TimeoutExpired:
   expired=True;os.killpg(child.pid,15)
   try:rc=child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,9);rc=child.wait()
 files={}
 for f in sorted(directory.rglob('*')):
  if f.is_file() and f.name!='terminal.json':files[str(f.relative_to(directory))]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 save(directory/'terminal.json',dict(exit_code=rc,timeout_triggered=expired,wall_seconds=time.time()-start,child_pid=child.pid,files=files,probe_state_discarded=True))
 print(json.dumps(dict(exit_code=rc,wall_seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
