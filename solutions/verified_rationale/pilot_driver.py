"""H69 prebound launch admission plus a separate2700+30 native budget."""
import argparse,datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(Path(p).parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--admission',required=True);z=p.parse_args();ep=Path(z.execution).resolve();ap=Path(z.admission).resolve();e=json.loads(ep.read_text());a=json.loads(ap.read_text());base=ep.parent
 def admit():assert a['valid_from_unix']-5<=time.time()<a['valid_until_unix'] and sha(ep)==a['execution_sha256']
 admit();assert sha(__file__)==e['sources'][Path(__file__).name]
 # Atomic exclusive publication is also the only valid fence point for UNKNOWN recovery.
 write(base/'launch.accepted.json',dict(at_utc=now(),driver_pid=os.getpid(),execution_sha256=sha(ep),admission_sha256=sha(ap),resource_PASS=False,admission_expiry=a['valid_until_unix']))
 admit();start=time.time();cmd=[e['python'],'-u',str(base/'pilot.py'),'--execution',str(ep),'--admission',str(ap)];write(base/'driver-start.json',dict(at_utc=now(),driver_pid=os.getpid(),command=cmd,TERM_unix=start+2700,KILL_unix=start+2730))
 with (base/'stdout').open('xb') as out,(base/'stderr').open('xb') as err:
  admit();child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=e['gpu_uuid']));write(base/'child.json',dict(at_utc=now(),pid=child.pid,driver_pid=os.getpid()))
  term=False;killed=False
  try:rc=child.wait(timeout=max(0,start+2700-time.time()))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM);term=True
   try:rc=child.wait(timeout=max(0,start+2730-time.time()))
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);killed=True;rc=child.wait()
 write(base/'driver-complete.json',dict(at_utc=now(),driver_pid=os.getpid(),child_pid=child.pid,returncode=rc,TERM_sent=term,KILL_sent=killed,seconds=time.time()-start))
 if rc:raise SystemExit(rc)
if __name__=='__main__':main()
