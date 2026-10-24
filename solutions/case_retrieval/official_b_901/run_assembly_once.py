"""Execute one fully bound official-B CPU assembly, with durable intent/terminal."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path


def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(Path(p).parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args()
 plan=json.loads(a.plan.read_text());assert plan['status']=='ACTUAL_REFERENCES_FROZEN_READY_FOR_UNIQUE_CPU_ASSEMBLY'
 assert plan['source']==ref(plan['source']['path']);assert all(isinstance(v,dict) and ref(v['path'])==v for v in plan['references'].values())
 assert plan['CPU_seconds']==60 and plan['quality_or_upload_calls']==0
 d=a.directory.resolve();assert d.parent.is_dir();d.mkdir(exist_ok=False);output=d/'actual-v1'
 fd=os.open(d.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 command=[sys.executable,plan['source']['path'],'--plan',str(a.plan.resolve()),'--output-directory',str(output)]
 save(d/'intent.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),plan=ref(a.plan),command=command,executions=1,outer_seconds=65,term_grace_seconds=5,no_quality_or_upload=True))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 start=time.monotonic();timedout=False
 with (d/'stdout.log').open('xb') as stdout,(d/'stderr.log').open('xb') as stderr:
  p=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
  try:rc=p.wait(timeout=65)
  except subprocess.TimeoutExpired:
   timedout=True;os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
  stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
 terminal=dict(at_utc=datetime.now(timezone.utc).isoformat(),status='complete' if rc==0 and not timedout else 'failed',returncode=rc,timed_out=timedout,seconds=time.monotonic()-start,executions=1,plan=ref(a.plan),files={str(q.relative_to(d)):ref(q) for q in sorted(d.rglob('*')) if q.is_file()})
 save(d/'terminal.json',terminal);print(json.dumps({k:terminal[k] for k in ['status','returncode','timed_out','seconds','executions']}))
 if rc!=0 or timedout:sys.exit(1)
if __name__=='__main__':main()
