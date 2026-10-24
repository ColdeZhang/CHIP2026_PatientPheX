"""H69 exclusive admission, one shared1800+30 clock including reward wait."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from common import admit,exclusive,now,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--admission',required=True);a=p.parse_args()
    ep=Path(a.execution).resolve();ap=Path(a.admission).resolve();e=json.loads(ep.read_text());ad=json.loads(ap.read_text());base=ep.parent
    admit(ad,ep);assert sha(__file__)==e['sources']['pilot_driver.py']
    exclusive(base/'launch.accepted.json',dict(at_utc=now(),driver_pid=os.getpid(),execution_sha256=sha(ep),admission_sha256=sha(ap),admission_expiry=ad['valid_until_unix']))
    admit(ad,ep);start=time.time();term_at=start+1800;kill_at=start+1830
    cmd=[e['python'],'-u',str(base/'pilot.py'),'--execution',str(ep),'--admission',str(ap),'--TERM-unix',str(term_at),'--KILL-unix',str(kill_at)]
    exclusive(base/'driver-start.json',dict(at_utc=now(),driver_pid=os.getpid(),command=cmd,TERM_unix=term_at,KILL_unix=kill_at))
    with (base/'stdout').open('xb') as out,(base/'stderr').open('xb') as err:
        admit(ad,ep);child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=e['gpu_uuid']))
        exclusive(base/'child.json',dict(at_utc=now(),pid=child.pid,driver_pid=os.getpid()))
        term=killed=False
        try:rc=child.wait(timeout=max(0,term_at-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM);term=True
            try:rc=child.wait(timeout=max(0,kill_at-time.time()))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);killed=True;rc=child.wait()
    exclusive(base/'driver-complete.json',dict(at_utc=now(),driver_pid=os.getpid(),child_pid=child.pid,returncode=rc,TERM_sent=term,KILL_sent=killed,seconds=time.time()-start,TERM_unix=term_at,KILL_unix=kill_at))
    if rc:raise SystemExit(rc)
if __name__=='__main__':main()
