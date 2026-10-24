"""Detach one local read-only inference watcher, then verify its own session and lock.

No new remote model start here. The watcher only observes the original47 job and
cannot start a model or change a deadline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def ref(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def save(p,obj):
    with p.open('x') as f:
        json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    q=p.parent
    while True:
        fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if q.parent==q:break
        q=q.parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();d=a.directory.resolve()
    package=json.loads((d/'package.json').read_text());plan=json.loads(Path(package['plan']['path']).read_text())
    assert ref(package['plan']['path'])==package['plan'] and plan['inference_authorized'] and plan['expected_rows']==47
    assert (d/'started.json').is_file() and (d/'start-intent.json').is_file()
    watcher=Path(__file__).with_name('watch.py');log=d/'watch';log.mkdir(exist_ok=True)
    observations=sorted(d.glob('observation-*.json'));assert observations
    initial=json.loads(observations[-1].read_text());assert 'wrapper-complete.json' not in initial
    deadline=initial['driver-started.json']['value']['deadline_TERM_unix'];assert time.time()<deadline
    assert not (d/'watch-launch-intent.json').exists(),'Inspect the original PID/lock; never relaunch blindly'
    with (log/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(d/'watch-launch-intent.json',dict(at_unix=time.time(),source=ref(__file__),watcher=ref(watcher),original_model_job=ref(d/'started.json'),existing_model_only=True))
    command=[sys.executable,'-u',str(watcher),'--directory',str(d)]
    with (d/'watch.stdout.log').open('xb') as stdout,(d/'watch.stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
    save(d/'watch-started.json',dict(pid=child.pid,command=command,started_new_session=True,stdio_explicit=True))
    end=time.monotonic()+10
    while time.monotonic()<end:
        intents=list(log.glob('watcher-*.json'))
        if intents:break
        assert child.poll() is None,'Detached scheduler exited; preserve stderr and original job'
        time.sleep(.1)
    candidates=[(p,json.loads(p.read_text())) for p in intents]
    own=[(p,v) for p,v in candidates if v['pid']==child.pid];assert len(own)==1
    p,intent=own[0];assert intent['readonly'] and intent['remote_start_forbidden']
    assert os.getsid(child.pid)==child.pid and child.poll() is None
    held=False
    with (log/'owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
    assert held,'Detached watcher must actually hold the exclusive lock'
    save(d/'watch-ready.json',dict(pid=child.pid,sid=os.getsid(child.pid),actual_intent=ref(p),exclusive_lock_verified=True,
        original_deadline_unix=deadline,remote_job_unchanged=True))
    print(json.dumps(ref(d/'watch-ready.json')))

if __name__=='__main__':main()
