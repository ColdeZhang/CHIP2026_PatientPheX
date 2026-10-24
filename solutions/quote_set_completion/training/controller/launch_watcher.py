"""Detach one local authorized scheduler, then verify its own session and lock.

No new remote model start here. The watcher owns existing campaign stages and
never retries a mutation with an unknown outcome.
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
    plan=json.loads((d/'controller-plan.json').read_text());campaign=json.loads((d/'campaign.json').read_text())
    watcher=Path(__file__).with_name('watch_authorized_campaign.py')
    assert ref(watcher)==plan['local_watcher'] and plan['formal_training_authorized']
    assert time.time()<campaign['absolute_deadline_unix'] and (d/'epoch-001-job.json').is_file()
    assert not (d/'watch-launch-intent.json').exists(),'Inspect the original PID/lock; never relaunch blindly'
    with (d/'monitor-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(d/'watch-launch-intent.json',dict(at_unix=time.time(),source=ref(__file__),watcher=ref(watcher),campaign=ref(d/'campaign.json'),existing_model_only=True))
    command=[sys.executable,'-u',str(watcher),'--directory',str(d)]
    with (d/'watch.stdout.log').open('xb') as stdout,(d/'watch.stderr.log').open('xb') as stderr:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
    save(d/'watch-started.json',dict(pid=child.pid,command=command,started_new_session=True,stdio_explicit=True))
    end=time.monotonic()+10
    while time.monotonic()<end:
        intents=list(d.glob('local-watch-intent-*.json'))
        if intents:break
        assert child.poll() is None,'Detached scheduler exited; preserve stderr and original job'
        time.sleep(.1)
    candidates=[(p,json.loads(p.read_text())) for p in intents]
    own=[(p,v) for p,v in candidates if v['pid']==child.pid];assert len(own)==1
    p,intent=own[0];assert intent['sid']==child.pid and intent['exclusive_lock']
    assert os.getsid(child.pid)==child.pid and child.poll() is None
    held=False
    with (d/'monitor-owner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
    assert held,'Detached watcher must actually hold the exclusive lock'
    save(d/'watch-ready.json',dict(pid=child.pid,sid=os.getsid(child.pid),actual_intent=ref(p),exclusive_lock_verified=True,
        original_deadline_unix=campaign['absolute_deadline_unix'],remote_job_unchanged=True))
    print(json.dumps(ref(d/'watch-ready.json')))

if __name__=='__main__':main()
