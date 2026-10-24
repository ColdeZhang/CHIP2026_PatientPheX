"""Sole detached read-only watcher for an existing core or inference intent."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    temp=path.with_suffix(path.suffix+'.next')
    with temp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main(directory,mode='core'):
    directory=Path(directory);root=directory/(mode+'-watch');root.mkdir(exist_ok=True)
    assert (directory/('GPU-start-intent.json' if mode=='core' else 'downstream-v1/infer-start-intent.json')).exists()
    lock=(root/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    save(root/'ready.json',dict(at_utc=now(),pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),start_ticks=int(stat[19]),exclusive_lock=True,read_only=True))
    command=([sys.executable,str(Path(__file__).resolve().parents[1]/'control.py'),'--directory',str(directory),'inspect'] if mode=='core' else
             [sys.executable,str(Path(__file__).with_name('stages.py')),'--directory',str(directory),'infer_status'])
    failures=0
    while True:
        tick=time.monotonic()
        try:
            r=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=95)
            if r.returncode:raise RuntimeError('read-only inspect failed')
            value=json.loads(r.stdout);failures=0
            status=dict(at_utc=now(),phase=value.get('driver-progress.json',{}).get('phase'),
                        reference=value.get('reference/progress.json'),probe=value.get('probe/progress.json'),
                        formal=value.get('formal/progress.json'),done='driver-result.json' in value,
                        original_job_restarted=False,read_only=True)
            if mode=='infer':
                status.update(phase='infer80',inference=value.get('native/progress.json'),failure=value.get('native/failure.json'))
            save(root/'latest.json',status)
            if status['done']:
                save(root/'complete.json',dict(at_utc=now(),driver_result=value['driver-result.json'],original_job_restarted=False));return
            delay=60
        except Exception as e:
            failures+=1;delay=30 if failures==1 else 185
            save(root/'last-read-failure.json',dict(at_utc=now(),error_type=type(e).__name__,consecutive=failures,retry_seconds=delay,job_state_unknown_not_failed=True))
            tick=time.monotonic()  # Retry delay starts after the failed read.
        time.sleep(max(1,delay-(time.monotonic()-tick)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--mode',choices=['core','infer'],default='core');a=p.parse_args();main(a.directory,a.mode)
