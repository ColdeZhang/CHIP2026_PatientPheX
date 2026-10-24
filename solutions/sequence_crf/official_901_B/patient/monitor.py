"""Read only the existing B patient job; never launch or resume remote inference."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import time


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    fd=os.open(str(path.parent), os.O_RDONLY|os.O_DIRECTORY); os.fsync(fd); os.close(fd)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir', required=True)
    ap.add_argument('--client-dir', required=True); ap.add_argument('--once', action='store_true')
    a=ap.parse_args(); root=Path(a.run_dir); plan=json.loads((root/'infer-plan.json').read_text())
    job=json.loads((root/'infer-job.json').read_text()); observed=root/'infer-observations'; observed.mkdir(exist_ok=True)
    lock=(observed/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=datetime.fromisoformat(job['approximate_KILL']).timestamp()+300
    sys.path.insert(0,a.client_dir); from mcp_client import Client
    client=Client(); client.initialize()
    code='''from pathlib import Path
from datetime import datetime,timezone
import json,os
control=Path(CONTROL);output=Path(OUTPUT);files={}
for path in [control/'driver-started.json',control/'child-started.json',control/'wrapper-complete.json',output/'generation-failure.json',output/'complete.json']:
 if path.exists():
  try:files[str(path.relative_to(control.parent.parent))]=json.loads(path.read_text())
  except Exception as e:files[str(path)]=dict(read_error=type(e).__name__)
tails={}
for name in ['stdout.log','stderr.log']:
 path=control/name
 if path.exists():
  with path.open('rb')as f:f.seek(max(0,path.stat().st_size-1600));tails[name]=f.read().decode(errors='replace')
alive={}
for value in files.values():
 if 'pid' in value:
  try:os.kill(value['pid'],0);alive[str(value['pid'])]=True
  except ProcessLookupError:alive[str(value['pid'])]=False
print(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),files=files,log_tails=tails,tracked_pids_alive=alive,read_only=True)))
'''.replace('CONTROL',repr(plan['control'])).replace('OUTPUT',repr(plan['output']))
    compile(code,'remote-read-only','exec')
    save(observed/('watcher-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),dict(pid=os.getpid(),remote_job=job['result']['id'],interval_seconds=180,stop_unix=stop,read_only=True))
    while time.time()<=stop:
        attempts=list(observed.glob('attempt-*.json'))
        latest=max((p.stat().st_mtime for p in attempts),default=0.)
        delay=max(0.,latest+180-time.time())
        if delay:
            if a.once: print(json.dumps(dict(status='backoff',remaining_seconds=delay))); return
            time.sleep(min(60.,delay)); continue
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save(observed/('attempt-'+name+'.json'),dict(at_utc=datetime.now(timezone.utc).isoformat(),remote_job=job['result']['id']))
        receipt=dict(at_utc=datetime.now(timezone.utc).isoformat())
        try:receipt['result']=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id=plan['mount'],command='python3 -c '+shlex.quote(code),cwd='.',timeout_seconds=30)))
        except Exception as e:receipt['client_error']=dict(type=type(e).__name__,message=str(e))
        save(observed/('receipt-'+name+'.json'),receipt)
        terminal=False; summary=dict(receipt=str(observed/('receipt-'+name+'.json')))
        result=receipt.get('result',{})
        if 'client_error' in receipt or result.get('isError'):summary['status']='UNKNOWN: preserve original job, no restart'
        else:
            actual=result['structuredContent']
            if actual['exit_code']!=0:summary.update(status='reader_failed',exit_code=actual['exit_code'])
            else:
                data=json.loads(actual['stdout']); summary.update(observed_at_utc=data['observed_at_utc'],tracked_pids_alive=data['tracked_pids_alive'])
                wrapper=next((v for k,v in data['files'].items() if k.endswith('/wrapper-complete.json')),None)
                records=[]
                for line in data['log_tails'].get('stdout.log','').splitlines():
                    try:records.append(json.loads(line))
                    except json.JSONDecodeError:pass
                summary['last_logged']=records[-1] if records else None
                summary['driver']=next((v for k,v in data['files'].items() if k.endswith('/driver-started.json')),None)
                if wrapper is not None:terminal=True;summary['wrapper']=wrapper
        print(json.dumps(summary),flush=True)
        if terminal:
            save(observed/('terminal-'+name+'.json'),summary);return
        if a.once:return
    print(json.dumps(dict(status='watcher_deadline_reached: actual remote terminal may be UNKNOWN')),flush=True)


if __name__=='__main__':main()
