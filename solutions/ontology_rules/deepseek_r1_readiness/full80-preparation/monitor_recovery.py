"""Restore only read-only F1 monitoring after actual transient RPC failure, fixed original observer deadline."""
from pathlib import Path
import fcntl,time,json,datetime,contextlib,traceback,os,hashlib
from poll_original import main
F=Path(__file__).parent;lock=(F/'monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
DEADLINE=datetime.datetime.fromisoformat('2026-10-24T07:17:36+00:00').timestamp();errors=F/'monitor-errors';errors.mkdir(exist_ok=True)
old=F/'monitor-error-latest.json'
if old.exists():
 b=old.read_bytes();p=errors/('original-'+hashlib.sha256(b).hexdigest()+'.json')
 if not p.exists():p.write_bytes(b)
(F/'monitor-recovery-intent.json').write_text(json.dumps({'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline':DEADLINE,'read_only_no_remote_start_stop':True},indent=2)+'\n')
while time.time()<DEADLINE:
 try:
  with (F/'monitor-recovery.stdout.log').open('a') as out,contextlib.redirect_stdout(out):done=main()
  s=json.loads((F/'monitor-state.json').read_text());print(json.dumps({'at':s['observed_at'],'terminal':done,'rows':(s.get('progress') or {}).get('rows_completed')}),flush=True)
  if done:break
  delay=60
 except BaseException as e:
  stamp=str(time.time_ns());record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(e),'traceback':traceback.format_exc(),'original_job_untouched':True}
  if old.exists():record['actual_tool_error']=json.loads(old.read_text())
  (errors/(stamp+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'read_error':repr(e),'next_read_after_seconds':180,'no_job_mutation':True}),flush=True);delay=180
 time.sleep(min(delay,max(0,DEADLINE-time.time())))
