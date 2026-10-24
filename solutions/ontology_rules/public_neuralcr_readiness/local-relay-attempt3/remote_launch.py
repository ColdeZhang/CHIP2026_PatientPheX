from pathlib import Path
import subprocess,sys,json,time,datetime
R=Path(__file__).resolve().parent;name=sys.argv[1];out=R/(name+'.job.json');assert not out.exists();start=time.monotonic()
job={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':['timeout','--signal=TERM','--kill-after=5s','120s',sys.executable,'remote_finalize.py',name]};out.write_text(json.dumps(job)+'\n')
rc=subprocess.run(job['command'],cwd=R).returncode;job.update(returncode=rc,elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());out.write_text(json.dumps(job,indent=2)+'\n');raise SystemExit(rc)
