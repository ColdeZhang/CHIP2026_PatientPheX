"""Detach one local monitor; the original remote campaign owns all model deadlines."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys

def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();D=a.directory;S=Path(__file__).resolve().parent
 assert (D/'epoch-001-job.json').is_file() and not (D/'local-watch-intent.json').exists()
 assert not (D/'watch-launch-intent.json').exists()
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().decode().split('\0')
  except (OSError,UnicodeDecodeError):continue
  if str(S/'watch_authorized_campaign.py') in argv:active.append(int(p.name))
 assert not active
 save(D/'watch-launch-intent.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_watchers=active,source_sha256=hashlib.sha256((S/'watch_authorized_campaign.py').read_bytes()).hexdigest(),detach='DEVNULL stdin, own session, close_fds; lifetime exclusive flock',no_model_restart=True))
 with (D/'watch.stdout.log').open('xb') as out,(D/'watch.stderr.log').open('xb') as err:
  child=subprocess.Popen([sys.executable,'-u',str(S/'watch_authorized_campaign.py'),'--directory',str(D)],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
 save(D/'watch-job.json',dict(pid=child.pid,session_id=child.pid,launcher_pid=os.getpid(),not_a_GPU_job=True))
 print(json.dumps(dict(pid=child.pid,directory=str(D))))
if __name__=='__main__':main()
