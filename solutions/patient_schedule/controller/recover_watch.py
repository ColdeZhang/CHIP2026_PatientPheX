"""Resume the unchanged watcher after confirmed local death; no epoch restart.

Only local lifecycle/log paths change. The frozen operations and remote campaign
remain authoritative. A process lock prevents a second recovery owner.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,sys,time
ORIGINAL_SHA='21afdf5f8e187660c3264c11c1db9afc9ed5f19f3fb2b5b5857f7d37004f53fd'
OPS_SHA='3d680768dd9d805dfae173e0896bd844c64ca770db5b9e151004a45aa51a71f2'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--recovery',required=True);a=ap.parse_args();D=a.directory;F=D/a.recovery;S=Path(__file__).resolve().parent
 assert Path(a.recovery).name==a.recovery and F.is_dir()
 lock=(D/'monitor-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=json.loads((F/'launch-intent.json').read_text());old=json.loads((D/'local-watch-intent.json').read_text())
 assert not Path('/proc/'+str(old['pid'])).exists(), 'Original local watcher is still alive'
 assert digest(S/'watch_authorized_campaign.py')==ORIGINAL_SHA and digest(S/'operations.py')==OPS_SHA
 c=json.loads((D/'campaign.json').read_text());assert c['absolute_deadline_unix']==1792841921.8255548 and time.time()<c['absolute_deadline_unix']
 save(F/'worker-ready.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0),original_pid_absent=True,original_source_sha256=ORIGINAL_SHA,operations_sha256=OPS_SHA,absolute_deadline_unix=c['absolute_deadline_unix'],first_remote_inspect_not_before_unix=launch['first_inspect_not_before_unix'],lock='exclusive monitor-owner.lock held for worker lifetime'))
 while time.time()<launch['first_inspect_not_before_unix']:time.sleep(min(30,launch['first_inspect_not_before_unix']-time.time()))
 source=(S/'watch_authorized_campaign.py').read_text()
 changes={'local-watch-intent.json':a.recovery+'/intent.json','local-watch-events.jsonl':a.recovery+'/events.jsonl','local-watch-complete.json':a.recovery+'/complete.json','local-watch-failure.json':a.recovery+'/failure.json'}
 for oldname,newname in changes.items():
  assert source.count(oldname)==1;source=source.replace(oldname,newname)
 save(F/'rendered-source-proof.json',dict(original_source_sha256=ORIGINAL_SHA,rendered_source_sha256=hashlib.sha256(source.encode()).hexdigest(),changes=changes,method_or_remote_plan_changed=False))
 sys.argv=[str(S/'watch_authorized_campaign.py'),'--directory',str(D)]
 exec(compile(source,str(S/'watch_authorized_campaign.py'),'exec'),dict(__name__='__main__',__file__=str(S/'watch_authorized_campaign.py')))
if __name__=='__main__':main()
