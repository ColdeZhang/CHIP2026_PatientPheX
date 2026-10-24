"""Unique necessary held public queries, absolute 300+TERM30 CPU FP32, no fit/bank encode."""
from datetime import datetime,timezone
import hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,value):
 p=Path(path)
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 p=p.resolve().parent
 while True:
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if p.parent==p:break
  p=p.parent
def resources():
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
 return dict(at_utc=datetime.now(timezone.utc).isoformat(),cpu_count=os.cpu_count(),memory=mem,disk_free_bytes=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_queried=False)
def main():
 os.umask(0o077);p=json.loads(Path('held-query-plan.json').read_text());cap=p['budget_seconds']
 assert p['schema']=='grounded-fixed-held-public-queries-v1' and p['authorized'] is True
 assert cap==p['authorized_budget_seconds']==300 and p['grace_seconds']==30 and p['complete_necessary_queries'] and p['query_count']>0
 for name,item in p['files'].items():assert Path(name).stat().st_size==item['bytes'] and sha(name)==item['sha256']
 assert all(not Path(n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'])
 assert not Path(p['output_dir']).exists() and os.environ['CUDA_VISIBLE_DEVICES']=='' and p['execution_device']=='cpu' and os.environ['PATIENTPHEX_DURABILITY_ROOT']==p['storage_parent']
 gate=resources();save('fresh-resource-gate.json',gate)
 assert gate['cpu_count']>=16 and gate['memory']['MemAvailable']>=3*1024**3 and gate['disk_free_bytes']>=2*1024**3 and gate['CUDA_VISIBLE_DEVICES']==''
 start=time.monotonic();utc=time.time();command=p['argv'];timed_out=False
 save('driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),plan_sha256=sha('held-query-plan.json'),command=command,deadline_TERM_unix=utc+cap,deadline_KILL_unix=utc+cap+30,budget_seconds=cap,grace_seconds=30,no_automatic_retry=True))
 with Path('stdout.log').open('xb') as out,Path('stderr.log').open('xb') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
  save('child-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),child_pid=child.pid))
  try:rc=child.wait(timeout=max(.001,start+cap-time.monotonic()))
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(child.pid,signal.SIGTERM)
   try:rc=child.wait(timeout=max(.001,start+cap+30-time.monotonic()))
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
  for f in [out,err]:f.flush();os.fsync(f.fileno())
 report=dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=rc,child_pid=child.pid,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,budget_seconds=cap,grace_seconds=30,no_automatic_retry=True)
 if rc==0 and not timed_out:
  try:
   complete=Path(p['output_dir'])/'complete.json';v=json.loads(complete.read_text())
   assert v['query_count']==p['query_count'] and v['public_bank_rows']==41099 and v['status']=='COMPLETE_FIXED_PUBLIC_HELD_QUERIES'
   assert v['source']['sha256']==p['files'][str(Path(p['remote_root'])/'held_query.py')]['sha256'] and v['no_bank_reencoding'] and v['no_fit']
   assert v['plan']['sha256']==sha('held-query-plan.json')
   for item in [v[k] for k in ['scores','vectors','counts','encoder']]:
    q=Path(item['path']);q.resolve().relative_to(Path(p['output_dir']).resolve());assert q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
   report.update(complete_sha256=sha(complete),complete_necessary_query_members_verified=True,no_quality_performed=True)
  except Exception as error:report['endpoint_error']=dict(type=type(error).__name__,message=str(error))
 report['CPU_resources_at_exit']=resources();report['GPU_used']=False
 report['total_controller_seconds']=time.monotonic()-start;save('wrapper-complete.json',report);print(json.dumps(report),flush=True)
 raise SystemExit(rc or (1 if timed_out or 'endpoint_error' in report else 0))
if __name__=='__main__':main()
