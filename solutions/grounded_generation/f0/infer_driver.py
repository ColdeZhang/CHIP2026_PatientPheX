"""Unique fresh47 native inference with original absolute 2100+TERM30 budget."""
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
 def query(argument):return subprocess.check_output(['nvidia-smi',argument,'--format=csv,noheader,nounits'],text=True,timeout=10)
 return dict(at_utc=datetime.now(timezone.utc).isoformat(),gpu=query('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'),disk_free_bytes=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize)
def main():
 os.umask(0o077);p=json.loads(Path('infer-plan.json').read_text());cap=p['budget_seconds']
 assert p['schema']=='grounded-held47-inference-v1' and p['inference_authorized'] is True
 assert cap==p['authorized_budget_seconds']==2100 and p['grace_seconds']==30 and p['expected_rows']==47 and p['all_rows_fresh']
 for name,item in p['files'].items():assert Path(name).stat().st_size==item['bytes'] and sha(name)==item['sha256']
 assert all(not Path(n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'])
 assert not Path(p['output_dir']).exists() and os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and os.environ['PATIENTPHEX_DURABILITY_ROOT']==p['storage_parent']
 gate=resources();save('fresh-resource-gate.json',gate);target=[s.split(',') for s in gate['gpu'].splitlines() if p['gpu_uuid'] in s]
 assert len(target)==1 and int(target[0][2])<=10 and int(target[0][3])>=79000 and p['gpu_uuid'] not in gate['processes'] and gate['disk_free_bytes']>=2*1024**3
 start=time.monotonic();utc=time.time();command=p['argv'];timed_out=False
 save('driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),plan_sha256=sha('infer-plan.json'),command=command,deadline_TERM_unix=utc+cap,deadline_KILL_unix=utc+cap+30,budget_seconds=cap,grace_seconds=30,no_automatic_retry=True))
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
   assert v['rows']==47 and v['documents']==16 and v['status']=='COMPLETE_FRESH47_RAW_GENERATION_NO_QUALITY'
   assert v['source']['sha256']==p['assets']['runner']['sha256'] and v['no_regeneration_or_answer_repair']
   assert v['plan']['sha256']==sha('infer-plan.json')
   for item in [v['responses']]+v['raw_refs']+v['generation_refs']:
    q=Path(item['path']);q.resolve().relative_to(Path(p['output_dir']).resolve());assert q.stat().st_size==item['bytes'] and sha(q)==item['sha256']
   report.update(complete_sha256=sha(complete),complete47_actual_members_verified=True,no_quality_performed=True)
  except Exception as error:report['endpoint_error']=dict(type=type(error).__name__,message=str(error))
 try:report['gpu_release']=resources();report['target_has_no_process']=p['gpu_uuid'] not in report['gpu_release']['processes']
 except Exception as error:report['release_unknown']=type(error).__name__
 report['total_controller_seconds']=time.monotonic()-start;save('wrapper-complete.json',report);print(json.dumps(report),flush=True)
 raise SystemExit(rc or (1 if timed_out or 'endpoint_error' in report else 0))
if __name__=='__main__':main()
