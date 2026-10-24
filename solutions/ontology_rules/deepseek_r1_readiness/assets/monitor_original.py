from pathlib import Path
import subprocess,sys,time,json,os,fcntl
A=Path(__file__).parent
deadline=json.loads((A/'intent.json').read_text())['absolute_deadline_epoch']
lock=(A/'local-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while time.time()<deadline+70:
 began=time.time()
 try:
  p=subprocess.run([sys.executable,str(A/'poll_original.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=140,text=True)
  if p.returncode:
   with (A/'monitor-errors.jsonl').open('a') as f:f.write(json.dumps({'observed_epoch':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'remote_original_job_not_stopped_or_restarted':True})+'\n')
   print(json.dumps({'status':'READONLY_MONITOR_STOPPED_ON_ERROR_OR_UNKNOWN','remote_job_unchanged':True}),flush=True);break
  r=json.loads(p.stdout);progress=r.get('progress',{})
  print(json.dumps({'running':r['running'],'observed_epoch':r['observed_epoch'],'complete_status':r.get('complete_status'),'weights':{k:v for k,v in progress.items() if k.endswith('.safetensors')},'completed_files':sum(v.get('status') in ('REUSED_FULL_SHA_MATCH','DOWNLOADED_FULL_SHA_MATCH') for v in progress.values())}),flush=True)
  if r.get('complete_status') or r['running'] is False:break
 except Exception as ex:
  with (A/'monitor-errors.jsonl').open('a') as f:f.write(json.dumps({'observed_epoch':time.time(),'error':repr(ex),'remote_original_job_not_stopped_or_restarted':True})+'\n')
  print(json.dumps({'status':'READONLY_MONITOR_STOPPED_ON_ERROR_OR_UNKNOWN','remote_job_unchanged':True}),flush=True);break
 # Low frequency original-job monitoring, using short interruptible sleeps only.
 until=min(began+120,deadline+70)
 while time.time()<until:time.sleep(min(30,until-time.time()))
