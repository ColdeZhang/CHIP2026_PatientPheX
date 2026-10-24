"""Observe original bounded copy only; no start/reupload. Stop owned source only after managed fetch terminal."""
from pathlib import Path
import json,time,datetime,os,fcntl,sys,shlex,base64,hashlib,traceback
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
A=Path(__file__).parent;F=A/'formal';p=json.loads((F/'run-plan.json').read_text());s=json.loads((F/'execution-state.json').read_text());deadline=s['absolute_deadline_unix'];fetch=json.loads((F/'fetch-job.json').read_text());server=json.loads((F/'server-job.json').read_text());lock=(F/'observer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(path,obj):
 t=path.with_suffix('.tmp')
 with t.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,path)
def un(x):
 if x.get('isError'):raise RuntimeError('MCP actual isError preserved')
 return x.get('structuredContent') or json.loads(x['content'][0]['text'])
def call(name,args):
 c=Client();c.initialize();r=c.request('tools/call',{'name':name,'arguments':args})
 if r.get('isError'):save(F/('observer-mcp-error-'+str(time.time_ns())+'.json'),{'name':name,'actual':r,'no_writer_started':True})
 return un(r)
save(F/'observer-recovery-intent.json',{'at':time.time(),'pid':os.getpid(),'original_fetch':fetch['id'],'original_server':server['id'],'original_deadline':deadline,'no_new_writer':True})
code="from pathlib import Path;import json,base64,hashlib,datetime;p=Path("+repr(p['destination'])+");a=p/'asset-copy-ack.json';q=p/'copy-progress.json';print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'progress':json.loads(q.read_text()) if q.exists() else None,'ACK':{'bytes':a.stat().st_size,'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'base64':base64.b64encode(a.read_bytes()).decode()} if a.exists() else None}))"
while time.time()<deadline+90:
 try:
  r=call('run_command',{'mount_id':p['target_mount_id'],'command':'python3 -c '+shlex.quote(code),'env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':15});assert r['exit_code']==0,r;v=json.loads(r['stdout']);save(F/'observer-latest.json',v)
  print(json.dumps({'at':v['at'],'progress':v['progress'],'ACK':bool(v['ACK'])}),flush=True)
  if v['ACK']:
   a=v['ACK'];b=base64.b64decode(a['base64']);assert len(b)==a['bytes'] and hashlib.sha256(b).hexdigest()==a['sha256'];ack=json.loads(b);manifest=json.loads((A/'expected-transport-manifest.json').read_text())
   assert ack['status']=='verified_SHM_copy_persistent_original_on_100' and ack['files']==manifest['files'] and ack['total_bytes']==p['total_assets_bytes'] and len(ack['files'])==9 and ack['fsync_all_files'] is True and ack['target_is_persistent'] is False
   job=call('read_process_output',{'mount_id':p['target_mount_id'],'process_id':fetch['id'],'stream':'stdout','cursor':0,'limit':30000});save(F/'observer-fetch-terminal.json',job)
   if job['running']:time.sleep(5);continue
   ap=F/'asset-copy-ack.json'
   if ap.exists():assert ap.read_bytes()==b
   else:
    with ap.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
   save(F/'complete.json',{'status':'ALL9_FULL_SHA_FSYNC_R1_PRIVATE_COPY_COMPLETE_SHM_NOT_PERSISTENT','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'copy_deadline':deadline,'ack_sha256':a['sha256'],'no_GPU':True,'monitor_recovery_only_no_restart':True,'fetch_terminal_confirmed':True})
   save(F/'observer-source-stop-intent.json',{'at':time.time(),'original_server':server['id'],'fetch_confirmed_terminal':True})
   stop=call('stop_process',{'mount_id':p['source_mount_id'],'process_id':server['id']});save(F/'server-stopped.json',stop);print(json.dumps({'copy_complete':True,'ACK_SHA':a['sha256'],'original_source_stopped':True}),flush=True);break
  delay=60
 except BaseException as e:
  save(F/('observer-error-'+str(time.time_ns())+'.json'),{'error':repr(e),'traceback':traceback.format_exc(),'at':time.time(),'no_new_writer':True,'original_absolute_deadline':deadline});print(json.dumps({'error':repr(e),'cooldown':180,'original_jobs_untouched':True}),flush=True);delay=180
  if (F/'complete.json').exists():break
 time.sleep(min(delay,max(0,deadline+90-time.time())))
else:save(F/'observer-deadline.json',{'status':'ORIGINAL_COPY_DEADLINE_PASSED_NO_NEW_WINDOW','at':time.time(),'deadline':deadline,'complete':(F/'complete.json').exists()})
