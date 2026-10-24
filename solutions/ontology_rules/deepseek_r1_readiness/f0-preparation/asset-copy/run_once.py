"""Unique R1 private asset copy, shared absolute2400+TERM30; H32 monitor-safe cleanup."""
from pathlib import Path
import sys,json,time,datetime,base64,hashlib,os,shlex
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W))
from run_paired_ack_paths import Run,sha,Client
F=Path(__file__).resolve().parent/'formal'
class CopyRun(Run):
 def call(self,mount,name,args):
  c=Client();c.initialize();result=c.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
  if result.get('isError'):
   self.save(self.root/('mcp-error-%s.json'%time.time_ns()),{'name':name,'mount_id':mount,'actual_result':result,'no_retry':True});raise RuntimeError('MCP tool returned isError; actual receipt retained')
  return result.get('structuredContent') or json.loads(result['content'][0]['text'])
 def watch_copy(self,mount,job):
  cursors={'stdout':0,'stderr':0};directory=self.root/'fetch';directory.mkdir(exist_ok=True)
  while True:
   running=False
   for stream in cursors:
    x=self.call(mount,'read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':30000});cursors[stream]=x['next_cursor'];running|=x['running']
    if x['content']:
     with (directory/(stream+'.log')).open('a') as f:f.write(x['content'])
   (directory/'monitor.json').write_text(json.dumps({'job':job['id'],'running':running,'cursors':cursors,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
   if not running:
    self.save(directory/'process-terminal.json',{'job':job,'running':False,'remote_exit_code_available':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});return
   assert time.time()<self.deadline+90,'Remote terminal uncertain; preserve known original jobs'
   time.sleep(60)
R=CopyRun(F);p=R.plan
assert sha(__file__)==p['controller_sha256'] and sha(W/'run_paired_ack_paths.py')==p['prior_controller_source_sha256']
assert not (F/'execution-state.json').exists()
for r in p['local_evidence']:assert sha(r['path'])==r['sha256']
R.deadline=time.time()+p['copy_budget_seconds'];fetch_terminal=False;job=None;fetch=None
R.save(F/'execution-state.json',{'status':'single_bounded_R1_copy_started','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'no_GPU':True})
try:
 job=R.start_bounded(p['source_mount_id'],['python3','-u',p['source_server'],'--model',p['source_model'],'--source-manifest',p['source_expected'],'--bind-private-ip',p['source_private_IP']],'.',{'CUDA_VISIBLE_DEVICES':''},F/'server-job.json')
 ready=None
 while time.time()<R.deadline and ready is None:
  x=R.call(p['source_mount_id'],'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000})
  for line in x.get('content','').splitlines():
   if line.startswith('{'):ready=json.loads(line)
  if ready is None:assert x['running'],'Source fullSHA/server readiness failed';time.sleep(5)
 assert ready and ready['ready'] and ready['manifest_sha256']==p['expected_transport_manifest_sha256'];R.save(F/'server-ready.json',ready)
 argv=['python3','-u',p['target_fetcher'],'--url','http://'+p['source_private_IP']+':'+str(ready['port']),'--manifest-sha256',ready['manifest_sha256'],'--destination',p['destination'],'--deadline',str(R.deadline),'--expected-files',p['target_expected']]
 fetch=R.start_bounded(p['target_mount_id'],argv,'.',{'CUDA_VISIBLE_DEVICES':''},F/'fetch-job.json');R.watch_copy(p['target_mount_id'],fetch);fetch_terminal=True
 remote_ack=p['destination']+'/asset-copy-ack.json'
 code='from pathlib import Path;import json,base64,hashlib;p=Path('+repr(remote_ack)+');assert p.is_file() and p.stat().st_size<65536;b=p.read_bytes();print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())))'
 actual=R.call(p['target_mount_id'],'run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':15});R.save(F/'actual-ACK-readback.json',actual);assert actual['exit_code']==0;v=json.loads(actual['stdout']);raw=base64.b64decode(v['data']);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 with (F/'asset-copy-ack.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 ack=json.loads(raw);assert ack['status']=='verified_SHM_copy_persistent_original_on_100' and ack['files']==ready['manifest']['files'] and ack['total_bytes']==p['total_assets_bytes'] and len(ack['files'])==9 and ack['fsync_all_files'] and not ack['target_is_persistent']
 R.save(F/'complete.json',{'status':'ALL9_FULL_SHA_FSYNC_R1_PRIVATE_COPY_COMPLETE_SHM_NOT_PERSISTENT','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'copy_deadline':R.deadline,'ack_sha256':sha(F/'asset-copy-ack.json'),'no_GPU':True});print(json.dumps({'status':'copy_complete','ack_sha256':sha(F/'asset-copy-ack.json')}),flush=True)
except BaseException as exc:
 R.save(F/'controller-interruption.json',{'error_type':type(exc).__name__,'error_text':str(exc),'fetch_terminal_confirmed':fetch_terminal,'known_server':job,'known_fetch':fetch,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'action':'preserve_remote_jobs_on_UNKNOWN_no_relaunch'})
 raise
finally:
 # Only a confirmed terminal fetch (success OR failure), or original TTL+30,
 # permits stopping this known owned server. Monitoring failure alone never does.
 if job is not None and (fetch_terminal or time.time()>=R.deadline+30):
  R.save(F/'server-stopped.json',R.call(p['source_mount_id'],'stop_process',{'process_id':job['id']}))
