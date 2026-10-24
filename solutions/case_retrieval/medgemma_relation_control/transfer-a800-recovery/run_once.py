"""One separately authorized bounded resume; monitor failure never kills healthy transfer."""
from pathlib import Path
import sys,json,time,datetime,base64,hashlib,os
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W))
from run_paired_ack_paths import Run,sha
F=Path(__file__).resolve().parent/'formal';R=Run(F);p=R.plan
assert sha(__file__)==p['controller_sha256'] and sha(W/'run_paired_ack_paths.py')==p['prior_controller_source_sha256']
assert not (F/'execution-state.json').exists()
for r in p['local_evidence']:
 assert sha(r['path'])==r['sha256']
R.deadline=time.time()+p['copy_budget_seconds'];fetch_terminal=False;job=None;fetch=None
R.save(F/'execution-state.json',{'status':'single_recovery_copy_started','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'no_GPU':True})
try:
 job=R.start_bounded(p['source_mount_id'],['python3','-u',p['source_server'],'--model',p['source_model'],'--source-manifest',p['source_expected'],'--resume-state',p['remote_recovery']+'/resume-state.json','--resume-state-sha',p['resume_state_sha256']],'.',{},F/'server-job.json')
 ready=None
 while time.time()<R.deadline and ready is None:
  x=R.call(p['source_mount_id'],'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000})
  for line in x.get('content','').splitlines():
   if line.startswith('{'):ready=json.loads(line)
  if not ready:assert x['running'],'Source server stopped before ready';time.sleep(5)
 assert ready and ready['ready'] and ready['manifest_sha256']==p['original_manifest_sha256'];R.save(F/'server-ready.json',ready)
 argv=['python3','-u',p['target_fetcher'],'--url','http://'+p['source_private_IP']+':'+str(ready['port']),'--manifest-sha256',ready['manifest_sha256'],'--destination',p['destination'],'--deadline',str(R.deadline),'--expected-files',p['preparation']+'/expected-files.json','--resume-state',p['remote_recovery']+'/resume-state.json','--resume-state-sha',p['resume_state_sha256']]
 fetch=R.start_bounded(p['target_mount_id'],argv,'.',{},F/'fetch-job.json');R.watch(p['target_mount_id'],fetch,F/'fetch');fetch_terminal=True
 remote_ack=p['destination']+'/asset-copy-ack.json'
 code='from pathlib import Path;import json,base64,hashlib;p=Path('+repr(remote_ack)+');assert p.is_file() and p.stat().st_size<65536;b=p.read_bytes();print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())))'
 actual=R.command(p['target_mount_id'],code,15);R.save(F/'actual-ACK-readback.json',actual);v=json.loads(actual['stdout']);raw=base64.b64decode(v['data'])
 assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 with (F/'asset-copy-ack.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 ack=json.loads(raw);assert ack['status']=='verified_SHM_copy_persistent_original_on_100' and ack['files']==ready['manifest']['files'] and ack['total_bytes']==p['total_assets_bytes'] and ack['resume_state_sha256']==p['resume_state_sha256']
 assert set(ack['verification_mode_by_file'])==set(ack['files']) and len(ack['files'])==19
 R.save(F/'complete.json',{'status':'all19_assets_verified_across_original_and_recovery_windows_SHM_only','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recovery_deadline':R.deadline,'ack_sha256':sha(F/'asset-copy-ack.json'),'no_GPU':True})
except BaseException as exc:
 R.save(F/'controller-interruption.json',{'error_type':type(exc).__name__,'error_text':str(exc),'fetch_terminal_confirmed':fetch_terminal,'known_server':job,'known_fetch':fetch,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'action':'preserve_remote_jobs_when_fetch_state_unknown_no_relaunch'})
 raise
finally:
 # Remote timeout is authoritative. A failed monitor/initialize is NOT a request
 # to abort transfer. Keep both known IDs for read-only recovery; never relaunch.
 if job is not None and (fetch_terminal or time.time()>=R.deadline+30):
  R.save(F/'server-stopped.json',R.call(p['source_mount_id'],'stop_process',{'process_id':job['id']}))
