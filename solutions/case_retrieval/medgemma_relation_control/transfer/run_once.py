"""One inherited bounded HTTP copy; this controller never starts model work."""
from pathlib import Path
import sys,json,time,datetime
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W))
from run_paired_ack_paths import Run,sha
F=Path(__file__).resolve().parent/'formal';R=Run(F);p=R.plan
assert sha(W/'run_paired_ack_paths.py')==p['prior_controller_source_sha256']
assert not (F/'execution-state.json').exists()
R.deadline=time.time()+p['copy_budget_seconds']
R.save(F/'execution-state.json',{'status':'single_copy_started','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'no_GPU':True})
job=R.start_bounded(p['source_mount_id'],['python3','-u',p['source_server'],'--model',p['source_model'],'--source-manifest',p['source_expected']],'.',{},F/'server-job.json')
try:
 ready=None
 while time.time()<R.deadline and ready is None:
  x=R.call(p['source_mount_id'],'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000})
  for line in x.get('content','').splitlines():
   if line.startswith('{'):ready=json.loads(line)
  if not ready:assert x['running'],'Source server stopped before ready';time.sleep(5)
 assert ready and ready['ready'];R.save(F/'server-ready.json',ready)
 argv=['python3','-u',p['preparation']+'/fetch_assets.py','--url','http://'+p['source_private_IP']+':'+str(ready['port']),'--manifest-sha256',ready['manifest_sha256'],'--destination',p['destination'],'--deadline',str(R.deadline),'--expected-files',p['preparation']+'/expected-files.json']
 fetch=R.start_bounded(p['target_mount_id'],argv,'.',{},F/'fetch-job.json');R.watch(p['target_mount_id'],fetch,F/'fetch')
 # Client already independently reopened and hashed every full target; fetch only
 # the immutable small ACK here instead of a second 8.6GB remote hash pass.
 R.get_file(p['target_mount_id'],p['destination']+'/asset-copy-ack.json',F/'asset-copy-ack.json')
 ack=json.loads((F/'asset-copy-ack.json').read_text());assert ack['status']=='persistent_medgemma_assets_verified' and ack['files']==ready['manifest']['files'] and ack['total_bytes']==p['total_assets_bytes']
 R.save(F/'complete.json',{'status':'all19_assets_persisted_on_A100','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_deadline':R.deadline,'ack_sha256':sha(F/'asset-copy-ack.json'),'no_GPU':True})
finally:
 R.save(F/'server-stopped.json',R.call(p['source_mount_id'],'stop_process',{'process_id':job['id']}))
