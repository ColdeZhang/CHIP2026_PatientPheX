"""One inherited bounded HTTP copy; this controller never starts model work."""
from pathlib import Path
import sys,json,time,datetime,base64,hashlib,os
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
 remote_ack=p['destination']+'/asset-copy-ack.json'
 code='from pathlib import Path;import json,base64,hashlib;p=Path('+repr(remote_ack)+');assert p.is_file() and p.stat().st_size<65536;b=p.read_bytes();print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())))'
 actual=R.command(p['target_mount_id'],code,15);R.save(F/'actual-ACK-readback.json',actual);v=json.loads(actual['stdout']);raw=base64.b64decode(v['data'])
 assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 with (F/'asset-copy-ack.json').open('xb') as ack_file:ack_file.write(raw);ack_file.flush();os.fsync(ack_file.fileno())
 ack=json.loads((F/'asset-copy-ack.json').read_text());assert ack['status']=='verified_SHM_copy_persistent_original_on_100' and ack['files']==ready['manifest']['files'] and ack['total_bytes']==p['total_assets_bytes']
 R.save(F/'complete.json',{'status':'all19_assets_SHA_verified_in_A800_SHM_original_stays_persistent_on_100','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_deadline':R.deadline,'ack_sha256':sha(F/'asset-copy-ack.json'),'no_GPU':True})
finally:
 R.save(F/'server-stopped.json',R.call(p['source_mount_id'],'stop_process',{'process_id':job['id']}))
