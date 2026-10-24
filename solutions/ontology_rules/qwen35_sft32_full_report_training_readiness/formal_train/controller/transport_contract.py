"""One tiny actual local HTTP/fsync transport fixture; no model or training state."""
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,subprocess,sys,tempfile,time
B=Path(__file__).parent;started=time.monotonic();checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('fixture_fetch',B/'fetch_checkpoint.py');fetch=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetch)
with tempfile.TemporaryDirectory(prefix='fulltext-export-CPU-') as temp:
 root=Path(temp);run=root/'source/formal2ep';cp=run/'epoch-001';target=root/'durable/checkpoints';cp.mkdir(parents=True)
 runfiles=['run_manifest.json','train.py','qwen35_smoke.py','qwen35_relation_cost.py','runtime_contract.py','selected_loss.py','progress.json']
 for n in runfiles[1:]: (run/n).write_bytes(('fixture only '+n+'\n').encode())
 trainer=sha(run/'train.py');fingerprint=dict(schema='fixture-fulltext',source_sha256=trainer,train_sha256='f'*64)
 save(run/'run_manifest.json',dict(fingerprint=fingerprint,epochs=2,total_optimizer_steps=32,warmup_steps=2))
 for n,data in [('training_state.pt',b'not-a-Torch-state-only-transport-fixture'*170),('adapter/adapter_model.safetensors',b'not-model-transport-fixture'*80),('adapter/adapter_config.json',b'{"fixture":true}\n')]:
  p=cp/n;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
 files={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()};save(cp/'checkpoint.json',dict(completed_epochs=1,global_step=16,smoke_only=False,fingerprint=fingerprint,files=files))
 plan=dict(output_dir=str(run),persistent_checkpoint_root=str(target),trainer_sha256=trainer,expanded_train_sha256='f'*64,fingerprint_schema='fixture-fulltext',export_run_files=runfiles,export_trainer_basename='train.py');save(root/'plan.json',plan)
 proc=subprocess.Popen([sys.executable,str(B/'export_checkpoint_server.py'),'--plan',str(root/'plan.json'),'--checkpoint',str(cp),'--deadline',str(time.time()+30)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  ready=json.loads(proc.stdout.readline());assert ready['ready'];checks.append('Actual tiny read-only HTTP manifest returned')
  base=[sys.executable,'--url','http://127.0.0.1:'+str(ready['port']),'--manifest-sha256',ready['manifest_sha256'],'--destination',str(target/'epoch-001'),'--deadline',str(time.time()+20),'--plan',str(root/'plan.json')]
  original_argv=sys.argv[:];original_uid=os.getuid
  def invoke(args):
   sys.argv=args
   with contextlib.redirect_stdout(io.StringIO()) as out:fetch.main()
   return json.loads(out.getvalue())
  try:
   # Synthetic local UID only; real remote UID remains an explicit preflight guard.
   os.getuid=lambda:1006
   ack=invoke(base);assert ack['completed_epochs']==1 and ack['global_step']==16 and ack['fsync_all_files'] and ack['destination_directories_fsynced'] and ack['reopened_final_filename_SHA_all_files'];checks.append('All small checkpoint/run members actual HTTP copied, final-file fsync/reopen SHA and parent fsync')
   for n,r in ack['files'].items():assert sha(target/'epoch-001'/n)==r['sha256'];assert (target/'epoch-001'/n).stat().st_size==r['bytes']
   checks.append('Source and persistent member bytes exact, including state/adapter/metadata')
   try:invoke(base)
   except AssertionError:checks.append('Already ACKed epoch cannot be transferred again')
   else:raise AssertionError('Duplicate ACK accepted')
   bad=base[:];bad[bad.index('--deadline')+1]=str(time.time()-1)
   try:invoke(bad)
   except AssertionError:checks.append('Expired absolute deadline rejected before transfer')
   else:raise AssertionError('Expired deadline accepted')
   bad=base[:];bad[bad.index('--manifest-sha256')+1]='0'*64
   try:invoke(bad)
   except AssertionError:checks.append('Wrong server manifest SHA rejected')
   else:raise AssertionError('Wrong manifest accepted')
  finally:sys.argv=original_argv;os.getuid=original_uid
 finally:
  proc.terminate();proc.communicate(timeout=5)
 # Exercise real same-deadline rejection paths before any subprocess creation.
 save(root/'campaign.json',dict(plan_sha256=sha(root/'plan.json'),cap_seconds=8400,grace_seconds=30,started_unix=time.time()-9000,absolute_deadline_unix=time.time()-600))
 save(root/'entry.json',dict(phase='server',epoch=1,plan_sha256=sha(root/'plan.json'),started_unix=time.time()-800,absolute_deadline_unix=time.time()-200))
 result=subprocess.run([sys.executable,str(B/'io_driver.py'),'--plan',str(root/'plan.json'),'--campaign',str(root/'campaign.json'),'--entry',str(root/'entry.json')],capture_output=True,text=True,timeout=5)
 assert result.returncode!=0 and not (root/'control').exists();checks.append('I/O stage cannot extend expired original campaign')
result=dict(status='PASS_ACTUAL_TINY_HTTP_DURABILITY_AND_CONTROL_CPU_FIXTURE',seconds=time.monotonic()-started,checks=checks,check_count=len(checks),source_sha256=sha(__file__),server_sha256=sha(B/'export_checkpoint_server.py'),fetch_sha256=sha(B/'fetch_checkpoint.py'),io_driver_sha256=sha(B/'io_driver.py'),fixture_payload_not_real_model_or_training_state=True,local_UID_guard_mocked_only=True,real_remote_UID_assets_disk_ACK_not_tested_here=True,no_GPU_or_quality=True)
with (B/'transport-contract-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
