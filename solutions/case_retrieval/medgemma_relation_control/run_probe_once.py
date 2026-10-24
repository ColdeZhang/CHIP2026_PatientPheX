"""Use existing transport receipt and runtime delta, then one bounded Med probe.

Does not train formal epochs, evaluate quality, fetch missing assets, or retry GPU.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,sys,time,datetime
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W))
from run_paired_ack_paths import Run,sha

def recover_json(R,mount,remote,destination):
 code='from pathlib import Path;import base64,hashlib,json;p=Path('+repr(remote)+');assert p.is_file() and p.stat().st_size<300000;b=p.read_bytes();print(json.dumps(dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())))'
 actual=R.command(mount,code,15);v=json.loads(actual['stdout']);raw=base64.b64decode(v['data'])
 assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return json.loads(raw)

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();F=Path(a.directory);R=Run(F);p=R.plan
 assert sha(__file__)==p['launcher_sha256']
 assert sha(W/'run_paired_ack_paths.py')==p['common_controller_sha256']
 assert not (F/'probe-job.intent.json').exists() and not (F/'runtime-delta.intent.json').exists(),'Read original known jobs/receipts; no repeat of this entry'
 for field in ['asset_copy_ack','sequence_final_release','source_manifest']:
  r=p[field];assert sha(r['path'])==r['sha256']
 release=json.loads(Path(p['sequence_final_release']['path']).read_text())['gpu_release']
 assert release['target']==p['gpu_uuid'] and release['target_has_no_process']
 ack=json.loads(Path(p['asset_copy_ack']['path']).read_text())
 assert ack['status']=='verified_SHM_copy_persistent_original_on_100' and ack['target_mount_id']==p['mount_id'] and ack['target_directory']==p['model'] and ack['total_bytes']==8639802098
 # Exact tiny source and actual completed model-copy ACK bind the server state.
 code='from pathlib import Path;import hashlib,json;refs='+repr(p['remote_source_files'])+';assert all(Path(n).stat().st_size==r["bytes"] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==r["sha256"] for n,r in refs.items());p=Path('+repr(p['model']+'/asset-copy-ack.json')+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(p['asset_copy_ack']['sha256'])+';print("exact native source and copyACK verified")'
 R.save(F/'source-preflight.json',R.command(p['mount_id'],code,20))
 # The new server uses the same native libraries; CPU-only delta checks its Python
 # minor version, actual longest tokenizer path, and meta language-LoRA topology.
 env={**p['env'],'CUDA_VISIBLE_DEVICES':''}
 command=['env',*[k+'='+v for k,v in env.items()],'/usr/bin/timeout','--signal=TERM','--kill-after=10s','45',*p['runtime_delta_argv']]
 R.save(F/'runtime-delta.intent.json',{'argv':command,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_weights_no_GPU':True})
 result=R.call(p['mount_id'],'run_command',{'command':shlex.join(command),'cwd':'.','timeout_seconds':55});R.save(F/'runtime-delta-command.json',result);assert result['exit_code']==0,'CPU delta failed; no GPU launch'
 delta=recover_json(R,p['mount_id'],p['runtime_delta_output'],F/'runtime-delta.json');assert delta['status']=='CPU_runtime_delta_pass' and not delta['CUDA_initialized']
 R.gpu_empty('gpu-preflight')
 R.deadline=time.time()+180
 R.save(F/'execution-state.json',{'status':'one_probe_started','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':R.deadline,'TERM_grace_seconds':30,'no_formal_training_or_quality':True})
 job=R.start_bounded(p['mount_id'],p['probe_argv'],'.',p['env'],F/'probe-job.json')
 try:R.watch(p['mount_id'],job,F/'monitor')
 except BaseException as exc:
  R.save(F/'monitor-interruption.json',{'error_type':type(exc).__name__,'error_text':str(exc),'known_probe_job':job,'original_deadline_unix':R.deadline,'action':'preserve_remote_timeout_and_known_job_read_before_any_recovery_no_relaunch'});raise
 R.gpu_empty('gpu-release')
 # Always retain failure markers, even when the probe did not reach completion.
 code='from pathlib import Path;import json;p=Path('+repr(p['probe_output'])+');print(json.dumps(sorted(f.name for f in p.glob("*.json") if f.is_file() and f.stat().st_size<300000)))'
 names=json.loads(R.command(p['mount_id'],code,15)['stdout'])
 for n in names:recover_json(R,p['mount_id'],p['probe_output']+'/'+n,F/'actual'/n)
 complete=F/'actual/complete.json';status='probe_complete' if complete.exists() else 'probe_failed_no_quality'
 R.save(F/'terminal.json',{'status':status,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_deadline_unix':R.deadline,'probe_job_id':job['id'],'actual_json_files':names,'managed_process_running':False,'remote_model_exit_code_available':False,'no_retry':True,'no_formal_training_or_score':True})
 print(status,flush=True)
if __name__=='__main__':main()
