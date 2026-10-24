"""One already-reviewed longest full-context engineering probe; no old CPU reruns."""
from pathlib import Path
import sys,json,time,datetime,hashlib,importlib.util
W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');sys.path.insert(0,str(W));from run_paired_ack_paths import Run,sha
E=Path(__file__).resolve().parents[1];F=Path(__file__).resolve().parent/'probe-a800';R=Run(F);p=R.plan
assert sha(__file__)==p['launcher_sha256'] and sha(W/'run_paired_ack_paths.py')==p['prior_controller_source_sha256']
assert not (F/'execution-state.json').exists() and not (F/'probe-job.intent.json').exists(),'Read known job; never repeat launch'
assert p['GPU_budget_seconds']==180 and p['TERM_grace_seconds']==30 and p['CPU_delta_not_to_run'] and p['fresh_base_no_adapter_resume'] and p['no_formal_training_no_quality']
for x in p['local_references'].values():assert sha(x['path'])==x['sha256']
spec=importlib.util.spec_from_file_location('frozen_med_recover',E/'run_probe_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert sha(E/'run_probe_once.py')=='f922e43532a05d55365597c47a4ee981367a5c128a900bf692f5c6c3e7e8debd'
refs={**p['remote_source_files'],**p['reused_source_files']}
code='from pathlib import Path;import hashlib,json,os,shutil;refs='+repr(refs)+';assert all(Path(n).stat().st_size==r["bytes"] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==r["sha256"] for n,r in refs.items());out=Path('+repr(p['probe_output'])+');assert not out.exists();assert out.parent.stat().st_uid==os.getuid()==1045;assert shutil.disk_usage("/dev/shm").free>2*1024**3;print(json.dumps(dict(status="current_source_and_unique_output_verified_no_weights_opened",SHM_free=shutil.disk_usage("/dev/shm").free)))'
R.save(F/'source-preflight.json',R.command(p['mount_id'],code,20));R.gpu_empty('gpu-preflight')
R.deadline=time.time()+180;deadline=R.deadline
R.save(F/'execution-state.json',{'status':'one_longest_fullcontext_probe_running','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'absolute_deadline_unix':deadline,'TERM_grace_seconds':30,'no_formal_training_no_quality':True})
job=R.start_bounded(p['mount_id'],p['probe_argv'],p['cwd'],p['env'],F/'probe-job.json')
try:R.watch(p['mount_id'],job,F/'monitor')
except BaseException as exc:
 R.save(F/'monitor-interruption.json',{'error_type':type(exc).__name__,'error_text':str(exc),'known_job':job,'absolute_deadline_unix':deadline,'action':'preserve_remote_timeout_read_original_job_no_relaunch'});raise
# Model process is confirmed stopped. Subsequent work is read-only recovery;
# retaining its expired compute deadline must not turn small recovery calls into1s.
R.deadline=None;R.gpu_empty('gpu-release')
code='from pathlib import Path;import json;p=Path('+repr(p['probe_output'])+');print(json.dumps(sorted(f.name for f in p.glob("*.json") if f.is_file() and f.stat().st_size<300000)))'
names=json.loads(R.command(p['mount_id'],code,15)['stdout'])
for n in names:m.recover_json(R,p['mount_id'],p['probe_output']+'/'+n,F/'actual'/n)
complete=F/'actual/complete.json';status='engineering_failed_no_quality'
if complete.exists():
 d=json.loads(complete.read_text());assert d['status']=='engineering_probe_complete' and d['source_sha256']==p['probe_source_sha256'] and d['longest_full_tokens']==12824 and d['no_gold_no_quality_score'];status='engineering_probe_complete_no_quality'
R.save(F/'terminal.json',{'status':status,'absolute_deadline_unix':deadline,'TERM_grace_seconds':30,'known_job':job['id'],'managed_process_running':False,'remote_exit_code_not_reported_by_API':True,'actual_json_files':names,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_retry':True,'no_formal_training_no_quality':True})
print(status,flush=True)
