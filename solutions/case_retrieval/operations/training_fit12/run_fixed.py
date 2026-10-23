#!/usr/bin/env python3
"""Single bounded train-only fixed12 inference; existing jobs never restart."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True);a=p.parse_args();b=a.directory.resolve()
 sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r
 plan=json.loads((b/'run-plan.json').read_text());assert sha(__file__)==plan['driver_sha256'];r.M=plan['mount_id']
 assert plan['hard_cap_seconds']==300 and plan['argv'][:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s','300']
 assert not (b/'job.json').exists(),'Recover original managed job, never repeat launch'
 for v in plan['local_references'].values():assert Path(v['path']).stat().st_size==v['bytes'] and sha(v['path'])==v['sha256']
 def save(n,x):
  with (b/n).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 def readtool(n,kw):
  for attempt in range(3):
   try:return r.call(n,kw)
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 def gpu(name):
  code="import subprocess,time,json;u="+repr(plan['gpu_uuid'])+";apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(x.split(',')[0].strip()==u for x in apps.splitlines());devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);target=next(x for x in devices.splitlines() if x.split(',')[0].strip()==u);assert int(target.split(',')[2].strip().split()[0])>=70000;print(json.dumps({'gpu_uuid':u,'target_empty':True,'target':target,'compute_apps':apps,'server_epoch':time.time()}))"
  x=readtool('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=plan['cwd'],timeout_seconds=20));save(name,x);assert x['exit_code']==0
 code="from pathlib import Path;import hashlib,json,shutil;expected="+repr(plan['remote_sha256'])+";observed={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert observed==expected;assert not list(Path('full64-training-fit12').glob('epoch2.responses*'));print(json.dumps({'actual_sha256':observed,'free_disk_bytes':shutil.disk_usage('.').free}))"
 x=readtool('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=plan['cwd'],timeout_seconds=30));save('source-preflight.json',x);assert x['exit_code']==0
 gpu('gpu-preflight.json')
 job=r.call('start_process',dict(command=shlex.join(plan['argv']),cwd=plan['cwd'],env=plan['env']));save('job.json',dict(job=job,plan_sha256=sha(b/'run-plan.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(json.dumps({'started':job['id'],'cap_seconds':300}),flush=True)
 cursors=dict(stdout=0,stderr=0)
 while True:
  running=False
  for stream in cursors:
   x=readtool('read_process_output',dict(process_id=job['id'],stream=stream,cursor=cursors[stream],limit=20000));cursors[stream]=x['next_cursor'];running|=x['running']
   if x['content']:
    with (b/(stream+'.log')).open('a') as f:f.write(x['content'])
    if stream=='stdout':print(x['content'],flush=True)
  (b/'monitor.json').write_text(json.dumps(dict(job=job['id'],running=running,cursors=cursors,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
  if not running:break
  time.sleep(10)
 gpu('gpu-release.json')
 for suffix in ['.jsonl','.checkpoint.json','.manifest.json','.loading.json']:
  n='epoch2.responses'+suffix;x=readtool('read_file',dict(path=plan['cwd']+'/full64-training-fit12/'+n,offset=0,limit=200000));assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(b/n).write_text(x['content'])
 raw=[json.loads(x) for x in (b/'epoch2.responses.jsonl').read_text().splitlines()];meta=json.loads((b/'epoch2.responses.manifest.json').read_text());fp=meta['fingerprint']
 assert len(raw)==len({x['row_id'] for x in raw})==12 and {x['row_id']:str(x['pmc_id']) for x in raw}==plan['row_documents']
 assert {x['row_id']:x['input_tokens'] for x in raw}==plan['prefix_tokens']
 assert all(x['removed_demonstrations']==0 and isinstance(x['hit_token_limit'],bool) for x in raw)
 assert fp['source_sha256']==plan['runner_sha256'] and fp['prompts_sha256']==plan['prompts_sha256'] and fp['source_manifest_sha256']==plan['model_source_manifest_sha256']
 assert fp['adapter_files']==plan['adapter_files'] and fp['configuration']==plan['runner_configuration']
 assert meta['expected_rows']==12 and meta['output_sha256']==sha(b/'epoch2.responses.jsonl') and meta['elapsed_seconds']<300
 assert meta['base_parameters']==8953803264 and meta['total_parameters']==8997081600 and meta['eos_token_ids']==[248044,248046]
 save('predictions-freeze.json',dict(status='complete_fixed12_training_predictions_frozen_before_fit_score',plan_sha256=sha(b/'run-plan.json'),raw_sha256=sha(b/'epoch2.responses.jsonl'),manifest_sha256=sha(b/'epoch2.responses.manifest.json'),release_sha256=sha(b/'gpu-release.json'),rows=12,actual_original_prefix_tokens_exact=True,elapsed_seconds=meta['elapsed_seconds'],raw_generation_seconds=sum(x['batch_seconds'] for x in raw),hit_token_limits=sum(x['hit_token_limit'] for x in raw),no_originaldev_or_outergold=True))
 print('FIXED12_COMPLETE_RELEASED_FROZEN',flush=True)
if __name__=='__main__':main()
