#!/usr/bin/env python3
"""Only own fixed432 new forward job, native600s+TERM30, no retry or gold access."""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,time,datetime,math
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True);a=p.parse_args();b=a.directory.resolve();plan=json.loads((b/'run-plan.json').read_text());sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r;r.M=plan['mount_id']
 assert sha(__file__)==plan['launcher_sha256'] and not (b/'job.json').exists(),'Read original job, never relaunch automatically'
 for ref in plan['references'].values():assert sha(ref['path'])==ref['sha256'] and Path(ref['path']).stat().st_size==ref['bytes']
 def save(n,x):
  with (b/n).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 def readcall(name,args):
  for i in range(3):
   try:return r.call(name,args)
   except Exception:
    if i==2:raise
    time.sleep(2)
 def command(code):
  x=readcall('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':plan['cwd'],'timeout_seconds':60});assert x['exit_code']==0,x;return x
 def gpu(n):
  x=command("import subprocess,json,time;u="+repr(plan['gpu_uuid'])+";apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(v.split(',')[0].strip()==u for v in apps.splitlines());s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);d=next(v for v in s.splitlines() if v.split(',')[0].strip()==u);assert int(d.split(',')[2].strip().split()[0])>=70000;print(json.dumps(dict(gpu_uuid=u,empty=True,device=d,server_epoch=time.time(),apps=apps)))");save(n,x)
 save('source-preflight.json',command("from pathlib import Path;import hashlib,shutil,json;expected="+repr(plan['remote_sources_sha256'])+";actual={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in expected};assert actual==expected;assert not Path("+repr(plan['remote_root']+'/scores.jsonl')+").exists();free=shutil.disk_usage('.').free;assert free>20000000;print(json.dumps(dict(actual_sha256=actual,free_disk_bytes=free)))"));gpu('gpu-preflight.json');assert plan['budget']['native_seconds']==600 and plan['argv'][:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s','600']
 job=r.call('start_process',dict(command=shlex.join(plan['argv']),cwd=plan['cwd'],env=plan['env']));save('job.json',dict(job=job,plan_sha256=sha(b/'run-plan.json'),cap_seconds=600,grace_seconds=30,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(json.dumps({'job':job['id'],'cap':600}),flush=True);cursors={'stdout':0,'stderr':0}
 while True:
  running=False
  for stream in cursors:
   x=readcall('read_process_output',dict(process_id=job['id'],stream=stream,cursor=cursors[stream],limit=20000));cursors[stream]=x['next_cursor'];running|=x['running']
   if x['content']:
    with (b/(stream+'.log')).open('a') as f:f.write(x['content'])
    if stream=='stdout':print(x['content'],flush=True)
  (b/'monitor.json').write_text(json.dumps(dict(job=job['id'],running=running,cursors=cursors,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
  if not running:break
  time.sleep(10)
 gpu('gpu-release.json')
 for n in ['scores.jsonl','run_manifest.json','longest-actual.json','complete.json']:
  remote=plan['remote_root']+'/'+n;x=command('from pathlib import Path;import json;print(json.dumps(Path('+repr(remote)+').exists()))')
  if not json.loads(x['stdout']):save('incomplete.json',dict(missing=remote,no_retry_no_evaluation=True));return
  x=readcall('read_file',dict(path=plan['cwd']+'/'+remote,offset=0,limit=1000000));assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(b/n).write_text(x['content'])
 complete=json.loads((b/'complete.json').read_text());raw=[json.loads(l) for l in (b/'scores.jsonl').read_text().splitlines()];assert complete['rows']==len(raw)==len({(x['row_id'],x['cid'],x['mask']) for x in raw})==432 and complete['elapsed_seconds']<600
 assert complete['score_sha256']==sha(b/'scores.jsonl') and complete['manifest_sha256']==sha(b/'run_manifest.json') and complete['plan_sha256']==sha(b/'remote-plan.json') and all(math.isfinite(x['sum_logprob']) for x in raw)
 # Join only exact same model/runtime and exact complete encodings; prior scores are immutable.
 import gzip
 enc=json.loads(gzip.decompress((b/'all530-encoded.json.gz').read_bytes()));oldenc=json.loads(gzip.decompress(Path(plan['reused_encoded']['path']).read_bytes()));key=lambda x:(x['row_id'],x['cid'],x['mask']);by={key(x):x for x in enc};oldby={key(x):x for x in oldenc};old=[json.loads(l) for l in Path(plan['reused_scores']['path']).read_text().splitlines()]
 assert sha(plan['reused_scores']['path'])==plan['reused_scores']['sha256'] and sha(plan['reused_encoded']['path'])==plan['reused_encoded']['sha256'] and len(old)==98
 previous=json.loads(Path(plan['reused_manifest']['path']).read_text());current=json.loads((b/'run_manifest.json').read_text());assert sha(plan['reused_manifest']['path'])==plan['reused_manifest']['sha256']
 assert all(previous[k]==current[k] for k in ['model_files','adapter_files','runtime','configuration','loading','trainable_parameters']) and all(by[k]==v for k,v in oldby.items())
 joined=old+raw;assert len(joined)==len({key(x) for x in joined})==len(by)==530 and {key(x) for x in old}.isdisjoint({key(x) for x in raw})
 for x in joined:
  e=by[key(x)];assert x['score_label_ids']==e['score_labels'] and x['score_start']==e['score_start'] and x['input_ids_sha256']==hashlib.sha256(json.dumps(e['input_ids']).encode()).hexdigest() and x['logits_to_keep']==e['logits_to_keep'] and x['delimiter']==e['delimiter']
  assert all(math.isfinite(v) and v<=0 for v in x['token_logprobs']) and abs(sum(x['token_logprobs'])-x['sum_logprob'])<1e-4
 with (b/'all530-scores.jsonl').open('x') as f:
  for x in sorted(joined,key=key):f.write(json.dumps(x,ensure_ascii=False)+'\n')
 save('scores-freeze.json',dict(status='all530_scores_frozen_before_local_labels',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(b/'run-plan.json'),raw_sha256=sha(b/'all530-scores.jsonl'),new432_sha256=sha(b/'scores.jsonl'),reused98_sha256=sha(plan['reused_scores']['path']),complete_sha256=sha(b/'complete.json'),manifest_sha256=sha(b/'run_manifest.json'),gpu_release_sha256=sha(b/'gpu-release.json'),same_model_runtime_and_complete98_encodings_verified=True,no_labels_read=True));print('COMPLETE530_FROZEN_GPU_RELEASED',flush=True)
if __name__=='__main__':main()
