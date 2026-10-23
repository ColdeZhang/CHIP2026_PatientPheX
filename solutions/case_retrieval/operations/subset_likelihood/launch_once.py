#!/usr/bin/env python3
"""Only own fixed98 forward job, native300s+TERM30, no retry or gold access."""
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
 save('source-preflight.json',command("from pathlib import Path;import hashlib,shutil,json;expected="+repr(plan['remote_sources_sha256'])+";actual={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in expected};assert actual==expected;assert not Path("+repr(plan['remote_root']+'/scores.jsonl')+").exists();free=shutil.disk_usage('.').free;assert free>20000000;print(json.dumps(dict(actual_sha256=actual,free_disk_bytes=free)))"));gpu('gpu-preflight.json');assert plan['argv'][:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s','300']
 job=r.call('start_process',dict(command=shlex.join(plan['argv']),cwd=plan['cwd'],env=plan['env']));save('job.json',dict(job=job,plan_sha256=sha(b/'run-plan.json'),cap_seconds=300,grace_seconds=30,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(json.dumps({'job':job['id'],'cap':300}),flush=True);cursors={'stdout':0,'stderr':0}
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
 complete=json.loads((b/'complete.json').read_text());raw=[json.loads(l) for l in (b/'scores.jsonl').read_text().splitlines()];assert complete['rows']==len(raw)==len({(x['row_id'],x['cid'],x['mask']) for x in raw})==98 and complete['elapsed_seconds']<300
 assert complete['score_sha256']==sha(b/'scores.jsonl') and complete['manifest_sha256']==sha(b/'run_manifest.json') and complete['plan_sha256']==sha(b/'remote-plan.json') and all(math.isfinite(x['sum_logprob']) for x in raw)
 save('scores-freeze.json',dict(status='all98_actual_scores_frozen_before_local_labels',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(b/'run-plan.json'),raw_sha256=sha(b/'scores.jsonl'),complete_sha256=sha(b/'complete.json'),manifest_sha256=sha(b/'run_manifest.json'),gpu_release_sha256=sha(b/'gpu-release.json'),no_labels_read=True));print('COMPLETE98_FROZEN_GPU_RELEASED',flush=True)
if __name__=='__main__':main()
