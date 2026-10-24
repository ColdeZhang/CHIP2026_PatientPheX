#!/usr/bin/env python3
"""One inference-only original80-weighted124 run, fixed600s +TERM30; no retries."""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,time,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True);a=p.parse_args();b=a.directory.resolve();plan=json.loads((b/'run-plan.json').read_text());sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r;r.M=plan['mount_id']
 assert sha(__file__)==plan['driver_sha256'] and not (b/'job.json').exists(),'Recover existing job rather than restart'
 for ref in plan['references'].values():assert sha(ref['path'])==ref['sha256'] and Path(ref['path']).stat().st_size==ref['bytes']
 cpu=json.loads((b/'token-audit.json').read_text());assert cpu['prompts']==80 and cpu['concepts']==587 and cpu['patient_concept_pairs']==1557 and cpu['max_tokens']<14000 and cpu['truncated']==0
 assert (b/'baseline-prepared.json').exists()
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
  x=command("import subprocess,json,time;u="+repr(plan['gpu_uuid'])+";apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(v.split(',')[0].strip()==u for v in apps.splitlines());s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);d=next(v for v in s.splitlines() if v.split(',')[0].strip()==u);assert int(d.split(',')[2].strip().split()[0])>=38000;print(json.dumps(dict(gpu_uuid=u,empty=True,device=d,server_epoch=time.time(),apps=apps)))");save(n,x)
 expected=plan['remote_sources_sha256'];save('source-preflight.json',command("from pathlib import Path;import hashlib,shutil,json;expected="+repr(expected)+";actual={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in expected};assert actual==expected;assert not Path("+repr(plan['remote_root']+'/responses.jsonl')+").exists();free=shutil.disk_usage('.').free;assert free>20000000;print(json.dumps(dict(actual_sha256=actual,free_disk_bytes=free)))"));gpu('gpu-preflight.json')
 argv=plan['argv'];assert argv[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s','600']
 job=r.call('start_process',{'command':shlex.join(argv),'cwd':plan['cwd'],'env':plan['env']});save('job.json',dict(job=job,plan_sha256=sha(b/'run-plan.json'),cap_seconds=600,grace_seconds=30,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(json.dumps({'job':job['id'],'cap':600}),flush=True)
 cursors={'stdout':0,'stderr':0}
 while True:
  running=False
  for stream in cursors:
   x=readcall('read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':20000});cursors[stream]=x['next_cursor'];running|=x['running']
   if x['content']:
    with (b/(stream+'.log')).open('a') as f:f.write(x['content'])
    if stream=='stdout':print(x['content'],flush=True)
  (b/'monitor.json').write_text(json.dumps(dict(job=job['id'],running=running,cursors=cursors,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
  if not running:break
  time.sleep(10)
 gpu('gpu-release.json')
 for suffix in ['jsonl','manifest.json','checkpoint.json','loading.json']:
  remote=plan['remote_root']+'/responses.'+suffix;exists=command('from pathlib import Path;import json;print(json.dumps(Path('+repr(remote)+').exists()))')
  if not json.loads(exists['stdout']):
   save('incomplete.json',{'missing':remote,'no_retry_no_evaluation':True});return
  x=readcall('read_file',{'path':plan['cwd']+'/'+remote,'offset':0,'limit':1000000});assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(b/('responses.'+suffix)).write_text(x['content'])
 raw=[json.loads(l) for l in (b/'responses.jsonl').read_text().splitlines()];meta=json.loads((b/'responses.manifest.json').read_text());fp=meta['fingerprint'];assert meta['expected_rows']==80 and len(raw)==len({x['row_id'] for x in raw})==80
 assert {x['row_id']:str(x['pmc_id']) for x in raw}==plan['row_documents'] and meta['elapsed_seconds']<600 and meta['output_sha256']==sha(b/'responses.jsonl')
 assert fp['adapter_files']==plan['adapter_files'] and fp['source_sha256']==plan['runner_sha256'] and fp['prompts_sha256']==sha(b/'prompts.jsonl') and fp['thinking'] is False
 audit={x['row_id']:x['input_tokens'] for x in cpu['rows']};assert {x['row_id']:x['input_tokens'] for x in raw}==audit
 save('responses-freeze.json',dict(status='complete80_weighted_original_concepts_frozen',plan_sha256=sha(b/'run-plan.json'),raw_sha256=sha(b/'responses.jsonl'),manifest_sha256=sha(b/'responses.manifest.json'),gpu_release_sha256=sha(b/'gpu-release.json'),rows=80,elapsed_seconds=meta['elapsed_seconds'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('COMPLETE80_FROZEN_GPU_RELEASED',flush=True)
if __name__=='__main__':main()
