#!/usr/bin/env python3
"""One fixed input-only thinking-sampling diagnostic, 300s+TERM30; no retries."""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,time,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 b=Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/thinking-sampling-followup');plan=json.loads((b/'run-plan.json').read_text())
 sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as r;r.M=plan['mount_id']
 job=json.loads((b/'job.json').read_text())['job'];cpu=json.loads((b/'token-audit.json').read_text())
 assert job['id']=='proc_260680cf-7024-405d-a752-3f00e09f8d80'
 assert not (b/'responses-freeze.json').exists(),'Already collected, do not repeat'
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
  x=command("import subprocess,json,time;u="+repr(plan['gpu_uuid'])+";apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(v.split(',')[0].strip()==u for v in apps.splitlines());s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);d=next(v for v in s.splitlines() if v.split(',')[0].strip()==u);assert int(d.split(',')[2].strip().split()[0])>=76000;print(json.dumps(dict(gpu_uuid=u,empty=True,device=d,server_epoch=time.time(),apps=apps)))");save(n,x)
 cursors=json.loads((b/'monitor.json').read_text())['cursors']
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
 raw=[json.loads(l) for l in (b/'responses.jsonl').read_text().splitlines()];meta=json.loads((b/'responses.manifest.json').read_text());fp=meta['fingerprint'];assert meta['expected_rows']==1 and len(raw)==len({x['row_id'] for x in raw})==1
 assert {x['row_id']:str(x['pmc_id']) for x in raw}==plan['row_documents'] and meta['elapsed_seconds']<300 and meta['output_sha256']==sha(b/'responses.jsonl')
 assert fp['adapter_files']==plan['adapter_files'] and fp['source_sha256']==plan['runner_sha256'] and fp['prompts_sha256']==sha(b/'prompts.jsonl') and fp['thinking'] is True and fp['sampler']==plan['sampler'] and fp['presence_sha256']==plan['presence_sha256']
 audit={x['row_id']:x['input_tokens'] for x in cpu['rows']};assert {x['row_id']:x['input_tokens'] for x in raw}==audit
 assert raw[0]['input_ids_sha256']==cpu['true_input_ids_sha256'] and raw[0]['presence_calls']==raw[0]['output_tokens']
 save('responses-freeze.json',dict(status='complete_one_sampling_response_frozen_no_targets_or_quality',plan_sha256=sha(b/'run-plan.json'),raw_sha256=sha(b/'responses.jsonl'),manifest_sha256=sha(b/'responses.manifest.json'),gpu_release_sha256=sha(b/'gpu-release.json'),rows=1,elapsed_seconds=meta['elapsed_seconds'],at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('COMPLETE1_FROZEN_GPU_RELEASED',flush=True)
if __name__=='__main__':main()
