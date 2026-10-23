#!/usr/bin/env python3
"""Once-only fresh32 training and epoch2 inference, with separate fixed hard caps."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','infer']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True);a=p.parse_args();b=a.directory.resolve();plan=json.loads((b/'run-plan.json').read_text());sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r;r.M=plan['mount_id']
 assert sha(__file__)==plan['driver_sha256']
 def save(n,x):
  with (b/n).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 def call(name,kw):
  for attempt in range(3):
   try:return r.call(name,kw)
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 def command(code,limit=30):
  x=call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':plan['cwd'],'timeout_seconds':limit});assert x['exit_code']==0,x;return x
 def gpu(name):
  x=command("import subprocess,json,time;u="+repr(plan['gpu_uuid'])+";a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(l.split(',')[0].strip()==u for l in a.splitlines());d=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);v=next(l for l in d.splitlines() if l.split(',')[0].strip()==u);assert int(v.split(',')[2].strip().split()[0])>=70000;print(json.dumps({'gpu_uuid':u,'empty':True,'device':v,'apps':a,'server_epoch':time.time()}))");save(name,x)
 def fetch(remote,local,optional=False):
  x=command('from pathlib import Path;import json;print(json.dumps({"exists":Path('+repr(remote)+').exists()}))')
  if not json.loads(x['stdout'])['exists']:
   if optional:return False
   raise RuntimeError('Missing required terminal: '+remote)
  z=call('read_file',{'path':plan['cwd']+'/'+remote,'offset':0,'limit':1000000});assert hashlib.sha256(z['content'].encode()).hexdigest()==z['sha256'];p=b/local;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(z['content']);return True
 for item in plan['local_references'].values():assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
 phase=a.phase;assert not (b/(phase+'-job.json')).exists(),'Do not repeat existing managed job; recover by ID'
 expected=dict(plan['remote_sources_sha256']);remote_root=plan['remote_root'];cap=1800 if phase=='train' else 900
 if phase=='train':
  assert plan['training_cap_seconds']==cap;argv=plan['training_argv'];empty=remote_root+'/formal2ep'
 else:
  assert plan['inference_cap_seconds']==cap
  terminal=json.loads((b/'formal2ep/training_complete.json').read_text());actual=json.loads((b/'checkpoint-actual-hashes.json').read_text());assert terminal['epochs']==2 and terminal['global_step']==32 and terminal['source_sha256']==plan['trainer_sha256'] and terminal['elapsed_seconds']<1800
  cp=json.loads((b/'formal2ep/epoch-002/checkpoint.json').read_text());assert cp['completed_epochs']==2 and cp['global_step']==32 and cp['files']==actual['epoch-002']['files'];adapter={k.split('/')[-1]:v for k,v in cp['files'].items() if k in ['adapter/adapter_model.safetensors','adapter/adapter_config.json']};assert len(adapter)==2
  for name,value in adapter.items():expected[remote_root+'/formal2ep/epoch-002/adapter/'+name]=value
  argv=plan['inference_argv'];empty=remote_root+'/meta.responses.jsonl';save('actual-inference-plan.json',{'plan_sha256':sha(b/'run-plan.json'),'training_complete_sha256':sha(b/'formal2ep/training_complete.json'),'checkpoint_sha256':sha(b/'formal2ep/epoch-002/checkpoint.json'),'actual_checkpoint_hashes_sha256':sha(b/'checkpoint-actual-hashes.json'),'adapter_files':adapter,'argv':argv,'cap_seconds':900,'grace_seconds':30})
 assert argv[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(cap)]
 code="from pathlib import Path;import hashlib,json,shutil;expected="+repr(expected)+";actual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected;assert not Path("+repr(empty)+").exists();free=shutil.disk_usage('.').free;assert free>"+str(1500_000_000 if phase=='train' else 10_000_000)+";print(json.dumps({'actual_source_sha256':actual,'free_disk_bytes':free}))"
 save(phase+'-source-preflight.json',command(code,60));gpu(phase+'-gpu-preflight.json')
 job=r.call('start_process',{'command':shlex.join(argv),'cwd':plan['cwd'],'env':plan['env']});save(phase+'-job.json',{'job':job,'plan_sha256':sha(b/'run-plan.json'),'cap_seconds':cap,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps({'phase':phase,'job':job['id'],'cap_seconds':cap}),flush=True)
 cursors={'stdout':0,'stderr':0}
 while True:
  running=False
  for stream in cursors:
   x=call('read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':20000});cursors[stream]=x['next_cursor'];running|=x['running']
   if x['content']:
    with (b/(phase+'.'+stream+'.log')).open('a') as f:f.write(x['content'])
    if stream=='stdout':print(x['content'],flush=True)
  (b/(phase+'-monitor.json')).write_text(json.dumps({'job':job['id'],'running':running,'cursors':cursors,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
  if not running:break
  time.sleep(20 if phase=='train' else 10)
 gpu(phase+'-gpu-release.json')
 if phase=='train':
  complete=fetch(remote_root+'/formal2ep/training_complete.json','formal2ep/training_complete.json',True)
  for name in ['run_manifest.json','progress.json','first-nonzero-update.json','base-loading-info.json','token-mask-audit.json','epoch-001/checkpoint.json','epoch-002/checkpoint.json']:
   fetch(remote_root+'/formal2ep/'+name,'formal2ep/'+name,True)
  if not complete:save('training-incomplete.json',{'status':'incomplete_no_auto_resume_no_inference','job':job['id'],'source':'saved stdout/stderr and stage reports','cap_seconds':1800});return
  data=json.loads((b/'formal2ep/training_complete.json').read_text());assert data['global_step']==32 and data['epochs']==2 and not data['smoke_only'] and data['elapsed_seconds']<1800 and data['source_sha256']==plan['trainer_sha256']
  code="from pathlib import Path;import hashlib,json;root=Path("+repr(remote_root+'/formal2ep')+");out={};\nfor ep in ['epoch-001','epoch-002']:\n m=json.loads((root/ep/'checkpoint.json').read_text());actual={n:hashlib.sha256((root/ep/n).read_bytes()).hexdigest() for n in m['files']};assert actual==m['files'];out[ep]={'files':actual,'completed_epochs':m['completed_epochs'],'global_step':m['global_step']}\nprint(json.dumps(out))"
  x=command(code,60);(b/'checkpoint-actual-hashes.json').write_text(x['stdout']);print('TRAIN32_COMPLETE_RELEASED',flush=True)
 else:
  complete=fetch(remote_root+'/meta.responses.manifest.json','meta.responses.manifest.json',True)
  for suffix in ['jsonl','checkpoint.json','loading.json']:fetch(remote_root+'/meta.responses.'+suffix,'meta.responses.'+suffix,True)
  if not complete:save('inference-incomplete.json',{'status':'incomplete_no_pair_fit_no_retry','job':job['id'],'cap_seconds':900});return
  raw=[json.loads(l) for l in (b/'meta.responses.jsonl').read_text().splitlines()];meta=json.loads((b/'meta.responses.manifest.json').read_text());fp=meta['fingerprint'];actual=json.loads((b/'actual-inference-plan.json').read_text())
  assert len(raw)==len({r['row_id'] for r in raw})==80 and {r['row_id']:str(r['pmc_id']) for r in raw}==plan['meta_row_documents']
  assert fp['adapter_files']==actual['adapter_files'] and fp['prompts_sha256']==plan['meta_prompts_sha256'] and fp['source_sha256']==plan['runner_sha256'] and fp['thinking'] is False
  assert meta['base_parameters']==8953803264 and meta['total_parameters']==8997081600 and meta['elapsed_seconds']<900 and meta['output_sha256']==sha(b/'meta.responses.jsonl') and meta['expected_rows']==80
  save('meta-responses-freeze.json',{'status':'complete_F1_SFT32_responses_frozen_before_meta_fit','plan_sha256':sha(b/'run-plan.json'),'actual_inference_plan_sha256':sha(b/'actual-inference-plan.json'),'raw_sha256':sha(b/'meta.responses.jsonl'),'manifest_sha256':sha(b/'meta.responses.manifest.json'),'rows':80,'elapsed_seconds':meta['elapsed_seconds'],'release_sha256':sha(b/'infer-gpu-release.json'),'no_outer_or_originaldev_gold':True});print('META80_COMPLETE_RELEASED_FROZEN',flush=True)
if __name__=='__main__':main()
