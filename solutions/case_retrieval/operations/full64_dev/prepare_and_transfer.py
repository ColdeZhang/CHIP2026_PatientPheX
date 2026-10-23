"""Copy only the complete fresh full64 epoch002 adapter; no GPU/inference/gold."""
from pathlib import Path
import hashlib,json,shlex,sys,time
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup']
import remote_io as r
B=Path(__file__).parent;T=B.parent/'full64-relation-cpu-preparation';O=B.parent/'dev-relation-transfer-preparation';R='case-retrieval/qwen35-preparation';SUB='full64-dev-transfer';S='69b991ee-183f-4add-af1a-579af92479c9';D='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,x):(B/n).write_text(json.dumps(x,indent=2)+'\n')
assert not (B/'adapter-source-job.json').exists(),'Do not duplicate a previous transfer; recover its recorded managed process'
assert sha(B/'proposal.json')=='edf29025dc054b64842d5112e7e23f9fec0d3c2e771b6439dbe160034336e571'
assert sha(T/'full64-run-plan.json')=='5f36de31452a4666b31454c38c0c903f15a29a5203d4e55cd8785a1fd5e1cd25'
complete=load(T/'formal2ep/training_complete.json');checkpoint=load(T/'formal2ep/epoch-002/checkpoint.json');actual=load(T/'checkpoint-actual-hashes.json')['epoch-002'];manifest=load(T/'formal2ep/run_manifest.json')
assert complete['epochs']==2 and complete['global_step']==68 and complete['smoke_only'] is False
assert complete['source_sha256']=='f83236587d8f307bb6e51ca44c9c69177efce27661b0f889bd00745808030f03'
assert checkpoint['completed_epochs']==2 and checkpoint['global_step']==68 and checkpoint['smoke_only'] is False
assert actual['files']==checkpoint['files'] and actual['completed_epochs']==2 and actual['global_step']==68
assert manifest['examples']==265 and manifest['total_tokens']==864937 and manifest['total_optimizer_steps']==68
assert checkpoint['fingerprint']==manifest['fingerprint']
assert (T/'formal-release.json').is_file()
files={name:checkpoint['files']['adapter/'+name] for name in ['adapter_config.json','adapter_model.safetensors']}
source=O/'serve_adapter.py';fetch=O/'fetch_adapter.py'
s=source.read_text();s=s.replace("ROOT=Path('relation-training/formal2ep/epoch-002/adapter').resolve()","ROOT=Path('full64-relation-cpu-preparation/formal2ep/epoch-002/adapter').resolve()")
lines=s.splitlines();lines=[('ALLOW='+repr(files)) if line.startswith('ALLOW=') else line for line in lines];s='\n'.join(lines)+'\n';s=s.replace('41237','41238');(B/'serve_adapter.py').write_text(s)
s=fetch.read_text().replace("ROOT=Path('dev-relation-transfer/adapter-epoch002')","ROOT=Path('full64-dev-transfer/adapter-epoch002')")
lines=s.splitlines();lines=[('FILES='+repr(files)) if line.startswith('FILES=') else line for line in lines];s='\n'.join(lines)+'\n';s=s.replace('41237','41238').replace("'source_training_global_step':52","'source_training_global_step':68");(B/'fetch_adapter.py').write_text(s)
write('adapter-transfer-plan.json',{'source_training_complete_sha256':sha(T/'formal2ep/training_complete.json'),'source_checkpoint_sha256':sha(T/'formal2ep/epoch-002/checkpoint.json'),'source_actual_hashes_sha256':sha(T/'checkpoint-actual-hashes.json'),'files':files,'serve_source_sha256':sha(B/'serve_adapter.py'),'fetch_source_sha256':sha(B/'fetch_adapter.py'),'source_mount':S,'target_mount':D,'source_epoch':2,'source_training_global_step':68,'target':R+'/'+SUB+'/adapter-epoch002','port':41238,'only_two_adapter_files':True})
r.M=S;r.upload(str(B/'serve_adapter.py'),R+'/'+SUB+'/serve_adapter.py')
job=r.call('start_process',{'command':'/usr/bin/timeout --signal=TERM --kill-after=15s 600 python3 -u '+SUB+'/serve_adapter.py','cwd':R});write('adapter-source-job.json',job)
try:
 r.M=D;r.upload(str(B/'fetch_adapter.py'),R+'/'+SUB+'/fetch_adapter.py')
 j=r.call('start_process',{'command':'/usr/bin/timeout --signal=TERM --kill-after=15s 300 python3 -u '+SUB+'/fetch_adapter.py','cwd':R});write('adapter-download-job.json',j);cursors={'stdout':0,'stderr':0}
 while True:
  running=False
  for stream in cursors:
   x=r.call('read_process_output',{'process_id':j['id'],'stream':stream,'cursor':cursors[stream],'limit':16000});cursors[stream]=x['next_cursor'];running|=x['running']
   if x['content']:
    with (B/('adapter-transfer.'+stream+'.log')).open('a') as f:f.write(x['content'])
    print(x['content'],flush=True)
  write('adapter-transfer-monitor.json',{'job':j['id'],'cursors':cursors,'running':running})
  if not running:break
  time.sleep(10)
 x=r.call('read_file',{'path':R+'/'+SUB+'/adapter-epoch002/transfer-manifest.json','offset':0,'limit':1000000});assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(B/'adapter-transfer-manifest.json').write_text(x['content'])
 copied=load(B/'adapter-transfer-manifest.json');assert copied['status']=='complete_full_sha' and copied['source_training_global_step']==68
 assert {n:v['sha256'] for n,v in copied['files'].items()}==files
finally:
 r.M=S;write('adapter-source-stop.json',r.call('stop_process',{'process_id':job['id']}));r.M=D
 code="import socket,json;s=socket.socket();s.settimeout(5);x=s.connect_ex(('10.5.4.7',41238));s.close();print(json.dumps({'connect_ex':x,'source_closed':x!=0}));assert x!=0"
 x=r.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':R,'timeout_seconds':15});write('adapter-source-closed.json',x);assert x['exit_code']==0
# Re-read both exact adapter files after copy; no inference and no shared env writes.
actual_sides={}
for label,mount,folder in [('source',S,'full64-relation-cpu-preparation/formal2ep/epoch-002/adapter'),('destination',D,SUB+'/adapter-epoch002')]:
 r.M=mount
 code="from pathlib import Path;import hashlib,json;root=Path("+repr(folder)+");names="+repr(list(files))+";print(json.dumps({n:{'bytes':(root/n).stat().st_size,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in names}))"
 x=r.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':R,'timeout_seconds':30});assert x['exit_code']==0
 actual_sides[label]=json.loads(x['stdout'])
 assert {n:v['sha256'] for n,v in actual_sides[label].items()}==files
assert actual_sides['source']==actual_sides['destination']
write('adapter-transfer-evidence.json',{'status':'complete_full_sha','source_epoch':2,'source_training_global_step':68,'source_files':actual_sides['source'],'destination_files':actual_sides['destination'],'copy_manifest_sha256':sha(B/'adapter-transfer-manifest.json'),'source_closed_evidence_sha256':sha(B/'adapter-source-closed.json'),'transfer_plan_sha256':sha(B/'adapter-transfer-plan.json')})
print('FULL64_ADAPTER_COPIED_SOURCE_CLOSED',flush=True)
