"""One approved full64 epoch002 arm on the same A30; no evaluation or gold access.
Requires complete training, copied actual adapter, immutable plan and explicit GPU handoff.
"""
from pathlib import Path
import datetime,hashlib,json,shlex,sys,time
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup']
import remote_io as r
B=Path(__file__).parent;O=B.parent/'dev-relation-transfer-preparation';T=B.parent/'full64-relation-cpu-preparation';R='case-retrieval/qwen35-preparation';SUB='full64-dev-transfer';GPU='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
r.M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ref(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def call(name,args):
 for attempt in range(3):
  try:return r.call(name,args)
  except Exception:
   if attempt==2:raise
   time.sleep(2)
def fetch(n):
 x=call('read_file',{'path':R+'/'+SUB+'/'+n,'offset':0,'limit':1000000});assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(B/n).write_text(x['content'])
def gpu_state(name):
 code="import subprocess,json,time;v=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(x.split(',')[0].strip()=="+repr(GPU)+" for x in v.splitlines()),'Target GPU occupied; no process termination attempted';print(json.dumps({'target_empty':True,'gpu_uuid':"+repr(GPU)+",'compute_apps':v,'server_epoch':time.time()}))"
 x=call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':R,'timeout_seconds':20});write(name,x);assert x['exit_code']==0;return json.loads(x['stdout'])

launch='--launch' in sys.argv
assert not (B/'job.json').exists(),'Existing job requires recovery, never implicit restart'
assert (B/'run-plan.json').exists()==launch,'Default prepares once; --launch consumes an existing frozen plan'
assert sha(B/'proposal.json')=='edf29025dc054b64842d5112e7e23f9fec0d3c2e771b6439dbe160034336e571'
assert sha(T/'full64-run-plan.json')=='5f36de31452a4666b31454c38c0c903f15a29a5203d4e55cd8785a1fd5e1cd25'
handoff=load(B/'gpu-handoff.json');assert handoff['gpu_uuid']==GPU and handoff['released_to']=='round2_retrieval_resume' and handoff['root_authorized_single_arm'] is True
training=load(T/'formal2ep/training_complete.json');ckpt=load(T/'formal2ep/epoch-002/checkpoint.json');tm=load(T/'formal2ep/run_manifest.json');actual=load(T/'checkpoint-actual-hashes.json')['epoch-002']
assert training['epochs']==2 and training['global_step']==68 and training['smoke_only'] is False
assert training['source_sha256']=='f83236587d8f307bb6e51ca44c9c69177efce27661b0f889bd00745808030f03'
assert tm['examples']==265 and tm['total_tokens']==864937 and tm['total_optimizer_steps']==68
assert ckpt['completed_epochs']==2 and ckpt['global_step']==68 and ckpt['fingerprint']==tm['fingerprint'] and actual['files']==ckpt['files']
transfer=load(B/'adapter-transfer-evidence.json');assert transfer['status']=='complete_full_sha' and transfer['source_epoch']==2 and transfer['source_training_global_step']==68
assert transfer['source_files']==transfer['destination_files'];af=transfer['destination_files'];adapter={n:v['sha256'] for n,v in af.items()}
assert set(adapter)=={'adapter_config.json','adapter_model.safetensors'} and all(ckpt['files']['adapter/'+n]==v for n,v in adapter.items())
old=load(O/'paired-adapter-plan.json');oldfp=load(O/'paired-adapter.responses.checkpoint.json');env=old['env'];assert env['CUDA_VISIBLE_DEVICES']==GPU
argv=list(old['runner_argv']);argv[argv.index('--output')+1]=SUB+'/full64-adapter.responses.jsonl';argv[argv.index('--adapter')+1]=SUB+'/adapter-epoch002'
fp=json.loads(json.dumps(oldfp));fp['configuration']['output']=SUB+'/full64-adapter.responses.jsonl';fp['configuration']['adapter']=SUB+'/adapter-epoch002';fp['adapter_files']=adapter
# Actual read-only environment and all 15 model/source/adapter files are checked before launch.
checks={p:fp['model_files'][p]['sha256'] for p in fp['model_files']}
code="from pathlib import Path;import hashlib,json,torch,transformers,peft,tokenizers;root=Path("+repr(fp['configuration']['model'])+");expected="+repr(checks)+";model={n:{'bytes':(root/n).stat().st_size,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in expected};assert all(model[n]['sha256']==h for n,h in expected.items());sources="+repr({'dev-relation-transfer/qwen35_runner.snapshot.py':fp['source_sha256'],'dev-relation-transfer/prompts.jsonl':fp['prompts_sha256'],'dev-relation-transfer/model-source-manifest.json':fp['source_manifest_sha256'],**{SUB+'/adapter-epoch002/'+n:h for n,h in adapter.items()}})+";observed={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources};assert observed==sources;runtime={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'tokenizers':tokenizers.__version__,'cuda':torch.version.cuda};assert not torch.cuda.is_initialized();assert not list(Path("+repr(SUB)+").glob('full64-adapter.responses*'));print(json.dumps({'model_files':model,'runtime_cpu':runtime,'source_sha256':observed,'cuda_initialized':torch.cuda.is_initialized()}))"
preenv={**env,'CUDA_VISIBLE_DEVICES':''}
command='env '+ ' '.join(shlex.quote(k+'='+v) for k,v in preenv.items())+' '+shlex.quote(argv[0])+' -c '+shlex.quote(code)
preflight_name='source-runtime-launch-preflight.json' if launch else 'source-runtime-preflight.json'
x=call('run_command',{'command':command,'cwd':R,'timeout_seconds':60});write(preflight_name,x);assert x['exit_code']==0
observed=json.loads(x['stdout']);
if launch:
 frozen_preflight=load(B/'source-runtime-preflight.json');assert frozen_preflight['exit_code']==0 and observed==json.loads(frozen_preflight['stdout']),'Fresh source/runtime differs from frozen preflight'
assert observed['model_files']==fp['model_files'] and observed['cuda_initialized'] is False
assert observed['runtime_cpu']=={k:v for k,v in fp['runtime'].items() if k not in {'gpu','visible_gpu'}}
contract={'schema':'full64-dev-fixed-arm-v1','evaluator_source_sha256':sha(B/'evaluate_fixed_arm.py'),'training':{'complete':ref(T/'formal2ep/training_complete.json'),'checkpoint':ref(T/'formal2ep/epoch-002/checkpoint.json'),'actual_hashes':ref(T/'checkpoint-actual-hashes.json'),'run_manifest':ref(T/'formal2ep/run_manifest.json')},'adapter_files':af,'adapter_transfer':ref(B/'adapter-transfer-evidence.json'),'expected_runner_fingerprint':fp}
plan={'status':'frozen_single_new_adapter_arm_no_gold','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_sha256':sha(__file__),'proposal_sha256':sha(B/'proposal.json'),'gpu_handoff_sha256':sha(B/'gpu-handoff.json'),'source_runtime_preflight_sha256':sha(B/'source-runtime-preflight.json'),'runner_argv':argv,'command':shlex.join(['/usr/bin/timeout','--signal=TERM','--kill-after=30s','900',*argv]),'mount_id':r.M,'cwd':R,'env':env,'execution_cap_seconds':900,'kill_grace_seconds':30,'expected_rows':64,'old_control_prediction_freeze_sha256':sha(O/'paired-predictions-freeze.json'),'old_control_summary_sha256':sha(O/'paired-summary.json'),'evaluation_contract':contract,'scope':'Fixed955entities/64prompts; new full64 epoch002 only. Prior same-A30 base/SFT48/fullLR reused. No prompt/schema/threshold/epoch tuning. Originaldev already used for research, not blind.'}
if not launch:
 write('run-plan.json',plan);print('FROZEN_PLAN_PREPARED_NO_GPU_START',flush=True);sys.exit(0)
frozen=load(B/'run-plan.json')
assert {k:v for k,v in plan.items() if k!='frozen_at_utc'}=={k:v for k,v in frozen.items() if k!='frozen_at_utc'},'Frozen plan drift; no launch'
plan=frozen
fresh=gpu_state('preflight-gpu.json')
# Never retry a mutation: ambiguous start result is investigated by managed process identity.
job=r.call('start_process',{'command':plan['command'],'cwd':R,'env':env});write('job.json',{'job':job,'run_plan_sha256':sha(B/'run-plan.json'),'preflight_server_epoch':fresh['server_epoch'],'local_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps({'STARTED_SINGLE_ARM':job['id'],'cap_seconds':900}),flush=True)
cursors={'stdout':0,'stderr':0};started=time.time()
while True:
 running=False
 for stream in cursors:
  x=call('read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':24000});cursors[stream]=x['next_cursor'];running|=x['running']
  if x['content']:
   with (B/('inference.'+stream+'.log')).open('a') as f:f.write(x['content'])
   if stream=='stdout':print(x['content'],flush=True)
 write('monitor.json',{'job':job['id'],'cursors':cursors,'running':running,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if not running:break
 time.sleep(20)
gpu_state('released-gpu.json')
# Preserve actual raw first if interrupted; missing completion metadata fails without evaluation.
for suffix in ['.jsonl','.checkpoint.json','.loading.json','.manifest.json']:fetch('full64-adapter.responses'+suffix)
raw=B/'full64-adapter.responses.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines() if l.strip()];meta=load(B/'full64-adapter.responses.manifest.json');checkpoint=load(B/'full64-adapter.responses.checkpoint.json')
prompts=[json.loads(l) for l in (O/'prompts.jsonl').read_text().splitlines() if l.strip()];tokens={x['row_id']:x['input_tokens'] for x in load(O/'tokenizer-audit.json')['rows']}
assert len(rows)==len({x['row_id'] for x in rows})==64
assert {x['row_id']:str(x['pmc_id']) for x in rows}=={x['row_id']:str(x['pmc_id']) for x in prompts}
assert {x['row_id']:x['input_tokens'] for x in rows}==tokens
assert all(x['removed_demonstrations']==0 and isinstance(x['hit_token_limit'],bool) and isinstance(x['response'],str) and 0<=x['output_tokens']<=1600 for x in rows)
assert meta['expected_rows']==64 and meta['output_sha256']==sha(raw) and meta['fingerprint']==checkpoint==fp
assert meta['base_parameters']==8953803264 and meta['total_parameters']==8997081600 and meta['eos_token_ids']==[248044,248046]
assert all(meta['load_information'][k]==[] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
write('complete.json',{'status':'complete_pending_frozen_cpu_evaluation','run_plan_sha256':sha(B/'run-plan.json'),'raw_sha256':sha(raw),'manifest_sha256':sha(B/'full64-adapter.responses.manifest.json'),'checkpoint_sha256':sha(B/'full64-adapter.responses.checkpoint.json'),'released_gpu_sha256':sha(B/'released-gpu.json'),'rows':64,'hit_token_limits':sum(x['hit_token_limit'] for x in rows),'removed_demonstrations':0,'actual_token_lengths_match':True,'all15_model_SHA_match':True,'same_runtime_version_and_GPU':True,'elapsed_seconds':meta['elapsed_seconds'],'raw_generation_seconds':sum(x['batch_seconds'] for x in rows),'output_tokens':sum(x['output_tokens'] for x in rows),'peak_allocated_bytes':meta['max_gpu_allocated_bytes'],'local_monitor_seconds':time.time()-started})
print('FULL64_ARM_COMPLETE_NO_EVALUATION',flush=True)
