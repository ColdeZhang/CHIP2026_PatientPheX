from pathlib import Path
import sys,json,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
A=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';rel='research/deepseek-r1-0528-qwen3-8b-readiness/assets'
if (A/'start-attempt.json').exists():raise SystemExit('start attempt already recorded; inspect original job/PID, do not repeat')
pre=json.loads((A/'remote-source-preflight.json').read_text()); detail=pre.get('structuredContent') or json.loads(pre['content'][0]['text']); assert not pre.get('isError') and detail.get('exit_code')==0 and json.loads(detail['stdout'])['status']=='REMOTE_SOURCE_SHA_AND_PYTHON310_COMPILE_OK'
started=time.time();intent={'model_id':'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B','MS_revision':'5ee5fa4832e658ab84298806de06ed2a167834bd','model_dir':root+'research/deepseek-r1-0528-qwen3-8b-readiness/models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6','expected_uid':1006,'intent_created_epoch':started,'absolute_deadline_epoch':started+2100,'download_budget_seconds':2100,'termination_grace_seconds':30,'mode':'ASSETS_ONLY_NO_GPU_MODEL_LOAD_TRAINING_OR_INSTALL','expected_files_sha256':hashlib.sha256((A/'expected-files.json').read_bytes()).hexdigest(),'download_script_sha256':hashlib.sha256((A/'download_deepseek.py').read_bytes()).hexdigest(),'derived_shapes_sha256':hashlib.sha256((A/'derived-tensor-shapes.json').read_bytes()).hexdigest()}
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
write(A/'intent.json',intent)
c=Client();c.initialize();r=c.request('tools/call',{'name':'write_file','arguments':{'mount_id':M,'path':rel+'/intent.json','content':(A/'intent.json').read_text(),'encoding':'utf8','create_parents':True}});write(A/'intent-upload-result.json',r)
if r.get('isError'):raise SystemExit('intent upload failed; do not start')
launcher="import json,time,os,math; p="+repr(root+rel+'/intent.json')+"; d=json.load(open(p)); rem=math.ceil(d['absolute_deadline_epoch']-time.time()); assert rem>0; os.execvp('timeout',['timeout','--signal=TERM','--kill-after=30s',str(rem)+'s','python3','-u',"+repr(root+rel+'/download_deepseek.py')+",'--intent',p])"
cmd='python3 -c '+shlex.quote(launcher)
args={'mount_id':M,'command':cmd,'cwd':'','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
write(A/'start-attempt.json',{'status':'INTENT_PERSISTED_BEFORE_START_CALL','call_started_epoch':time.time(),'mount_id':M,'arguments':args,'intent_sha256':hashlib.sha256((A/'intent.json').read_bytes()).hexdigest(),'on_unknown_outcome':'Read remote assets/started.json and original managed process list; do not repeat start_process.'})
try:
 r=c.request('tools/call',{'name':'start_process','arguments':args});write(A/'download-job.json',r);print(json.dumps(r,ensure_ascii=False))
except Exception as ex:
 write(A/'start-unknown-outcome.json',{'status':'START_OUTCOME_UNKNOWN','exception_type':type(ex).__name__,'observed_epoch':time.time()});raise SystemExit('Unknown start outcome; inspect original job/PID before any action')
