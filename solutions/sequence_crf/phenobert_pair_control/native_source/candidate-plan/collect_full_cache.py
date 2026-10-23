from pathlib import Path
import base64,concurrent.futures,gzip,hashlib,io,json,math,shlex,sys,threading,time,zipfile
B=Path(__file__).resolve().parent;F=B/'full'
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
local=threading.local();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f';remote='phenobert-pair-train64'
def call(name,args):
 if not hasattr(local,'client'):local.client=Client();local.client.initialize()
 r=local.client.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
 assert not r.get('isError'),r
 return r['structuredContent']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=json.loads((F/'job.json').read_text())
terminal={s:call('read_process_output',{'process_id':j['result']['id'],'stream':s,'cursor':0,'limit':32768}) for s in ('stdout','stderr')}
(F/'process-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
assert all(not x['running'] for x in terminal.values()),'Inference is still running; no artifact collection attempted.'
code='''from pathlib import Path
import hashlib,json,subprocess,time,zipfile
b=Path('.');target='GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
selected=[x for x in gpu.splitlines() if target in x];assert len(selected)==1 and target not in apps
assert int(selected[0].split(',')[2].strip())<=512
(b/'release.json').write_text(json.dumps({'target':target,'target_gpu_row':selected[0],'target_no_compute_process':True,'time':time.time()},indent=2)+'\\n')
files=[]
with zipfile.ZipFile(b/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['result.json','probabilities.jsonl.gz','fresh-GPU.json','release.json','inference-plan.json','run_native_pairs.py']:
  p=b/name;data=p.read_bytes();files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});z.writestr(name,data)
data=(b/'evidence.zip').read_bytes();print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'files':files}))
'''
r=call('run_command',{'command':'/dataset/miniforge3/envs/vlm-det/bin/python -c '+shlex.quote(code),'cwd':remote,'timeout_seconds':30,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''}})
assert r['exit_code']==0,r
manifest=json.loads(r['stdout']);parts=F/'receive-parts';parts.mkdir(exist_ok=True)
chunk=262144;started=time.monotonic()
def get(i):
 n=min(chunk,manifest['bytes']-i*chunk);p=parts/f'part-{i:04d}'
 if p.exists():
  data=p.read_bytes();assert len(data)==n
 else:
  x=call('read_file',{'path':remote+'/evidence.zip','offset':i*chunk,'limit':n});assert x['encoding']=='base64'
  data=base64.b64decode(x['content']);assert len(data)==n;p.write_bytes(data)
 return {'index':i,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:receipts=list(pool.map(get,range(math.ceil(manifest['bytes']/chunk))))
data=b''.join((parts/f'part-{r["index"]:04d}').read_bytes() for r in receipts)
assert len(data)==manifest['bytes'] and hashlib.sha256(data).hexdigest()==manifest['sha256']
out=F/'terminal';out.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
 for item in manifest['files']:
  value=z.read(item['name']);assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
  (out/item['name']).write_bytes(value)
manifest.update(transfer_chunk_bytes=chunk,parallel_workers=8,chunk_receipts=receipts,transfer_seconds=time.monotonic()-started)
(F/'artifacts-transfer.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Input-only exact key and probability coverage: never open gold or compute a quality metric.
with gzip.open(B/'model-inputs.jsonl.gz','rt') as f:requested={json.loads(line)['pair_key'] for line in f}
seen=set();max_sum_error=0.0
with gzip.open(out/'probabilities.jsonl.gz','rt') as f:
 for line in f:
  row=json.loads(line);key=row['pair_key'];p=row['probabilities']
  assert key in requested and key not in seen;seen.add(key)
  assert len(p)==3 and all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in p)
  error=abs(sum(p)-1);assert error<=1e-6;max_sum_error=max(max_sum_error,error)
assert seen==requested and len(seen)==135000
result=json.loads((out/'result.json').read_text())
assert result['rows']==135000 and result['input_sha256']==sha(B/'model-inputs.jsonl.gz')
assert result['probabilities_sha256']==sha(out/'probabilities.jsonl.gz')
assert result['driver_sha256']==sha(B/'run_native_pairs.py') and result['plan_sha256']==sha(F/'inference-plan.json')
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
complete={'status':'PASS_COMPLETE_PUBLIC_PAIR_TRAIN64_INPUT_ONLY_CACHE','class_order':['Not Match','Related','Match'],
          'direction':'original_span_text -> current_preferred_name','public_model_task_fit':False,
          'weight_sha256':'353fc58889b2e59ddb9bf845c037283247268812ff3959cead118f35db90bea8',
          'atomic_request':ref(B/'atomic-request.jsonl.gz'),'scores':ref(out/'probabilities.jsonl.gz'),
          'hpo':ref(Path('/workspace/patientphex-data/hp.obo')),
          'model_inputs':ref(B/'model-inputs.jsonl.gz'),'plan':ref(F/'inference-plan.json'),
          'source':ref(B/'run_native_pairs.py'),'actual_result':ref(out/'result.json'),
          'artifacts_transfer':ref(F/'artifacts-transfer.json'),'release':ref(out/'release.json'),
          'native_independent_review':ref(B.parent.parent/'validation/phenobert-pair-native-review/review.json'),
          'requested_unique_tuples':len(requested),'actual_unique_tuples':len(seen),
          'missing_keys':0,'extra_keys':0,'duplicate_keys':0,'all_finite_normalized':True,
          'maximum_probability_sum_absolute_error':max_sum_error,
          'elapsed_seconds':result['elapsed_seconds'],'forward_seconds':result['inference_seconds'],
          'cap_seconds':600,'TERM_grace_seconds':15,'previous_pilot_probability_rows_reused':0,
          'previous_public_model_runtime_and_weights_reused':True,
          'no_gold_or_quality_evaluation':True,'no_dev_input':True,'no_training':True}
(F/'completion.json').write_text(json.dumps(complete,indent=2)+'\n')
print(json.dumps({'completion_sha256':sha(F/'completion.json'),'scores_sha256':complete['scores']['sha256'],
                  'rows':len(seen),'elapsed_seconds':result['elapsed_seconds'],'forward_seconds':result['inference_seconds'],
                  'release_sha256':complete['release']['sha256'],'transfer_seconds':manifest['transfer_seconds']}))
