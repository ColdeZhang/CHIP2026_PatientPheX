"""Freeze and deploy the bounded full input-only plan. Does not launch inference."""
from pathlib import Path
import base64,concurrent.futures,hashlib,json,math,shlex,sys,threading,time
B=Path(__file__).resolve().parent;F=B/'full';F.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pilot=json.loads((B/'pilot/terminal/result.json').read_text())
assert pilot['status']=='PASS_NATIVE_PUBLIC_PAIR_INPUT_ONLY_INFERENCE'
release=json.loads((B/'pilot/terminal/release.json').read_text());assert release['target_no_compute_process']
assert sha(B/'run_native_pairs.py')==pilot['driver_sha256']
plan=json.loads((B/'pilot/inference-plan.json').read_text())
warm_max=max(x['seconds'] for x in pilot['batches'][1:])
estimated=30+2*math.ceil(135000/64)*warm_max
assert estimated<=600
plan.update(scope='All135000 unique native tuples from fixed train64 input-only top3 candidate pools; no labels, quality evaluation, task fitting or dev input',
            rows=135000,input_file='model-inputs.jsonl.gz',input_sha256=sha(B/'model-inputs.jsonl.gz'),
            time_cap_seconds=600,TERM_grace_seconds=15,
            sampling='All model-input tuple keys, frozen lexical SHA order; no candidate selection',
            budget={'estimated_seconds':estimated,'formula':'30 + 2 * ceil(135000/64) * max(pilot warm batches)',
                    'pilot_result_sha256':sha(B/'pilot/terminal/result.json'),
                    'warm_max_batch_seconds':warm_max,'cold_first_batch_seconds':pilot['batches'][0]['seconds'],
                    'caveat':'Fixed startup is paid once; factor2 and30s reserve cover longer batch padding, JSON/output and resource variation; hardcap remains600s'},
            atomic_request_sha256=sha(B/'atomic-request.jsonl.gz'),
            current_ontology_sha256='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
            public_model_task_fit=False,class_order=['Not Match','Related','Match'],
            direction='original_span_text -> current_preferred_name',
            no_previous_pilot_probabilities_reused=True)
(F/'inference-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
local=threading.local();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f';remote='phenobert-pair-train64';started=time.monotonic()
def call(name,args):
 if not hasattr(local,'client'):local.client=Client();local.client.initialize()
 r=local.client.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
 assert not r.get('isError'),r
 return r['structuredContent']
def upload(path,name):
 data=Path(path).read_bytes();assert len(data)<=524288
 result=call('write_file',{'path':remote+'/'+name,'content':base64.b64encode(data).decode(),'encoding':'base64','create_parents':True})
 return {'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'receipt':result}
source=B/'model-inputs.jsonl.gz';chunk=524288
known={r['index']:r for r in map(json.loads,(F/'chunk-receipts.jsonl').read_text().splitlines())} if (F/'chunk-receipts.jsonl').exists() else {}
lock=threading.Lock()
def part(i):
 if i in known:return known[i]
 with source.open('rb') as f:f.seek(i*chunk);data=f.read(chunk)
 result=call('write_file',{'path':remote+f'/input-parts/part-{i:04d}','content':base64.b64encode(data).decode(),'encoding':'base64','create_parents':True})
 row={'index':i,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'receipt':result}
 with lock:
  with (F/'chunk-receipts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 return row
n=math.ceil(source.stat().st_size/chunk)
first=part(0)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 parts=[first]+list(pool.map(part,range(1,n)))
transfer={'source_sha256':sha(source),'bytes':source.stat().st_size,'chunk_bytes':chunk,'parallel_workers':8,
          'parts':[{k:v for k,v in r.items() if k!='receipt'} for r in parts], 'transfer_seconds':time.monotonic()-started}
(F/'input-transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
files=[upload(B/'run_native_pairs.py','run_native_pairs.py'),upload(F/'inference-plan.json','inference-plan.json'),upload(F/'input-transfer.json','input-transfer.json')]
code='''from pathlib import Path
import hashlib,json
b=Path('.');m=json.loads((b/'input-transfer.json').read_text());out=b/'model-inputs.jsonl.gz';h=hashlib.sha256();total=0
if out.exists():
 assert out.stat().st_size==m['bytes'] and hashlib.sha256(out.read_bytes()).hexdigest()==m['source_sha256']
else:
 with (b/'input-assemble.tmp').open('wb') as f:
  for row in m['parts']:
   data=(b/'input-parts'/('part-%04d'%row['index'])).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];f.write(data);h.update(data);total+=len(data)
 assert total==m['bytes'] and h.hexdigest()==m['source_sha256'];(b/'input-assemble.tmp').replace(out)
for row in m['parts']:(b/'input-parts'/('part-%04d'%row['index'])).unlink(missing_ok=True)
print(json.dumps({'full_input_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'no_inference_launched':True}))
'''
assembly=call('run_command',{'command':'/dataset/miniforge3/envs/vlm-det/bin/python -c '+shlex.quote(code),'cwd':remote,'timeout_seconds':30,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''}})
assert assembly['exit_code']==0,assembly
result={'plan_sha256':sha(F/'inference-plan.json'),'files':files,'input_assembly':assembly,'no_inference_launched':True,'seconds':time.monotonic()-started}
(F/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'plan_sha256':result['plan_sha256'],'estimated_seconds':estimated,'deployment_seconds':result['seconds'],'no_inference_launched':True}))
