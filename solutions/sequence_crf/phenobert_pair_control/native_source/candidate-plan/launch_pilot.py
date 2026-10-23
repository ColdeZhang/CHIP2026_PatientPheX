from pathlib import Path
import base64,json,hashlib,sys
B=Path(__file__).resolve().parent
N=B.parent/'phenobert-pair-native'
P=B/'pilot';P.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (P/'job.json').exists()
summary=json.loads((B/'summary.json').read_text())
env=json.loads((N/'cpu-plan.json').read_text())['env']
env['CUDA_VISIBLE_DEVICES']='GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1'
plan={'scope':'512 immutable real input-only pairs; throughput and finite numerical interface only; no gold/quality/train',
      'model_root':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/phenobert-pair-native',
      'gpu_uuid':env['CUDA_VISIBLE_DEVICES'],'rows':512,'batch_size':64,
      'input_file':'pilot-inputs.jsonl.gz','input_sha256':sha(B/'pilot-inputs.jsonl.gz'),
      'driver_sha256':sha(B/'run_native_pairs.py'),
      'source_index_sha256':sha(N/'upstream-source-index.json'),
      'weight_sha256':'353fc58889b2e59ddb9bf845c037283247268812ff3959cead118f35db90bea8',
      'census_plan_sha256':sha(B/'plan.json'),'census_summary_sha256':sha(B/'summary.json'),
      'native_review_sha256':'5a8a718520a61d44c0764e5f84c9bf87d8a103474acf6ce8d157c61625cd068a',
      'time_cap_seconds':120,'TERM_grace_seconds':15,'env':env,
      'precision':'native FP32, TF32 disabled, softmax dim1 at T=1; no truncate',
      'sampling':summary['pilot_sampling'],'source_shims':[]}
(P/'inference-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
c=Client();c.initialize();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
 assert not r.get('isError'),r
 return r['structuredContent']
records=[]
for p in [B/'run_native_pairs.py',B/'pilot-inputs.jsonl.gz',P/'inference-plan.json']:
 data=p.read_bytes();assert len(data)<=524288
 result=call('write_file',{'path':'phenobert-pair-pilot/'+p.name,'content':base64.b64encode(data).decode(),'encoding':'base64','create_parents':True})
 records.append({'name':p.name,'sha256':sha(p),'bytes':len(data),'receipt':result})
(P/'deployment.json').write_text(json.dumps(records,indent=2)+'\n')
args={'command':'timeout --signal=TERM --kill-after=15s 120s /dataset/miniforge3/envs/vlm-det/bin/python run_native_pairs.py',
      'cwd':'phenobert-pair-pilot','env':env}
result=call('start_process',args)
(P/'job.json').write_text(json.dumps({'plan_sha256':sha(P/'inference-plan.json'),'arguments':args,'result':result},indent=2)+'\n')
print(json.dumps({'plan_sha256':sha(P/'inference-plan.json'),'job':result}))
