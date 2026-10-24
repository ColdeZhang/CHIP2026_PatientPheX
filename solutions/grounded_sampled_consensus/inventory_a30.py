"""One read-only target identity inventory; no GPU, tokenization or model loading."""
import hashlib,json,os,shlex,sys,time
from pathlib import Path
from prepare import S,D,P,ref,save

REMOTE_CODE=r'''
import hashlib,importlib.util,json,os,subprocess
from pathlib import Path
cfg=CONFIG;model=Path(cfg['model']);prior=cfg['prior'];actual={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(p):p=Path(p);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
base_stats={p:v for p,v in prior['asset_stats'].items() if p.startswith(str(model)+'/')}
assert len(base_stats)>=10
for p,v in base_stats.items():actual[p]=stat(p);assert actual[p]==v,(p,'base identity changed; do not silently bless')
import torch,transformers,peft,tokenizers
import transformers.models.qwen3_5.modeling_qwen3_5 as m
import transformers.generation.utils as g
versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__)
assert versions==cfg['versions'] and not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
runtime=dict(qwen_modeling=sha(m.__file__),generation_utils=sha(g.__file__))
assert runtime==prior['runtime_sha256'];assert importlib.util.find_spec('fla') is None and importlib.util.find_spec('causal_conv1d') is None
tokens={}
for n in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json']:
 item=cfg['published'][n];p=model/n;assert p.stat().st_size==item['size'] and sha(p)==item['sha256'];tokens[n]=dict(bytes=p.stat().st_size,sha256=sha(p))
cache={}
for name in cfg['cache_paths']:
 root=Path(name);files={}
 for rel,item in cfg['adapter_files'].items():
  q=root/rel
  if q.is_file():files[rel]=dict(bytes=q.stat().st_size,sha256=sha(q))
 cache[name]=dict(exists=root.exists(),files=files,complete_exact=files==cfg['adapter_files'])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
v=dict(status='PASS_SAME_NATIVE_A30_ASSETS_NO_GPU_NO_TOKENIZATION',versions=versions,runtime_sha256=runtime,base_stats_same_as_prior_SHA_certificate=True,base_actual_stats=actual,tokenizer_assets=tokens,cache=cache,GPU_inventory=gpu,apps=apps,free_bytes=os.statvfs('.').f_bavail*os.statvfs('.').f_frsize,no_GPU_model_or_tokenization=True)
print(json.dumps(v))
'''

def main():
 target=D/'a30-host-inventory-v1.json';assert not target.exists();D.mkdir(parents=True,exist_ok=True)
 old=Path('/workspace/patientphex-runs/round2-20261023/research/nonthinking-sampled-consensus-full64-20261024')
 prior=json.loads((old/'cpu-complete.json').read_text());plan=json.loads((old/'gpu-plan.json').read_text())
 manifest=json.loads(Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/model-source-manifest.json').read_text())
 cp=json.loads((S/'copy_a30/copy-plan.json').read_text())
 config=dict(model=plan['model'],prior=prior,versions=plan['versions'],published={r['name']:r for r in manifest['files']},adapter_files=cp['files'],cache_paths=[cp['destination'],P+'/grounded-generation-v1/F0-inference-cache/epoch-002',P+'/grounded-generation-v1/formal-v1/output/epoch-002'])
 code='CONFIG='+repr(config)+'\n'+REMOTE_CODE
 env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':plan['env']['PYTHONPATH'],'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 command=shlex.join(['env']+[k+'='+v for k,v in env.items()]+[plan['python'],'-c',code])
 args=dict(mount_id='5a9955da-5240-41cc-9e52-05d3d8da8a3c',cwd='.',command=command,timeout_seconds=55)
 tag=D/('a30-readonly-'+str(time.time_ns()));save(tag.with_suffix('.intent.json'),dict(tool='run_command',arguments=args,source=ref(__file__),prior_CPU=ref(old/'cpu-complete.json'),prior_plan=ref(old/'gpu-plan.json')))
 sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
 c=Client();c.initialize();result=c.request('tools/call',dict(name='run_command',arguments=args))
 text=json.dumps(result)
 for key in ['MCP_URL','MCP_TOKEN']:
  if os.environ.get(key):text=text.replace(os.environ[key],'[REDACTED]')
 result=json.loads(text);save(tag.with_suffix('.tool.json'),result)
 assert not result.get('isError'),'Read-only original inventory failed; preserve error, no model start'
 actual=result.get('structuredContent')
 if actual is None:actual=json.loads(next(x['text'] for x in result['content'] if x['type']=='text'))
 assert actual['exit_code']==0,actual.get('stderr')
 value=json.loads(actual['stdout']);assert value['status']=='PASS_SAME_NATIVE_A30_ASSETS_NO_GPU_NO_TOKENIZATION'
 save(target,dict(actual=value,source=ref(__file__),actual_tool=ref(tag.with_suffix('.tool.json')),prior_CPU=ref(old/'cpu-complete.json'),prior_plan=ref(old/'gpu-plan.json'),no_model_no_tokenization=True))
 print(json.dumps(ref(target)))

if __name__=='__main__':main()
