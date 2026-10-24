"""Small immutable F0 source/CPU metadata deployment; no GPU/model/tokenizer run."""
from pathlib import Path
import base64,hashlib,json,os,sys,shlex,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9';J=json.loads((P/'deployment-manifest.json').read_text());remote=J['remote'];relative=remote.split('/CHIP2026_PatientPheX/',1)[1]
def save(n,v):
 with (P/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(v):assert not v.get('isError'),v;return v.get('structuredContent') or json.loads(v['content'][0]['text'])
b=(P/'deployment.tar.gz').read_bytes();expected={v['name']:{k:v['local'][k] for k in ['bytes','sha256']} for v in J['files']}
save('deployment-intent.json',{'at':time.time(),'mount':M,'source_bundle_SHA':hashlib.sha256(b).hexdigest(),'no_GPU_start':True,'files':expected})
c=Client();c.initialize();r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=relative+'/deployment.tar.gz',content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)});save('deployment-upload.json',r);un(r)
code='''from pathlib import Path
import json,hashlib,os,tarfile,io,shutil,inspect,time
root=Path(REMOTE);expected=EXPECTED
assert os.getuid()==1045 and root.stat().st_uid==1045
os.chmod(root,0o700)
b=(root/'deployment.tar.gz').read_bytes();assert hashlib.sha256(b).hexdigest()==BUNDLE_SHA
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 members=t.getmembers();assert {m.name for m in members}==set(expected) and len(members)==len(expected)
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  b=t.extractfile(m).read();e=expected[m.name];assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b,'Never overwrite different deployed bytes'
  else:
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert p.stat().st_uid==1045
for n,e in expected.items():
 p=root/n;assert p.stat().st_size==e['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
 if n.endswith('.py'):compile(p.read_text(),n,'exec')
assert not (root/'wrapper-started.json').exists() and not (root/'wrapper-result.json').exists() and not (root/'terminal').exists()
import torch,transformers,tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
cpu=json.loads((root/'metadata/CPU_result.json').read_text());yarn=json.loads((root/'metadata/yarn_result.json').read_text())
runtime={'torch':str(torch.__version__),'transformers':transformers.__version__,'tokenizers':tokenizers.__version__};assert runtime==cpu['runtime'];assert not torch.cuda.is_initialized()
generic=hashlib.sha256(Path(inspect.getsourcefile(PreTrainedTokenizerFast)).read_bytes()).hexdigest();rope=hashlib.sha256(Path(inspect.getsourcefile(ROPE_INIT_FUNCTIONS['yarn'])).read_bytes()).hexdigest();assert generic==cpu['generic_class_source_SHA'] and rope==yarn['native_full_source_sha256']
model=Path('/dev/shm/patientphex-r1-f0-1045-20261024/models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6')
progress=model/'copy-progress.json'
print(json.dumps({'status':'ACTUAL_DEPLOYMENT_17_SHA_COMPILE_RUNTIME_SOURCE_PASS_NO_GPU','uid':os.getuid(),'files':expected,'runtime':runtime,'generic_class_source_SHA':generic,'rotary_source_SHA':rope,'CUDA_initialized':torch.cuda.is_initialized(),'SHM_free':shutil.disk_usage('/dev/shm').free,'copy_progress':json.loads(progress.read_text()) if progress.exists() else None,'copy_ACK_exists':(model/'asset-copy-ack.json').exists()}))
'''.replace('REMOTE',repr(remote)).replace('EXPECTED',repr(expected)).replace('BUNDLE_SHA',repr(hashlib.sha256(b).hexdigest()))
plan=json.loads((P/'plan.pending.json').read_text());r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=shlex.quote(plan['python'])+' -c '+shlex.quote(code),cwd=relative,env={'CUDA_VISIBLE_DEVICES':'',**plan['GPU_environment'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},timeout_seconds=45)});save('actual-deployment-tool.json',r);x=un(r);assert x['exit_code']==0,x;v=json.loads(x['stdout']);save('actual-deployment.json',v);print(json.dumps({k:v[k] for k in ['status','runtime','CUDA_initialized','SHM_free','copy_progress','copy_ACK_exists']}))
