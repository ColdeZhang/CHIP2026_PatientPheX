from pathlib import Path
import sys,json,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
A=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/assets';modelrel='research/deepseek-r1-0528-qwen3-8b-readiness/models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
assert not (A/'deployment-intent.json').exists(), 'Original deployment intent exists; inventory before missing-only recovery'
(A/'deployment-intent.json').write_text(json.dumps({'at_epoch':time.time(),'mount_id':M,'remote_namespace':rel,'scope':'only own fixed-revision downloader inputs; unknown writes require SHA inventory'},indent=2)+'\n')
c=Client();c.initialize()
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':dict(mount_id=M,**args)})
 if r.get('isError'):raise RuntimeError('remote MCP error '+name)
 return r
code="from pathlib import Path; import os,json; assert os.getuid()==1006; p=Path("+repr(root+'research/deepseek-r1-0528-qwen3-8b-readiness')+"); p.mkdir(mode=0o700,parents=True,exist_ok=True); assert p.stat().st_uid==1006; os.chmod(str(p),0o700); print(json.dumps({'uid':os.getuid(),'path':str(p),'mode':oct(p.stat().st_mode & 0o777)}))"
r=call('run_command',{'command':'python3 -c '+shlex.quote(code),'timeout_seconds':30});(A/'private-directory-result.json').write_text(json.dumps(r,indent=2)+'\n')
uploads=[(A/'download_deepseek.py',rel+'/download_deepseek.py'),(A/'expected-files.json',rel+'/expected-files.json'),(A/'derived-tensor-shapes.json',rel+'/derived-tensor-shapes.json')]+[(p,modelrel+'/'+p.name) for p in sorted((A/'seeds').iterdir())]
with (A/'upload-receipts.jsonl').open('a') as log:
 for p,target in uploads:
  b=p.read_bytes();log.write(json.dumps({'status':'WRITE_INTENT','remote':target,'expected_sha256':hashlib.sha256(b).hexdigest()})+'\n');log.flush();os.fsync(log.fileno());r=call('write_file',{'path':target,'content':b.decode(),'encoding':'utf8','create_parents':True});record={'local':str(p),'remote':target,'expected_sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'response':r};log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno());print(json.dumps({'uploaded':target,'bytes':len(b)}),flush=True)
check="from pathlib import Path; import hashlib,json,os; b=Path("+repr(root+rel)+"); m=Path("+repr(root+modelrel)+"); assert os.getuid()==1006; expected="+repr({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,_ in uploads[:3]})+"; got={n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in expected}; assert got==expected; compile((b/'download_deepseek.py').read_text(),'download_deepseek.py','exec'); print(json.dumps({'status':'REMOTE_SOURCE_SHA_AND_PYTHON310_COMPILE_OK','uid':os.getuid(),'checked':got,'model_dir':str(m)}))"
r=call('run_command',{'command':'python3 -c '+shlex.quote(check),'timeout_seconds':30});(A/'remote-source-preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
