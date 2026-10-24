from pathlib import Path
import json,sys,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
G=Path(__file__).parent;D=G.parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/gpu-probe-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,obj):
 with (G/n).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':dict(mount_id=M,**args)})
 if r.get('isError'):save('deploy-error-'+str(time.time_ns())+'.json',r);raise RuntimeError('Inspect actual written source inventory before recovery')
 return r
files={n:G/n for n in ['run_probe.py','launch_probe.py','strict_final.py','plan.template.json']};files['yarn-result.json']=D/'yarn-semantics/actual-yarn-result.json';assert not (G/'source-deploy-intent.json').exists();save('source-deploy-intent.json',dict(at=time.time(),remote=rel,expected={n:sha(p) for n,p in files.items()},no_start_or_model=True));c=Client();c.initialize()
for n,p in files.items():
 r=call('write_file',dict(path=rel+'/'+n,content=p.read_text(),encoding='utf8',create_parents=True));save(n+'.upload-receipt.json',r)
code="from pathlib import Path;import json,hashlib,os;p=Path("+repr(root+rel)+");expected="+repr({n:sha(p) for n,p in files.items()})+";assert os.getuid()==1006 and p.stat().st_uid==1006;os.chmod(p,0o700);actual={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in expected};assert actual==expected;[compile((p/n).read_text(),n,'exec') for n in ['run_probe.py','launch_probe.py','strict_final.py']];assert json.loads((p/'plan.template.json').read_text())['execution_ready'] is False;print(json.dumps({'status':'ACTUAL_CPU_SOURCE_SHA_AND_COMPILE_PASS_NO_GPU_START','uid':os.getuid(),'files':actual}))"
r=call('run_command',dict(command='python3 -c '+shlex.quote(code),timeout_seconds=30));save('source-deploy-check.json',r);assert un(r)['exit_code']==0;print(un(r)['stdout'])
