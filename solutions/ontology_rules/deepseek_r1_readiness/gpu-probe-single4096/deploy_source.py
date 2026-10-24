from pathlib import Path
import sys,json,hashlib,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
G=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/gpu-probe-single4096-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def save(n,x):
 with (G/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
plan=json.loads((G/'plan.json').read_text());save('source-deploy-intent.json',dict(at=time.time(),remote=rel,sources=plan['sources'],no_model_start=True));c=Client();c.initialize()
for n,h in plan['sources'].items():
 assert hashlib.sha256((G/n).read_bytes()).hexdigest()==h
 r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/'+n,content=(G/n).read_text(),encoding='utf8',create_parents=True)});save(n+'.upload.json',r);assert not r.get('isError')
code="from pathlib import Path;import json,hashlib,os;p=Path("+repr(root+rel)+");e="+repr(plan['sources'])+";assert p.stat().st_uid==os.getuid()==1006;os.chmod(p,0o700);assert not (p/'wrapper-started.json').exists();a={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in e};assert a==e;[compile((p/n).read_text(),n,'exec') for n in e];print(json.dumps({'status':'ACTUAL_SINGLE4096_SOURCE_SHA_PYTHON310_COMPILE_PASS_NO_MODEL','sources':a}))"
r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)});save('source-deploy-check.json',r);v=un(r);assert v['exit_code']==0;print(v['stdout'])
