"""Deploy immutable owned formal sources; same actual input/schema path hidden-CUDA CPU only."""
from pathlib import Path
import json,sys,hashlib,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REL='research/deepseek-r1-0528-qwen3-8b-readiness/full80-v1';ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (F/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 plan=json.loads((F/'plan.json').read_text());assert not (F/'deployment-intent.json').exists();save('deployment-intent.json',{'at':time.time(),'remote':REL,'plan_sha256':sha(F/'plan.json'),'no_GPU_start':True})
 c=Client();c.initialize()
 for name,digest in {**plan['sources'],'plan.json':sha(F/'plan.json')}.items():
  assert sha(F/name)==digest
  r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=REL+'/'+name,content=(F/name).read_text(),encoding='utf8',create_parents=True)});save(name+'.upload.json',r);assert not r.get('isError')
 code="from pathlib import Path;import json,hashlib,os,shutil;p=Path("+repr(ROOT+REL)+");s=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();assert os.getuid()==p.stat().st_uid==1006;os.chmod(p,0o700);assert s(p/'plan.json')=="+repr(sha(F/'plan.json'))+";plan=json.loads((p/'plan.json').read_text());assert all(s(p/n)==h for n,h in plan['sources'].items());[compile((p/n).read_text(),n,'exec') for n in plan['sources']];assert not (p/'wrapper-started.json').exists() and not (p/'terminal').exists();print(json.dumps({'status':'ACTUAL_FORMAL_SOURCE_PLAN_SHA_COMPILE_NO_START','sources':plan['sources'],'plan_sha256':s(p/'plan.json'),'uid':os.getuid(),'disk_free_bytes':shutil.disk_usage(p).free}))"
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)});save('deployment-source-check.json',r);assert not r.get('isError') and un(r)['exit_code']==0
 cmd=shlex.quote(plan['python'])+' -u full80_contract.py --plan plan.json --output actual-input-contract.json'
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=cmd,cwd=REL,env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},timeout_seconds=30)});save('actual-input-contract-tool.json',r);v=un(r);assert not r.get('isError') and v['exit_code']==0;result=json.loads(v['stdout']);save('actual-input-contract.json',result);print(json.dumps(result))
if __name__=='__main__':main()
