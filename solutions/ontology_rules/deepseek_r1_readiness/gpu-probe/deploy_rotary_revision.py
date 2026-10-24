from pathlib import Path
import json,sys,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
G=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/gpu-probe-v1';root='/home/zhangyuheng/projects/CHIP2026_PheX/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):
 with (G/n).open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
files=['run_probe.py','plan.template.json'];save('rotary-redeploy-intent.json',dict(at=time.time(),expected_old={n:sha(G/'prestart-rotary-hard-equality'/n) for n in files},new={n:sha(G/n) for n in files},no_start=True));c=Client();c.initialize()
for n in files:
 r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/'+n,content=(G/n).read_text(),encoding='utf8',expected_sha256=sha(G/'prestart-rotary-hard-equality'/n),create_parents=False)});save('rotary-redeploy-'+n+'.receipt.json',r);assert not r.get('isError')
code="from pathlib import Path;import json,hashlib;p=Path("+repr('/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel)+");d=json.loads((p/'plan.template.json').read_text());assert d['execution_ready'] is False;assert not (p/'wrapper-started.json').exists();assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in d['sources'].items());compile((p/'run_probe.py').read_text(),'run_probe.py','exec');print(json.dumps({'status':'ACTUAL_NARROW_ROTARY_SOURCE_SHA_COMPILE_PASS_NO_GPU_START','template_sha256':hashlib.sha256((p/'plan.template.json').read_bytes()).hexdigest(),'sources':d['sources']}))"
r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),timeout_seconds=30)});save('rotary-redeploy-check.json',r);v=un(r);assert v['exit_code']==0;print(v['stdout'])
