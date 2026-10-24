from pathlib import Path
import sys,json,hashlib,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
C=Path(__file__).parent;P=C.parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1/codec-longest-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';runtime=json.loads((P/'gpu/plan.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):
 with (C/n).open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
files={n:C/n for n in ['normalized_final.py','strict_final.py','check_existing_first_cpu.py','codec-plan.json']};save('cpu-intent.json',dict(at=time.time(),sources={n:sha(p) for n,p in files.items()},scope='stdlibonly29+kill1,existingfirst746,newcodecnoNN'));c=Client();c.initialize()
for n,p in files.items():
 r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/'+n,content=p.read_text(),encoding='utf8',create_parents=True)});save(n+'.upload.json',r);assert not r.get('isError')
check="from pathlib import Path;import hashlib;p=Path("+repr(root+rel)+");e="+repr({n:sha(p) for n,p in files.items()})+";assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in e.items());assert not (p/'cpu-result.json').exists();[compile((p/n).read_text(),n,'exec') for n in e if n.endswith('.py')]"
command='python3 -c '+shlex.quote(check)+' && timeout --signal=TERM --kill-after=1s 29s python3 -u '+shlex.quote(root+rel+'/check_existing_first_cpu.py');save('cpu-execution-intent.json',dict(at=time.time(),command=command));r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=command,env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},timeout_seconds=35)});save('cpu-tool.json',r);v=un(r);print(json.dumps({'exit_code':v.get('exit_code'),'stdout':v.get('stdout'),'stderr':v.get('stderr')}));assert v['exit_code']==0
r=un(c.request('tools/call',{'name':'read_file','arguments':dict(mount_id=M,path=rel+'/cpu-result.json',offset=0,limit=100000)}));b=r['content'].encode();assert hashlib.sha256(b).hexdigest()==r['sha256'];(C/'actual-cpu-result.json').write_bytes(b);print(json.dumps({'result_sha256':sha(C/'actual-cpu-result.json'),'codec_sha256':sha(C/'normalized_final.py')}))
