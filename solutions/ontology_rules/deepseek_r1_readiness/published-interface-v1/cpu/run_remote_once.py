from pathlib import Path
import sys,json,hashlib,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
C=Path(__file__).parent;D=C.parents[1];M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/published-interface-v1/cpu';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';runtime=json.loads((D/'gpu-probe-single4096/plan.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):
 with (C/n).open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
files={n:C/n for n in ['prepare_cpu.py','plan.json']};save('intent.json',dict(at=time.time(),files={n:sha(p) for n,p in files.items()},scope='hiddenCPU59s+kill1/publishedrawgeneric80inputonly/noGPUweightsnewprediction'));c=Client();c.initialize()
for n,p in files.items():
 r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/'+n,content=p.read_text(),encoding='utf8',create_parents=True)});save(n+'.upload.json',r);assert not r.get('isError')
code="from pathlib import Path;import hashlib;p=Path("+repr(root+rel)+");e="+repr({n:sha(p) for n,p in files.items()})+";assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in e.items());assert not (p/'actual').exists();compile((p/'prepare_cpu.py').read_text(),'prepare_cpu.py','exec')"
command='python3 -c '+shlex.quote(code)+' && timeout --signal=TERM --kill-after=1s 59s '+shlex.quote(runtime['python'])+' -u '+shlex.quote(root+rel+'/prepare_cpu.py')+' --plan '+shlex.quote(root+rel+'/plan.json');env=dict(runtime['GPU_environment'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');save('execution-intent.json',dict(at=time.time(),command=command,env=env));r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=command,env=env,timeout_seconds=60)});save('actual-tool.json',r);v=un(r);print(json.dumps({'exit_code':v.get('exit_code'),'stdout':v.get('stdout'),'stderr_tail':v.get('stderr','')[-1500:]}));assert v['exit_code']==0
result=un(c.request('tools/call',{'name':'read_file','arguments':dict(mount_id=M,path=rel+'/actual/result.json',offset=0,limit=1000000)}));b=result['content'].encode();assert hashlib.sha256(b).hexdigest()==result['sha256'];(C/'actual-result.json').write_bytes(b);data=json.loads(b);out=C/'recovered';out.mkdir(exist_ok=False)
# Small human/plan inputs first; full encoded gzip remains SHA-bound in persistent own namespace.
for n,x in data['outputs'].items():
 if n.endswith('.gz'):continue
 r=un(c.request('tools/call',{'name':'read_file','arguments':dict(mount_id=M,path=rel+'/actual/'+n,offset=0,limit=1000000)}));b=r['content'].encode();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']==r['sha256'];(out/n).write_bytes(b)
print(json.dumps({'status':data['status'],'seconds':data['program_seconds'],'result_sha256':sha(C/'actual-result.json'),'selected':data['selected'],'lengths':data['native_lengths']}))
