from pathlib import Path
import sys,json,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
C=Path(__file__).parent;D=C.parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/vllm-cpu-compat';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';plan=json.loads((D/'gpu-probe-single4096/plan.json').read_text());public=Path('/workspace/patientphex-runs/round2-20261023/research/r1-serving-config-review/vllm-v0.9.1-rotary_embedding.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o):
 with (C/n).open('x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
files={'check_cpu.py':C/'check_cpu.py','vllm-v0.9.1-rotary_embedding.py':public};save('intent.json',dict(at=time.time(),files={n:sha(p) for n,p in files.items()},CPU_hidden_cap29pluskill1=True,no_model=True));c=Client();c.initialize()
for n,p in files.items():
 r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/'+n,content=p.read_text(),encoding='utf8',create_parents=True)});save(n+'.upload.json',r);assert not r.get('isError')
check="from pathlib import Path;import hashlib;p=Path("+repr(root+rel)+");e="+repr({n:sha(p) for n,p in files.items()})+";assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in e.items());assert not (p/'result.json').exists();compile((p/'check_cpu.py').read_text(),'check_cpu.py','exec')"
command='python3 -c '+shlex.quote(check)+' && timeout --signal=TERM --kill-after=1s 29s '+shlex.quote(plan['python'])+' -u '+shlex.quote(root+rel+'/check_cpu.py')+' --source '+shlex.quote(root+rel+'/vllm-v0.9.1-rotary_embedding.py')+' --config '+shlex.quote(plan['model_dir']+'/config.json')+' --output '+shlex.quote(root+rel+'/result.json')
save('execution-intent.json',dict(at=time.time(),command=command,env=dict(plan['GPU_environment'],CUDA_VISIBLE_DEVICES=''),no_GPU_or_weights=True));r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command=command,env=dict(plan['GPU_environment'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),timeout_seconds=40)});save('actual-tool.json',r);v=un(r);assert v['exit_code']==0;value=json.loads(v['stdout']);print(json.dumps({k:value[k] for k in ['status','program_seconds','vllm_scale','HF_original_scale','HF_derived_scale','all64_frequency_values_exact','CUDA_initialized']}))
r=c.request('tools/call',{'name':'read_file','arguments':dict(mount_id=M,path=rel+'/result.json',offset=0,limit=100000)});v=un(r);b=v['content'].encode();assert hashlib.sha256(b).hexdigest()==v['sha256'];(C/'actual-result.json').write_bytes(b)
