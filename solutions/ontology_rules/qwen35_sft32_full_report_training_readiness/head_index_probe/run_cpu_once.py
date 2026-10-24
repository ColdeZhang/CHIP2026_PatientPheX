"""One CUDA-hidden CPU helper contract via existing Torch, no remote files/GPU."""
from pathlib import Path
import datetime,hashlib,json,shlex,sys,zlib,base64
D=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
payload={name:(D/name).read_text() for name in ['indexed_head.py','cpu_checks.py']}
source="import sys,types,json,zlib,base64\npayload=json.loads(zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())+")))\nm=types.ModuleType('indexed_head');m.__file__='indexed_head.py';sys.modules['indexed_head']=m\nexec(compile(payload['indexed_head.py'],'indexed_head.py','exec'),m.__dict__)\nexec(compile(payload['cpu_checks.py'],'cpu_checks.py','exec'),{'__name__':'__main__'})"
refs={n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in payload.items()}
intent={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':refs,'cpu_only':True,'GPU_started':False,'remote_files_written':False,'mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','cap_seconds':30}
with (D/'cpu-run-intent.json').open('x') as f:json.dump(intent,f,indent=2);f.write('\n')
c=Client();c.initialize()
r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':intent['mount_id'],'command':shlex.join(['/usr/bin/timeout','30s','/home/zhangyuheng/projects/ScratchVL/.venv/bin/python','-c',source]),'cwd':'','env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},'timeout_seconds':40}})
v=r.get('structuredContent') or json.loads(r['content'][0]['text'])
with (D/'cpu-tool-result.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
if v.get('exit_code')!=0 or r.get('isError'):
 print(json.dumps({'status':'FAILED_OR_UNKNOWN','exit_code':v.get('exit_code'),'stdout':v.get('stdout'),'stderr':v.get('stderr')}));raise SystemExit(1)
result=json.loads(v['stdout']);result['source_files']=refs
with (D/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['status','torch','CUDA_initialized','seconds','training_cases','head_cases','negative_checks']}))
