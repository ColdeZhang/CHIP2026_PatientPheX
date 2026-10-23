from pathlib import Path
import base64,hashlib,io,json,shlex,sys,zipfile
B=Path(__file__).resolve().parent/'pilot'
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
c=Client();c.initialize();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(name,args):
 r=c.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
 assert not r.get('isError'),r
 return r['structuredContent']
j=json.loads((B/'job.json').read_text())
terminal={s:call('read_process_output',{'process_id':j['result']['id'],'stream':s,'cursor':0,'limit':32768}) for s in ('stdout','stderr')}
(B/'process-terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
assert all(not x['running'] for x in terminal.values())
code='''from pathlib import Path
import hashlib,json,subprocess,time,zipfile
b=Path('.')
target='GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
selected=[x for x in gpu.splitlines() if target in x]
assert len(selected)==1 and target not in apps
assert int(selected[0].split(',')[2].strip())<=512
(b/'release.json').write_text(json.dumps({'target':target,'target_gpu_row':selected[0],'target_no_compute_process':True,'time':time.time()},indent=2)+'\\n')
files=[]
with zipfile.ZipFile(b/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['result.json','probabilities.jsonl.gz','fresh-GPU.json','release.json','inference-plan.json','run_native_pairs.py']:
  p=b/name;data=p.read_bytes();files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});z.writestr(name,data)
data=(b/'evidence.zip').read_bytes();print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'files':files}))
'''
r=call('run_command',{'command':'/dataset/miniforge3/envs/vlm-det/bin/python -c '+shlex.quote(code),'cwd':'phenobert-pair-pilot','timeout_seconds':30,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''}})
assert r['exit_code']==0,r
manifest=json.loads(r['stdout']);chunks=[]
for off in range(0,manifest['bytes'],262144):
 x=call('read_file',{'path':'phenobert-pair-pilot/evidence.zip','offset':off,'limit':min(262144,manifest['bytes']-off)})
 assert x['encoding']=='base64'
 chunks.append(base64.b64decode(x['content']))
data=b''.join(chunks);assert len(data)==manifest['bytes'] and hashlib.sha256(data).hexdigest()==manifest['sha256']
out=B/'terminal';out.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
 for item in manifest['files']:
  data=z.read(item['name']);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
  (out/item['name']).write_bytes(data)
(B/'artifacts-transfer.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
