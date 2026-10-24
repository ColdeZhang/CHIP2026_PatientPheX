from pathlib import Path
import sys, json, hashlib, shlex, base64, os, time
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
F=Path(__file__).parent
M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
ROOT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
REL='research/qwen35-sft32-full-report-feasibility/input-cpu-v1'
def save(name,v):
    with (F/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
    assert not (F/'deployment-intent.json').exists(), 'Inspect original intent rather than retry'
    mf=json.loads((F/'deployment-manifest.json').read_text());b=(F/'source-inputs.tar.gz').read_bytes()
    assert len(b)==mf['bundle']['bytes'] and hashlib.sha256(b).hexdigest()==mf['bundle']['sha256']
    save('deployment-intent.json',dict(at_epoch=time.time(),manifest_sha256=hashlib.sha256((F/'deployment-manifest.json').read_bytes()).hexdigest(),target=REL))
    c=Client();c.initialize()
    r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=REL+'/source-inputs.tar.gz',content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)})
    save('upload-tool.json',r)
    assert not r.get('isError'), 'UNKNOWN may have written; stop and inspect existing file'
    code='''from pathlib import Path
import os,json,hashlib,tarfile
p=Path(PARAM['root']);m=PARAM['manifest']
assert os.getuid()==p.stat().st_uid==1006
os.chmod(p,0o700)
b=(p/'source-inputs.tar.gz').read_bytes()
assert len(b)==m['bundle']['bytes'] and hashlib.sha256(b).hexdigest()==m['bundle']['sha256']
with tarfile.open(p/'source-inputs.tar.gz','r:gz') as t:
 members=t.getmembers();assert len(members)==len(m['files'])
 assert set(x.name for x in members)==set(m['files'])
 for x in members:
  assert x.isfile() and x.name==Path(x.name).name
  b=t.extractfile(x).read();e=m['files'][x.name]
  assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
  target=p/x.name
  if target.exists():assert target.read_bytes()==b
  else:
   with target.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for n in m['files']:
 if n.endswith('.py'):compile((p/n).read_text(),n,'exec')
assert not (p/'CPU-start-intent.json').exists() and not (p/'result.json').exists()
print(json.dumps({'status':'ACTUAL_SOURCE_INPUT_SHA_COMPILE_NO_MODEL','files':m['files']}))
'''
    code='PARAM='+repr({'root':ROOT+'/'+REL,'manifest':mf})+'\n'+code
    compile(code,'remote-deployment.py','exec')
    (F/'remote-deployment.py').write_text(code)
    r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),timeout_seconds=30,env={'CUDA_VISIBLE_DEVICES':''})});save('deployment-tool.json',r)
    assert not r.get('isError') and un(r)['exit_code']==0
    save('CPU-start-intent.json',dict(at_epoch=time.time(),main_seconds=60,no_model=True,original_namespace=REL))
    wrapper='''from pathlib import Path
import subprocess,time,json,os
p=Path(ROOT)
with (p/'CPU-start-intent.json').open('x') as f:json.dump({'at_epoch':time.time(),'cap_seconds':60},f);f.flush();os.fsync(f.fileno())
t=time.monotonic()
v=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5s','60s',PYTHON,'-u',str(p/'inspect_inputs.py')],cwd=p,text=True,capture_output=True)
for name,value in [('stdout.log',v.stdout),('stderr.log',v.stderr)]:
 with (p/name).open('x') as f:f.write(value);f.flush();os.fsync(f.fileno())
r={'returncode':v.returncode,'wrapper_seconds':time.monotonic()-t,'model_loaded':False}
with (p/'CPU-complete.json').open('x') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({'wrapper':r,'stdout':v.stdout,'stderr':v.stderr}))
'''
    wrapper='ROOT='+repr(ROOT+'/'+REL)+'\nPYTHON='+repr('/home/zhangyuheng/projects/ScratchVL/.venv/bin/python')+'\n'+wrapper
    compile(wrapper,'remote-wrapper.py','exec');(F/'remote-wrapper.py').write_text(wrapper)
    env={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':ROOT+'/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:'+ROOT+'/case-retrieval/round2/runtime'}
    r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(wrapper),timeout_seconds=75,env=env)});save('CPU-tool.json',r)
    assert not r.get('isError'), 'Unknown original CPU execution; read original status, never repeat'
    v=un(r);assert v['exit_code']==0
    result=json.loads(v['stdout']);save('CPU-complete.json',result['wrapper'])
    assert result['wrapper']['returncode']==0, result['stderr'][-3000:]
    value=json.loads(result['stdout']);save('actual-result.json',value);print(json.dumps(value))
if __name__=='__main__':main()
