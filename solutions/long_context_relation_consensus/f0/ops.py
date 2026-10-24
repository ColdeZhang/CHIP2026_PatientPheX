"""Bounded existing-host I/O, native CPU, and one H69 inference launch."""
import argparse,base64,hashlib,json,shlex,time
from pathlib import Path
from rpc import call
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relation-consensus-v1/F0-v1')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inventory','upload','cpu','start','observe']);a=ap.parse_args()
 pkg=json.loads((D/'package.json').read_text());remote=pkg['remote'];rel=pkg['relative'];mount=pkg['mount'];plan=json.loads(Path(pkg['plan']['path']).read_text())
 def run(code,tag,timeout=30):
  v=call(D,'run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=timeout),tag)
  assert v.get('exit_code',0)==0,v
  (D/(tag+'-ack.json')).write_text(json.dumps(v,indent=2));print(v.get('stdout',''));return v
 if a.mode=='inventory':
  run("import subprocess,json,time,pathlib;print(json.dumps(dict(at=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True),existing_files=[str(p)for p in pathlib.Path(%r).glob('*')],state_exists=pathlib.Path(%r).exists())))"%(remote,plan['checkpoint']['path']),'inventory')
 elif a.mode=='upload':
  raw=(D/'runtime.zip').read_bytes();assert hashlib.sha256(raw).hexdigest()==pkg['archive']['sha256']
  v=call(D,'write_file',dict(mount_id=mount,path=rel+'/runtime.zip',content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True),'upload')
  (D/'upload-ack.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
 elif a.mode=='cpu':
  code='''import pathlib,hashlib,zipfile,os,json,subprocess
r=pathlib.Path(%r);p=r/'runtime.zip';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==%r
members=%r
with zipfile.ZipFile(p)as z:
 assert set(z.namelist())==set(members)
 for n,v in members.items():
  b=z.read(n);assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256']
  q=r/n
  if q.exists():assert q.read_bytes()==b
  else:
   with q.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=r/'cpu-complete.json'
if not receipt.exists():
 p=subprocess.run([%r,'-u',str(r/'native_cpu.py')],cwd=r,env={**os.environ,**%r,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=40)
 print(json.dumps({'cpu_rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));assert p.returncode==0
print(receipt.read_text())
'''%(remote,pkg['archive']['sha256'],pkg['members'],plan['python'],plan['environment'])
  run(code,'native-cpu',timeout=50)
 elif a.mode=='start':
  assert(D/'native-cpu-ack.json').exists()
  ticket=dict(created_unix=time.time(),launch_not_after_unix=time.time()+300,plan=dict(path=remote+'/infer-plan.json',bytes=pkg['plan']['bytes'],sha256=pkg['plan']['sha256']))
  with(D/'start.intent.json').open('x')as f:json.dump(ticket,f,indent=2)
  code='''import pathlib,json,os
p=pathlib.Path(%r);ticket=%r
with p.open('x')as f:json.dump(ticket,f);f.flush();os.fsync(f.fileno())
fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert json.loads(p.read_text())==ticket
print(json.dumps({'status':'EXCLUSIVE_TICKET_DURABLE','ticket':ticket}))
'''%(remote+'/launch-ticket.json',ticket)
  run(code,'launch-ticket')
  v=call(D,'start_process',dict(mount_id=mount,cwd=rel,command=shlex.join([plan['python'],'-u',remote+'/driver.py']),env=plan['environment']),'start')
  (D/'started.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
 else:
  code='''import pathlib,json
r=pathlib.Path(%r);o={}
for n in ['job/started.json','job/ended.json','job/expired-before-entry.json','output/complete.json']:
 p=r/n
 if p.exists():o[n]=json.loads(p.read_text())
p=r/'job/stderr'
if p.exists():o['stderr_tail']=p.read_text()[-3000:]
print(json.dumps(o))'''%remote
  run(code,'observation-'+str(time.time_ns()))

if __name__=='__main__':main()
