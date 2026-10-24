"""Recorded original-package deployment, native CPU, one GPU start and reads."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import time
from rpc import call

D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1/probe-v1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','cpu','start','observe','collect']);a=ap.parse_args()
    pkg=json.loads((D/'package.json').read_text());mount=pkg['mount'];rel=pkg['relative'];remote=pkg['remote']
    plan=json.loads(Path(pkg['plan']['path']).read_text())
    def run(code,tag,timeout=40):
        v=call(D,'run_command',{'mount_id':mount,'cwd':'.','command':'python3 -c '+shlex.quote(code),'timeout_seconds':timeout},tag)
        assert v.get('exit_code',0)==0,v;return v
    if a.action=='deploy':
        data=(D/'package.zip').read_bytes();assert hashlib.sha256(data).hexdigest()==pkg['archive']['sha256'];parts=[]
        for i,start in enumerate(range(0,len(data),262144)):
            name='package.part%03d'%i;parts.append(name)
            call(D,'write_file',{'mount_id':mount,'path':rel+'/'+name,'content':base64.b64encode(data[start:start+262144]).decode(),'encoding':'base64','create_parents':True},'part-'+str(i))
        code='''import pathlib,json,io,zipfile,hashlib,os
r=pathlib.Path(%r);m=pathlib.Path(%r);m.mkdir(parents=True,exist_ok=True)
b=b''.join((r/n).read_bytes() for n in %r);assert hashlib.sha256(b).hexdigest()==%r
z=zipfile.ZipFile(io.BytesIO(b));members=%r;assert set(z.namelist())==set(members)
for n,v in members.items():
 b=z.read(n);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 p=m/n.split('/',1)[1] if n.startswith('assets/') else r/n
 if p.exists():assert p.read_bytes()==b
 else:
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
for p in [r,m]:
 fd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'ALL_DECLARED_MEMBERS_EXTRACTED_EXACT','members':len(members)}))
'''%(remote,pkg['model'],parts,pkg['archive']['sha256'],pkg['members'])
        v=run(code,'extract');(D/'deployment.json').write_text(json.dumps(v,indent=2));print(v['stdout'])
    elif a.action=='cpu':
        cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=']+[k+'='+v for k,v in plan['environment'].items()]+[plan['python'],'-u',remote+'/native_cpu.py'])
        v=call(D,'run_command',{'mount_id':mount,'cwd':rel,'command':cmd,'timeout_seconds':55},'native-cpu')
        (D/'cpu-response.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));assert v.get('exit_code',0)==0
    elif a.action=='start':
        # Admission is bound locally before dispatch, separate from compute180.
        ticket={'created_unix':time.time(),'launch_not_after_unix':time.time()+300,
                'plan':{'path':remote+'/execution-plan.json','bytes':pkg['plan']['bytes'],'sha256':pkg['plan']['sha256']}}
        with (D/'start.intent.json').open('x') as f:json.dump(ticket,f,indent=2)
        call(D,'write_file',{'mount_id':mount,'path':rel+'/launch-ticket.json','content':json.dumps(ticket),'encoding':'utf-8','create_parents':False},'launch-ticket')
        v=call(D,'start_process',{'mount_id':mount,'cwd':rel,'command':shlex.join([plan['python'],'-u',remote+'/driver.py']),'env':plan['environment']},'start')
        (D/'started.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
    elif a.action=='observe':
        code='''import pathlib,json
r=pathlib.Path(%r);o={}
for n in ['job/started.json','job/ended.json','job/expired-before-entry.json','probe-output/complete.json']:
 p=r/n
 if p.exists():o[n]=json.loads(p.read_text())
p=r/'job/stderr'
if p.exists():o['stderr_tail']=p.read_text()[-4000:]
print(json.dumps(o))'''%remote
        v=run(code,'observe');(D/('observation-'+str(time.time_ns())+'.json')).write_text(json.dumps(v,indent=2));print(v['stdout'])
    else:
        names=['cpu-complete.json','job/fresh-gate.json','job/started.json','job/ended.json','probe-output/complete.json','probe-output/loading.json','probe-output/actual-shapes.json']
        for n in names:
            v=call(D,'read_file',{'mount_id':mount,'path':rel+'/'+n,'offset':0,'limit':32768},'read-'+n.replace('/','_'))
            data=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode()
            p=D/'collected'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        print(json.dumps({'status':'COLLECTED_7_NECESSARY_METADATA','large_state_not_downloaded':True}))


if __name__=='__main__':main()
