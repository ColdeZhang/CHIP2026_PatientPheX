"""Durable bounded MCP operations; UNKNOWN never dispatches a replacement start."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

R=Path('/workspace/patientphex-runs/round2-20261023')
D=R/'research/supervised-linker-readiness-20261024/probe-v2'
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client


def call(name,args,tag):
    h=D/'transport';h.mkdir(exist_ok=True)
    target=h/(str(time.time_ns())+'-'+tag)
    target.with_suffix('.intent.json').write_text(json.dumps(dict(name=name,arguments=args),indent=2)+'\n')
    client=Client();client.initialize()
    try:r=client.request('tools/call',dict(name=name,arguments=args))
    except Exception as e:
        target.with_suffix('.unknown.json').write_text(json.dumps(dict(error_type=type(e).__name__,action='DO_NOT_REPEAT_START; inspect original exclusive/job path'))+'\n');raise
    target.with_suffix('.response.json').write_text(json.dumps(r,indent=2)+'\n')
    if r.get('isError'):raise RuntimeError(json.dumps(r)[:1000])
    v=r.get('structuredContent')
    if v is None:v=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
    return v


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','cpu','start','observe','collect']);args=ap.parse_args()
    package=json.loads((D/'package.json').read_text());mount=package['mount'];rel=package['relative'];remote=package['remote']
    plan=json.loads(Path(package['plan']['path']).read_text())
    def run(code,tag,timeout=40):
        v=call('run_command',dict(mount_id=mount,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=timeout),tag)
        assert v.get('exit_code',0)==0,v
        return v
    if args.action=='deploy':
        data=(D/'package.zip').read_bytes();assert hashlib.sha256(data).hexdigest()==package['archive']['sha256']
        chunks=[]
        for i,start in enumerate(range(0,len(data),262144)):
            part=data[start:start+262144];name='package.part%03d'%i;chunks.append(name)
            call('write_file',dict(mount_id=mount,path=rel+'/'+name,content=base64.b64encode(part).decode(),encoding='base64',create_parents=True),'upload-'+str(i))
        code='''import pathlib,hashlib,zipfile,io,json,os
root=pathlib.Path(%r);blob=b''.join((root/x).read_bytes() for x in %r)
assert hashlib.sha256(blob).hexdigest()==%r
z=zipfile.ZipFile(io.BytesIO(blob));members=%r
assert set(z.namelist())==set(members)
for name,r in members.items():
 b=z.read(name);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 p=root/name
 with open(p,'xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'EXTRACTED_ALL_MEMBERS','members':len(members),'archive_sha256':hashlib.sha256(blob).hexdigest()}))
'''%(remote,chunks,package['archive']['sha256'],package['members'])
        v=run(code,'extract');(D/'deployment-complete.json').write_text(json.dumps(v,indent=2)+'\n');print(v['stdout'])
    elif args.action=='cpu':
        cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=']+[k+'='+v for k,v in plan['environment'].items()]+[plan['python'],'-u',remote+'/native_cpu.py'])
        v=call('run_command',dict(mount_id=mount,cwd=rel,command=cmd,timeout_seconds=50),'cpu')
        (D/'cpu-response.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));assert v.get('exit_code',0)==0,v
    elif args.action=='start':
        marker=D/'start.intent.json'
        ticket=dict(created_unix=time.time(),launch_not_after_unix=time.time()+300,
            plan=dict(path=remote+'/execution-plan.json',bytes=package['plan']['bytes'],sha256=package['plan']['sha256']))
        with open(marker,'x') as f:json.dump(ticket,f,indent=2)
        call('write_file',dict(mount_id=mount,path=rel+'/launch-ticket.json',content=base64.b64encode(json.dumps(ticket).encode()).decode(),encoding='base64',create_parents=False),'launch-ticket')
        v=call('start_process',dict(mount_id=mount,cwd=rel,command=shlex.join([plan['python'],'-u',remote+'/driver.py']),env=plan['environment']),'start')
        (D/'start.response.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
    elif args.action=='observe':
        code='''import pathlib,json
r=pathlib.Path(%r);d={}
for n in ['job/started.json','job/ended.json','job/expired-before-entry.json','probe-output/complete.json']:
 p=r/n
 if p.exists():d[n]=json.loads(p.read_text())
p=r/'job/stderr'
if p.exists():d['stderr_tail']=p.read_text()[-3000:]
print(json.dumps(d))'''%remote
        v=run(code,'observe');target=D/('observation-'+str(time.time_ns())+'.json');target.write_text(json.dumps(v,indent=2)+'\n');print(v['stdout'])
    else:
        for name in ['cpu-complete.json','job/fresh-gate.json','job/started.json','job/ended.json','probe-output/complete.json','probe-output/loading.json']:
            v=call('read_file',dict(mount_id=mount,path=rel+'/'+name,offset=0,limit=32768),'read-'+name.replace('/','_'))
            raw=v['content'];data=base64.b64decode(raw) if v.get('encoding')=='base64' else raw.encode()
            p=D/'collected'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        print(json.dumps(dict(status='COLLECTED_NECESSARY_SMALL_METADATA',files=6,large_checkpoint_not_copied=True)))


if __name__=='__main__':main()
