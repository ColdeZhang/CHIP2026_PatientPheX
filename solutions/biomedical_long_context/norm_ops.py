"""Deploy/start original Sap normalization only after actual full GP completion."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import time
from rpc import call
D=Path('/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024/normalization-v1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','start','observe']);a=ap.parse_args()
    p=json.loads((D/'package.json').read_text());cfg=json.loads(Path(p['plan']['path']).read_text())
    mount=p['mount'];rel=p['relative'];remote=p['remote']
    def run(code,tag):
        r=call(D,'run_command',{'mount_id':mount,'cwd':'.','command':'python3 -c '+shlex.quote(code),'timeout_seconds':35},tag)
        assert r.get('exit_code',0)==0,r;return r
    if a.action=='deploy':
        b=(D/'source.zip').read_bytes();assert hashlib.sha256(b).hexdigest()==p['archive']['sha256']
        call(D,'write_file',{'mount_id':mount,'path':rel+'/source.zip','content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True},'upload')
        code='''import pathlib,zipfile,json,hashlib,os
r=pathlib.Path(%r);b=(r/'source.zip').read_bytes();assert hashlib.sha256(b).hexdigest()==%r
z=zipfile.ZipFile(r/'source.zip');members=%r;assert set(z.namelist())==set(members)
for n,v in members.items():
 b=z.read(n);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 with (r/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'NORMALIZATION_SOURCE_EXACT','members':len(members)}))'''%(remote,p['archive']['sha256'],p['members'])
        r=run(code,'extract');(D/'deployment.json').write_text(json.dumps(r,indent=2));print(r['stdout'])
    elif a.action=='start':
        assert (D/'deployment.json').exists()
        t={'created_unix':time.time(),'launch_not_after_unix':time.time()+300,'plan':{'path':remote+'/norm-plan.json','bytes':p['plan']['bytes'],'sha256':p['plan']['sha256']}}
        with (D/'start.intent.json').open('x') as f:json.dump(t,f,indent=2)
        call(D,'write_file',{'mount_id':mount,'path':rel+'/launch-ticket.json','content':json.dumps(t),'encoding':'utf-8','create_parents':False},'ticket')
        r=call(D,'start_process',{'mount_id':mount,'cwd':rel,'command':shlex.join([cfg['python'],'-u',remote+'/driver.py']),'env':cfg['environment']},'start')
        (D/'started.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
    else:
        code='''import pathlib,json
r=pathlib.Path(%r);o={}
for n in ['job/started.json','job/ended.json','linked/complete.json']:
 p=r/n
 if p.exists():o[n]=json.loads(p.read_text())
p=r/'job/stderr'
if p.exists():o['stderr_tail']=p.read_text()[-4000:]
print(json.dumps(o))'''%remote
        r=run(code,'observe');(D/('observation-'+str(time.time_ns())+'.json')).write_text(json.dumps(r,indent=2));print(r['stdout'])


if __name__=='__main__':main()
