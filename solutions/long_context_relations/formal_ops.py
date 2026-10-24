"""Small source-only formal deployment; reuse already proven exact input bytes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import time
from rpc import call

D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1/formal-v1')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['deploy','extract','fallback-upload','extract-fallback','start','observe']);a=ap.parse_args()
    pkg=json.loads((D/'package.json').read_text());mount=pkg['mount'];rel=pkg['relative'];remote=pkg['remote']
    plan=json.loads(Path(pkg['plan']['path']).read_text())
    def run(code,tag,timeout=35):
        v=call(D,'run_command',{'mount_id':mount,'cwd':'.','command':'python3 -c '+shlex.quote(code),'timeout_seconds':timeout},tag)
        assert v.get('exit_code',0)==0,v;return v
    if a.action=='fallback-upload':
        blob=(D/'source.zip').read_bytes();assert hashlib.sha256(blob).hexdigest()==pkg['archive']['sha256']
        code='''import pathlib,base64,hashlib,os,json
r=pathlib.Path(%r);r.mkdir(parents=True,exist_ok=True)
b=base64.b64decode(%r);assert hashlib.sha256(b).hexdigest()==%r
p=r/'source-transfer-v2.zip'
if p.exists():assert p.read_bytes()==b
else:
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert p.read_bytes()==b
print(json.dumps({'status':'FIXED_SOURCE_BYTES_FSYNC_REOPEN','path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'model_calls':0}))
'''%(remote,base64.b64encode(blob).decode(),pkg['archive']['sha256'])
        v=run(code,'fixed-source-fallback');(D/'source-fallback-ack.json').write_text(json.dumps(v,indent=2));print(v['stdout'])
    elif a.action in ('deploy','extract','extract-fallback'):
        blob=(D/'source.zip').read_bytes();assert hashlib.sha256(blob).hexdigest()==pkg['archive']['sha256']
        if a.action=='deploy':
            call(D,'write_file',{'mount_id':mount,'path':rel+'/source.zip','content':base64.b64encode(blob).decode(),'encoding':'base64','create_parents':True},'source-upload')
        code='''import pathlib,json,zipfile,hashlib,os
r=pathlib.Path(%r);archive=r/%r;b=archive.read_bytes();assert hashlib.sha256(b).hexdigest()==%r
z=zipfile.ZipFile(archive);members=%r;assert set(z.namelist())==set(members)
for n,v in members.items():
 data=z.read(n);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
 with (r/n).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
data=pathlib.Path(%r).read_bytes();assert hashlib.sha256(data).hexdigest()==%r
with (r/'data.json.gz').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'status':'FORMAL_SOURCE_AND_PRIOR_EXACT_DATA_DEPLOYED','members':len(members),'data_sha256':hashlib.sha256(data).hexdigest(),'model_loads':0}))
'''%(remote,'source-transfer-v2.zip' if a.action=='extract-fallback' else 'source.zip',pkg['archive']['sha256'],pkg['members'],plan['remote_data_source'],plan['data']['sha256'])
        v=run(code,'extract');(D/'deployment.json').write_text(json.dumps(v,indent=2));print(v['stdout'])
    elif a.action=='start':
        assert (D/'deployment.json').exists()
        ticket={'created_unix':time.time(),'launch_not_after_unix':time.time()+300,
                'plan':{'path':remote+'/formal-plan.json','bytes':pkg['plan']['bytes'],'sha256':pkg['plan']['sha256']}}
        with (D/'start.intent.json').open('x') as f:json.dump(ticket,f,indent=2)
        code='''import pathlib,json,os,hashlib
p=pathlib.Path(%r);ticket=%r
with p.open('x') as f:json.dump(ticket,f);f.flush();os.fsync(f.fileno())
fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert json.loads(p.read_text())==ticket
print(json.dumps({'status':'BOUND_ORIGINAL_LAUNCH_TICKET','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ticket':ticket}))
'''%(remote+'/launch-ticket.json',ticket)
        run(code,'launch-ticket-exclusive-fsync',timeout=30)
        v=call(D,'start_process',{'mount_id':mount,'cwd':rel,'command':shlex.join([plan['python'],'-u',remote+'/driver.py']),'env':plan['environment']},'start')
        (D/'started.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
    else:
        code='''import pathlib,json
r=pathlib.Path(%r);o={}
for n in ['job/started.json','job/ended.json','job/expired-before-entry.json','output/progress.json','output/inference-progress.json','output/complete.json','output/final-state-ACK.json']:
 p=r/n
 if p.exists():o[n]=json.loads(p.read_text())
p=r/'job/stderr'
if p.exists():o['stderr_tail']=p.read_text()[-4000:]
print(json.dumps(o))'''%remote
        v=run(code,'observe');(D/('observation-'+str(time.time_ns())+'.json')).write_text(json.dumps(v,indent=2));print(v['stdout'])


if __name__=='__main__':main()
