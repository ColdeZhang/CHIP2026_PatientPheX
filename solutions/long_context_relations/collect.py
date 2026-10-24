"""Collect bounded necessary complete stage files, not multi-GB model state."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import zipfile
from rpc import call


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['formal']);a=ap.parse_args()
    D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1')/(a.stage+'-v1')
    pkg=json.loads((D/'package.json').read_text());remote=pkg['remote'];rel=pkg['relative'];mount=pkg['mount']
    names=['job/started.json','job/ended.json','job/fresh-gate.json']
    names+=['output/complete.json','output/final-state-ACK.json','output/loading.json','output/inputs.json','output/held-scores.jsonl','formal-plan.json']
    code='''import pathlib,json,zipfile,hashlib,os
r=pathlib.Path(%r);names=%r;end=json.loads((r/'job/ended.json').read_text());assert end['status']=='COMPLETE' and end['returncode']==0 and not end['timed_out']
members={}
for n in names:
 b=(r/n).read_bytes();members[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
p=r/'necessary-complete.zip'
if not p.exists():
 with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names:z.write(r/n,n)
 with p.open('rb') as f:os.fsync(f.fileno())
 fd=os.open(r,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with zipfile.ZipFile(p) as z:
 assert set(z.namelist())==set(names)
 for n,v in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
b=p.read_bytes();print(json.dumps({'status':'COMPLETE_NECESSARY_ARCHIVE','path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'members':members,'large_model_not_included':True}))
'''%(remote,names)
    if (D/'collection-inventory.json').exists():inv=json.loads((D/'collection-inventory.json').read_text())
    else:
        v=call(D,'run_command',{'mount_id':mount,'cwd':'.','command':'python3 -c '+shlex.quote(code),'timeout_seconds':35},'freeze-necessary-archive')
        assert v.get('exit_code',0)==0,v;inv=json.loads(v['stdout']);(D/'collection-inventory.json').write_text(json.dumps(inv,indent=2))
    parts=D/'download-parts';parts.mkdir(exist_ok=True)
    for offset in range(0,inv['bytes'],32768):
        p=parts/str(offset);length=min(32768,inv['bytes']-offset)
        if p.exists():assert p.stat().st_size==length;continue
        v=call(D,'read_file',{'mount_id':mount,'path':rel+'/necessary-complete.zip','offset':offset,'limit':length},'read-'+str(offset))
        b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode();assert len(b)==length
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    blob=b''.join((parts/str(offset)).read_bytes() for offset in range(0,inv['bytes'],32768));assert hashlib.sha256(blob).hexdigest()==inv['sha256']
    archive=D/'restored-necessary.zip'
    with archive.open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
    target=D/'collected';target.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert set(z.namelist())==set(inv['members'])
        for n,r in inv['members'].items():
            b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];p=target/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==b
            else:
                with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    for p in sorted([target]+[x for x in target.rglob('*') if x.is_dir()],key=lambda x:len(x.parts),reverse=True):
        fd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    for n,r in inv['members'].items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==r['sha256']
    result={'status':'ALL_NECESSARY_MEMBERS_ACTUAL_READBACK_SHA_FSYNC_REOPEN','archive':{'path':str(archive),'bytes':len(blob),'sha256':inv['sha256']},'members':len(inv['members']),'large_state_not_locally_downloaded':True}
    (D/'collection-complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
