"""Collect an immutable terminal campaign archive; never restarts a GPU job."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import sys
import time
import zipfile
from common import ref,write

P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    campaign=Path(a.campaign);plan=json.loads((campaign/'candidate-plan-actual.json').read_text());root=str(Path(plan['run_root']).parent)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    client=Client();client.initialize()
    def call(tool,args,tag):
        write(out/(tag+'.intent.json'),dict(tool=tool,args=args))
        try:r=client.request('tools/call',dict(name=tool,arguments=args))
        except Exception as exc:
            write(out/(tag+'.UNKNOWN.json'),dict(error_type=type(exc).__name__,no_model_restart=True));raise
        if r.get('isError'):write(out/(tag+'.ERROR.json'),r);raise RuntimeError('Read/collection error preserved; no model retry')
        value=r.get('structuredContent')
        if value is None:value=json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
        write(out/(tag+'.response.json'),{k:v for k,v in value.items() if tool!='read_file' or k!='content'})
        return value
    code='ROOT='+repr(root)+'\n'+REMOTE
    result=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=45),'freeze-terminal')
    assert result['exit_code']==0,result.get('stderr');ready=json.loads(result['stdout']);write(out/'ready.json',ready)
    assert ready['terminal'] and ready['driver_status'] in ['COMPLETE_THREE_PHASES','STOPPED_ORIGINAL_NO_RETRY']
    wanted=ready['archive'];data=bytearray()
    for pos in range(0,wanted['bytes'],1048576):
        r=call('read_file',dict(mount_id=MOUNT,path=wanted['path'][len(P)+1:],offset=pos,limit=min(1048576,wanted['bytes']-pos)),f'read-{pos}')
        data.extend(base64.b64decode(r['content']) if r.get('encoding')=='base64' else r['content'].encode())
    assert len(data)==wanted['bytes'] and hashlib.sha256(data).hexdigest()==wanted['sha256']
    archive=out/'terminal.zip'
    with archive.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
    assert ref(archive)['sha256']==wanted['sha256'];dest=out/'actual';dest.mkdir()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names));index=json.loads(z.read('member-index.json'))
        assert set(names)==set(index['members'])|{'member-index.json'}
        for name,item in index['members'].items():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            b=z.read(name);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
            assert ref(target)['sha256']==item['sha256']
        write(dest/'member-index.json',index)
    write(out/'collection.json',dict(status='TERMINAL_ARCHIVE_ALL_MEMBERS_SHA_FSYNC_REOPEN',ready=ref(out/'ready.json'),
        archive=ref(archive),members=len(index['members']),index=ref(dest/'member-index.json'),driver_status=ready['driver_status'],
        weights_not_downloaded=True,large_states_remain_at_durable_ACK_locators=True,no_new_model_or_quality=True))
    print(json.dumps(ref(out/'collection.json')))

REMOTE=r'''
from pathlib import Path
import os,json,hashlib,zipfile
root=Path(ROOT);run=root/'run';terminal=run/'driver-result.json'
assert terminal.is_file(),'Original driver is not terminal; do not freeze a partial campaign'
driver=json.loads(terminal.read_text())
def identity(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ref(p):return dict(path=str(p),**identity(p.read_bytes()))
for stage in driver['stages']:
 p=Path('/proc')/str(stage['pid'])/'stat';state=p.read_text().split(') ',1)[1].split()[0] if p.exists() else None
 assert state in [None,'Z','X'],'Original child still running'
out=root/'terminal';out.mkdir(exist_ok=True);archive=out/'terminal.zip'
if not archive.exists():
 files={}
 for folder,prefix in [(run,'run'),(root/'source','source'),(root/'data','data')]:
  for p in sorted(folder.rglob('*')):
   assert not p.is_symlink()
   if p.is_file() and p.suffix in ['.json','.jsonl','.log','.py','.gz'] and p.stat().st_size<=2*1048576:
    files[prefix+'/'+str(p.relative_to(folder))]=p
 files['candidate-plan.json']=root/'candidate-plan.json'
 index=dict(remote_root=str(root),driver_status=driver['status'],members={n:identity(p.read_bytes()) for n,p in files.items()})
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.write(p,n)
  z.writestr('member-index.json',json.dumps(index,indent=2)+'\n')
 with archive.open('rb') as h:os.fsync(h.fileno())
 fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with zipfile.ZipFile(archive) as z:
 idx=json.loads(z.read('member-index.json'));assert set(z.namelist())==set(idx['members'])|{'member-index.json'}
 for n,item in idx['members'].items():assert identity(z.read(n))==item
print(json.dumps(dict(terminal=True,driver_status=driver['status'],archive=ref(archive),members=len(idx['members']))))
'''

if __name__=='__main__':main()
