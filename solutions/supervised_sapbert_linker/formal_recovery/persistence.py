"""Small terminal source/raw303/prediction bundle, real remote fsync and readback."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import zipfile
import ops


def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))


def main():
    D=ops.D;out=D/'terminal-backup-v1';out.mkdir(exist_ok=False)
    assert (D/'quality-v1/result.json').exists()
    members={}
    for folder,prefix in [(Path(__file__).parent,'source'),(D/'collected','actual'),(D/'quality-v1','quality')]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in {'.py','.json','.jsonl'}:
                members[prefix+'/'+str(p.relative_to(folder))]=p.read_bytes()
    manifest={k:dict(bytes=len(v),sha256=sha(v)) for k,v in members.items()}
    members['content-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in members.items():z.writestr(name,data)
    blob=buf.getvalue();archive=out/'terminal.zip';archive.write_bytes(blob)
    package=json.loads((D/'package.json').read_text());rel='research/supervised-sapbert-linker/terminal-F1-v1'
    ops.call('write_file',dict(mount_id=package['mount'],path=rel+'/terminal.zip',content=base64.b64encode(blob).decode(),encoding='base64',create_parents=True),'terminal-write')
    code='''import pathlib,hashlib,zipfile,json,os
p=pathlib.Path(%r);b=p.read_bytes();assert len(b)==%d and hashlib.sha256(b).hexdigest()==%r
z=zipfile.ZipFile(p);m=json.loads(z.read('content-manifest.json'))
for n,r in m.items():
 b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
with p.open('rb') as f:os.fsync(f.fileno())
fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
print(json.dumps({'status':'ALL_TERMINAL_MEMBERS_VERIFIED_FSYNC_REOPEN','members':len(m)+1,'sha256':%r}))
'''%('/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel+'/terminal.zip',len(blob),sha(blob),sha(blob),sha(blob))
    receipt=ops.call('run_command',dict(mount_id=package['mount'],cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=30),'terminal-verify')
    assert receipt.get('exit_code',0)==0
    pieces=[]
    for offset in range(0,len(blob),32768):
        path='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel+'/terminal.zip'
        code="import pathlib,base64,json;p=pathlib.Path(%r);f=p.open('rb');f.seek(%d);b=f.read(32768);print(json.dumps({'bytes':len(b),'base64':base64.b64encode(b).decode()}))"%(path,offset)
        v=ops.call('run_command',dict(mount_id=package['mount'],cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=25),'terminal-readback-'+str(offset))
        assert v.get('exit_code',0)==0
        item=json.loads(v['stdout']);part=base64.b64decode(item['base64']);assert len(part)==item['bytes'];pieces.append(part)
    got=b''.join(pieces);assert got==blob
    p=out/'restored.zip'
    with open(p,'wb') as f:f.write(got);f.flush();os.fsync(f.fileno())
    z=zipfile.ZipFile(p)
    for name,r in manifest.items():assert sha(z.read(name))==r['sha256']
    complete=dict(status='COMPLETE_REAL_REMOTE_PERSISTENCE_AND_READBACK',archive=ref(p),members=len(members),remote=rel+'/terminal.zip',
        remote_verification=receipt,large_scope='Final1.309GBfullstate/public126MBbank stay at actual formal .3 persistent locators in complete.json; bundle is not self-contained.',
        readback_calls=len(pieces),new_models=0,new_quality=0)
    (out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete))


if __name__=='__main__':main()
