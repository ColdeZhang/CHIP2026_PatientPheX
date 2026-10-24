"""Small immutable backup using the project's existing verified upload helper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from transport import ref,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root);s=Path(__file__).resolve().parent;out=r/'backup-v1';out.mkdir(exist_ok=False)
    result=json.loads((r/'quality-F1-v1/evaluation/result.json').read_text());assert result['status']=='BYTEEXACT_NO_ADDITIONS_ZERO_GOLD_ZERO_V2'
    files={}
    for folder,prefix in [(r/'quality-F1-v1','quality'),(r/'transport-v1/actual','native'),(s,'source')]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.jsonl','.py','.md']:
                assert p.stat().st_size<1048576;files[prefix+'/'+str(p.relative_to(folder))]=p
    files['native-collection.json']=r/'transport-v1/collection.json';idx={n:ref(p) for n,p in files.items()};archive=out/'canonical-prefix-agreement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('index.json',json.dumps(idx,indent=2)+'\n')
    with archive.open('rb') as f:os.fsync(f.fileno())
    sys.path.insert(0,'/workspace/patientphex-tools');sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
    import backup_submissions as helper
    remote=helper.Remote(helper.GIT_REMOTE);save(out/'intent.json',dict(archive=ref(archive),helper=ref(helper.__file__),members=len(idx)))
    receipt=remote.put('canonical-prefix-agreement-20261024-'+ref(archive)['sha256'][:16]+'.zip',archive.read_bytes())
    code='PATH='+repr(receipt['remote_path'])+'\nSHA='+repr(receipt['sha256'])+'\n'+VERIFY
    verified=remote.python(code);save(out/'ack.json',dict(archive=ref(archive),remote=receipt,verified=verified,members=len(idx),helper=ref(helper.__file__)))
    print(json.dumps(ref(out/'ack.json')))

VERIFY=r'''
from pathlib import Path
import os,json,hashlib,zipfile
p=Path(PATH);assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
with zipfile.ZipFile(p) as z:
 idx=json.loads(z.read('index.json'));assert set(z.namelist())==set(idx)|{'index.json'}
 for n,r in idx.items():
  b=z.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
with p.open('rb') as f:os.fsync(f.fileno())
fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
print(json.dumps(dict(status='ALL_MEMBERS_SHA_FSYNC_REOPEN',path=str(p.resolve()),sha256=SHA,members=len(idx))))
'''

if __name__=='__main__':main()
