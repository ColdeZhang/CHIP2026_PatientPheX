"""One authorized base asset copy from pinned ModelScope revision; no model/GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def main(a):
    p=json.loads(Path(a.plan).read_text());root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    assert p['selected']=='google/long-t5-tglobal-base' and shutil.disk_usage(root).free>3*2**30
    start=time.monotonic();files={}
    for name,spec in p['assets'].items():
        path=root/name
        if path.exists():
            raw=path.read_bytes();assert len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
            files[name]={**spec,'reused':True};continue
        url=p['preferred_download_base']+name;temp=root/(name+'.part');h=hashlib.sha256();size=0;tick=time.monotonic()
        with temp.open('xb') as target,urllib.request.urlopen(url,timeout=60) as response:
            while True:
                block=response.read(4*1024*1024)
                if not block:break
                target.write(block);h.update(block);size+=len(block)
            target.flush();os.fsync(target.fileno())
        assert size==spec['bytes'] and h.hexdigest()==spec['sha256'],name
        os.rename(temp,path)
        reopened_hash=hashlib.sha256()
        with path.open('rb') as reopened:
            for block in iter(lambda:reopened.read(4*1024*1024),b''):reopened_hash.update(block)
        assert reopened_hash.hexdigest()==spec['sha256']
        files[name]={**spec,'reused':False,'seconds':time.monotonic()-tick,'url':url}
    fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    result={'status':'COMPLETE_SINGLE_BASE_ASSETS_FULL_SHA_FSYNC_REOPEN','files':files,'seconds':time.monotonic()-start,'GPU_calls':0}
    with (root/'complete.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);main(p.parse_args())
