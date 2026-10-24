"""Bounded public origin download; no model import or GPU access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    with p.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def main():
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'candidate-plan.json').read_text())
    ticket=json.loads((root/'asset-launch-ticket.json').read_text())
    assert time.time()<=ticket['launch_not_after_unix'],'Expired asset intent; no download'
    dest=Path(cfg['remote_model']);dest.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(dest).free>=4_000_000_000
    with (root/'asset-exclusive.json').open('x') as f:f.write(str(os.getpid()))
    start=time.time();deadline=start+600
    save(root/'asset-started.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'gpu_calls':0})
    items={}
    for name,expected in cfg['files'].items():
        p=dest/name
        if p.exists():
            assert p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256']
            items[name]={'reused_verified':True,**expected};continue
        url='https://huggingface.co/'+cfg['model']+'/resolve/'+cfg['revision']+'/'+name
        partial=p.with_name(name+'.partial')
        # A failed original partial is retained. Resume needs a separate explicit
        # intent after exact byte-range verification, never blind replacement.
        with partial.open('xb') as f:
            with urllib.request.urlopen(url,timeout=35) as response:
                assert response.status==200
                while True:
                    assert time.time()<deadline,'Original asset deadline elapsed'
                    block=response.read(1024*1024)
                    if not block:break
                    f.write(block)
            f.flush();os.fsync(f.fileno())
        assert partial.stat().st_size==expected['bytes'] and digest(partial)==expected['sha256']
        os.replace(partial,p);fd=os.open(dest,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        items[name]={'downloaded_and_full_SHA_verified':True,**expected}
        save(root/'asset-progress.json',{'files':items,'elapsed_seconds':time.time()-start})
    save(root/'asset-complete.json',{'status':'ALL_PUBLISHED_ASSETS_FULL_SHA_VERIFIED','files':items,
         'model_directory':str(dest),'seconds':time.time()-start,'GPU_or_model_calls':0})


if __name__=='__main__':main()
