"""H13 official CDN Range transport; same pinned file, no model load."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def save(p,x):
    with p.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def main():
    root=Path(__file__).resolve().parent;private=root/'cdn.private.json';cfg=json.loads(private.read_text())
    assert time.time()<=cfg['launch_not_after_unix']
    with (root/'exclusive.json').open('x') as f:f.write(str(os.getpid()))
    start=time.time();deadline=start+600;dest=Path(cfg['destination']);dest.mkdir(parents=True,exist_ok=True)
    save(root/'started.json',{'pid':os.getpid(),'started_unix':start,'deadline_unix':deadline,'no_GPU':True})
    n=cfg['bytes'];step=16*1024*1024;parts=root/'parts';parts.mkdir()
    ranges=[(i,a,min(a+step,n)-1) for i,a in enumerate(range(0,n,step))]
    def fetch(item):
        i,a,b=item;assert time.time()<deadline
        req=urllib.request.Request(cfg['cdn_url'],headers={'Range':f'bytes={a}-{b}'})
        p=parts/('%03d'%i)
        with urllib.request.urlopen(req,timeout=45) as r:
            assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{n}'
            with p.open('xb') as f:
                while True:
                    assert time.time()<deadline
                    buf=r.read(1024*1024)
                    if not buf:break
                    f.write(buf)
                f.flush();os.fsync(f.fileno())
        assert p.stat().st_size==b-a+1;return p.stat().st_size
    total=0
    with ThreadPoolExecutor(4) as executor:
        for future in as_completed([executor.submit(fetch,x) for x in ranges]):
            total+=future.result();save(root/'progress.json',{'bytes':total,'expected':n,'seconds':time.time()-start})
    h=hashlib.sha256();p=dest/'model.safetensors.cdn-partial'
    with p.open('xb') as out:
        for i,_,_ in ranges:
            with (parts/('%03d'%i)).open('rb') as inp:
                for buf in iter(lambda:inp.read(1024*1024),b''):out.write(buf);h.update(buf)
        out.flush();os.fsync(out.fileno())
    assert total==n and h.hexdigest()==cfg['sha256']
    final=dest/'model.safetensors';assert not final.exists();os.replace(p,final)
    fd=os.open(dest,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    private.unlink()
    save(root/'complete.json',{'status':'OFFICIAL_CDN_FULL_SHA_VERIFIED','bytes':n,'sha256':h.hexdigest(),'path':str(final),
        'seconds':time.time()-start,'original_resolve_url':cfg['original_resolve_url'],'private_url_removed':True,'no_GPU_or_model':True})


if __name__=='__main__':
    try:main()
    except Exception as e:
        # urllib exceptions can carry signed URLs. Persist only the error type.
        save(Path(__file__).resolve().parent/'failure.json',{'error_type':type(e).__name__,'unix':time.time()})
        raise SystemExit(1)
