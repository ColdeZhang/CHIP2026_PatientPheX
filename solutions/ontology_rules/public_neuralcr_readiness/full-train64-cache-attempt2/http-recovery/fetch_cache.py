"""Copy one allowlisted complete epoch to durable .100 storage and verify every byte."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, time, urllib.request


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    started=time.monotonic();root=Path(a.destination)
    assert not root.is_absolute() and '..' not in root.parts
    assert root==Path('public-neuralcr-readiness/persistent-recovery-v1/artifacts')
    assert time.time()<a.deadline
    with urllib.request.urlopen(a.url+'/manifest.json',timeout=20) as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==a.manifest_sha256
    manifest=json.loads(data);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'export-ack.json').exists(),'Cache artifacts already exported; inspect receipt instead of repeat'
    for name,record in manifest['files'].items():
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        target=root/name;partial=target.with_name(target.name+'.part');target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.is_symlink() and not partial.is_symlink()
        if target.exists():
            assert target.stat().st_size==record['bytes'] and sha(target)==record['sha256'];continue
        for attempt in range(3):
            assert time.time()<a.deadline,'Overall deadline before checkpoint fetch'
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                assert offset<=record['bytes']
                if offset<record['bytes']:
                    req=urllib.request.Request(a.url+'/files/'+name,headers={'Range':f'bytes={offset}-'} if offset else {})
                    with urllib.request.urlopen(req,timeout=min(30,max(1,a.deadline-time.time()))) as response:
                        assert response.status==(206 if offset else 200)
                        if offset:assert response.headers['Content-Range']==f"bytes {offset}-{record['bytes']-1}/{record['bytes']}"
                        assert int(response.headers['Content-Length'])==record['bytes']-offset
                        with partial.open('ab' if offset else 'wb') as f:
                            while True:
                                assert time.time()<a.deadline,'Overall deadline during checkpoint fetch'
                                block=response.read(1048576)
                                if not block:break
                                f.write(block)
                            f.flush();os.fsync(f.fileno())
                assert partial.stat().st_size==record['bytes'] and sha(partial)==record['sha256']
                partial.replace(target);break
            except Exception:
                if attempt==2:raise
                time.sleep(.5)
        assert target.stat().st_size==record['bytes'] and sha(target)==record['sha256']
    result={**manifest,'status':'persistent_cache_artifacts_verified','persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c',
        'persistent_directory':str(root),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seconds':time.monotonic()-started,'total_bytes':sum(v['bytes'] for v in manifest['files'].values()),
        'transport_manifest_sha256':a.manifest_sha256,'fsync_all_files':True}
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    with (root/'export-ack.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
