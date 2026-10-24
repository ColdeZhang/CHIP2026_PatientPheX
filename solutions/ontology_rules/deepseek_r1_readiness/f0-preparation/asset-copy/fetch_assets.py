"""Copy fixed R1 nine assets to private A800 SHM using inherited Range/full-SHA/fsync mechanism."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, time, urllib.request


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--expected-files',required=True);a=p.parse_args()
    started=time.monotonic();root=Path(a.destination)
    assert root.is_absolute() and '..' not in root.parts and os.getuid()==1045 and root.resolve()==root
    assert str(root)=='/dev/shm/patientphex-r1-f0-1045-20261024/models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6'
    assert time.time()<a.deadline
    with urllib.request.urlopen(a.url+'/manifest.json',timeout=20) as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==a.manifest_sha256
    manifest=json.loads(data)
    assert sha(a.expected_files)=='ee707db3a5b314c94c213386c2c8b8640fd9406759e68f9e729640f70ccbfaec'
    expected=json.loads(Path(a.expected_files).read_text())
    assert manifest['files']=={r['name']:{k:r[k] for k in ['bytes','sha256']} for r in expected['files']}
    assert manifest['model_id']==expected['model_id']=='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' and manifest['MS_revision']==expected['revision']=='5ee5fa4832e658ab84298806de06ed2a167834bd' and len(manifest['files'])==9
    assert manifest['expected_files_sha256']==sha(a.expected_files)
    root.mkdir(parents=True,exist_ok=True)
    for owned in [root.parent.parent,root.parent,root]:
        assert owned.stat().st_uid==1045 and not owned.is_symlink();os.chmod(owned,0o700)
    assert not (root/'asset-copy-ack.json').exists(),'Checkpoint already exported; inspect receipt instead of repeat'
    copied=[];last_progress=0.0
    def progress(name,offset):
        value={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_file':name,'current_bytes':offset,'completed_files':copied,'deadline':a.deadline}
        temporary=root/'copy-progress.tmp';temporary.write_text(json.dumps(value)+'\n');temporary.replace(root/'copy-progress.json')
    for name,record in manifest['files'].items():
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        target=root/name;partial=target.with_name(target.name+'.part');target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.is_symlink() and not partial.is_symlink()
        if target.exists():
            assert target.stat().st_size==record['bytes'] and sha(target)==record['sha256'];copied.append(name);progress(name,record['bytes']);continue
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
                                if time.monotonic()-last_progress>=5:progress(name,f.tell());last_progress=time.monotonic()
                            f.flush();os.fsync(f.fileno())
                assert partial.stat().st_size==record['bytes'] and sha(partial)==record['sha256']
                partial.replace(target);break
            except Exception:
                if attempt==2:raise
                time.sleep(.5)
        assert target.stat().st_size==record['bytes'] and sha(target)==record['sha256']
        copied.append(name);progress(name,record['bytes'])
    fd=os.open(str(root),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    result={**manifest,'status':'verified_SHM_copy_persistent_original_on_100','target_mount_id':'69b991ee-183f-4add-af1a-579af92479c9','persistent_source_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','target_is_persistent':False,
        'target_directory':str(root),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seconds':time.monotonic()-started,'total_bytes':sum(v['bytes'] for v in manifest['files'].values()),
        'transport_manifest_sha256':a.manifest_sha256,'fsync_all_files':True}
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    with (root/'asset-copy-ack.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(str(root),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
