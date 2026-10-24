"""Recover fixed MedGemma19 assets to private A800 SHM using inherited Range/full-SHA/fsync mechanism."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, time, urllib.request


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--expected-files',required=True);p.add_argument('--resume-state',required=True);p.add_argument('--resume-state-sha',required=True);a=p.parse_args()
    started=time.monotonic();root=Path(a.destination)
    assert root.is_absolute() and '..' not in root.parts and os.getuid()==1045 and root.resolve()==root
    assert str(root)=='/dev/shm/patientphex-medgemma15-1045-20261024/models/medgemma-1.5-4b-it-9bca96ca3e30'
    assert time.time()<a.deadline
    with urllib.request.urlopen(a.url+'/manifest.json',timeout=20) as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==a.manifest_sha256
    manifest=json.loads(data)
    assert sha(a.expected_files)=='690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1'
    expected=json.loads(Path(a.expected_files).read_text())
    assert manifest['files']=={r['name']:{k:r[k] for k in ['bytes','sha256']} for r in expected['files']}
    assert all(manifest[k]==expected[k] for k in ['model_id','MS_revision','HF_revision'])
    assert manifest['expected_files_sha256']==sha(a.expected_files)
    assert sha(a.resume_state)==a.resume_state_sha
    resume=json.loads(Path(a.resume_state).read_text());initial=resume['initial_finalized'];modes={}
    assert resume['target_root']==str(root) and set(initial)<=set(manifest['files'])
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'asset-copy-ack.json').exists(),'Checkpoint already exported; inspect receipt instead of repeat'
    assert set(x.name for x in root.iterdir())==set(initial)|{n+'.part' for n in resume['initial_partials']}
    for n,r in resume['initial_partials'].items():
        part=root/(n+'.part');assert part.is_file() and not part.is_symlink();st=part.stat()
        assert {k:getattr(st,v) for k,v in [('bytes','st_size'),('inode','st_ino'),('mtime_ns','st_mtime_ns'),('ctime_ns','st_ctime_ns')]}==r
    for name,record in manifest['files'].items():
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        target=root/name;partial=target.with_name(target.name+'.part');target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.is_symlink() and not partial.is_symlink()
        if name in initial:
            assert target.is_file() and not partial.exists()
            st=target.stat();assert {k:getattr(st,v) for k,v in [('bytes','st_size'),('inode','st_ino'),('mtime_ns','st_mtime_ns'),('ctime_ns','st_ctime_ns')]}==initial[name]
            assert st.st_size==record['bytes'];modes[name]='initial-finalized-full-SHA-inherited-no-new-read';continue
        assert not target.exists(),'Unfrozen final file cannot be silently trusted'
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
        modes[name]='recovery-verified-full-SHA'
    assert set(modes)==set(manifest['files']) and len(modes)==19
    fd=os.open(str(root),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    result={**manifest,'status':'verified_SHM_copy_persistent_original_on_100','target_mount_id':'69b991ee-183f-4add-af1a-579af92479c9','persistent_source_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','target_is_persistent':False,
        'target_directory':str(root),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seconds':time.monotonic()-started,'total_bytes':sum(v['bytes'] for v in manifest['files'].values()),
        'transport_manifest_sha256':a.manifest_sha256,'fsync_all_files':True,
        'verification_mode_by_file':modes,'resume_state_sha256':a.resume_state_sha,
        'original_window':resume['original_window'],'completed_target_files_reread_this_window':False}
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    with (root/'asset-copy-ack.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(str(root),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
