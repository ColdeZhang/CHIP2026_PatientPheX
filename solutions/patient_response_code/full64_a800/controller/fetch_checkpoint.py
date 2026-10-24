"""Copy one complete Qwen35 response-code epoch to its durable .100 namespace.
Only CPU HTTP/disk I/O; owner freezes the separate absolute min600/remaining-campaign export deadline.
"""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, re, time, urllib.request

CHECKPOINT_REQUIRED = {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}
CHECKPOINT_ALLOWED = CHECKPOINT_REQUIRED | {'adapter/README.md'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()



def fsync_directory(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def make_durable_directory(path):
    # Only called after destination/relative-name allowlist checks below.
    missing=[];current=Path(path)
    while not current.exists():missing.append(current);current=current.parent
    assert current.is_dir() and not current.is_symlink()
    for directory in reversed(missing):
        directory.mkdir();fsync_directory(directory);fsync_directory(directory.parent)
    assert Path(path).is_dir() and not Path(path).is_symlink()


def durable_verify(path,record):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size==record['bytes']
    with path.open('rb') as f:os.fsync(f.fileno())
    # sha() opens the final filename again after fsync, not an in-memory buffer.
    assert sha(path)==record['sha256']
    fsync_directory(path.parent)


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());SOURCE_RUN_ROOT=Path(plan['output_dir']);PERSISTENT_ROOT=Path(plan['persistent_checkpoint_root']);TRAINER_SHA=plan['trainer_sha256'];TRAIN_SHA=plan['train_sha256'];RUN_FILES=plan['export_run_files']
    started=time.monotonic();root=Path(a.destination)
    assert root.is_absolute() and '..' not in root.parts and root.parent==PERSISTENT_ROOT
    assert root.name in {'epoch-001','epoch-002'} and not root.is_symlink()
    assert time.time()<a.deadline
    with urllib.request.urlopen(a.url+'/manifest.json',timeout=min(20,max(1,a.deadline-time.time()))) as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==a.manifest_sha256
    manifest=json.loads(data);epoch=manifest['completed_epochs']
    assert epoch in (1,2) and manifest['global_step']==34*epoch and root.name==f'epoch-{epoch:03d}'
    assert manifest['source_checkpoint_directory']==str(SOURCE_RUN_ROOT/root.name)
    assert manifest['trainer_sha256']==TRAINER_SHA
    ack_name=f'export-ack-epoch-{epoch:03d}.json';assert manifest['ack_basename']==ack_name
    expected_run={'_run/'+n for n in RUN_FILES+(['training_complete.json'] if epoch==2 else [])}
    required=CHECKPOINT_REQUIRED|{'checkpoint.json'}|expected_run
    allowed=CHECKPOINT_ALLOWED|{'checkpoint.json'}|expected_run
    assert required<=set(manifest['files'])<=allowed
    assert manifest['files']['checkpoint.json']['sha256']==manifest['checkpoint_sha256']
    assert manifest['files']['_run/run_manifest.json']['sha256']==manifest['run_manifest_sha256']
    assert manifest['files']['_run/'+plan['export_trainer_basename']]['sha256']==TRAINER_SHA
    assert os.getuid()==1006
    make_durable_directory(root)
    os.chmod(root,0o700)
    assert not (root/ack_name).exists(),'Checkpoint already exported; inspect receipt instead of repeat'
    for name,record in manifest['files'].items():
        assert time.time()<a.deadline,'Overall deadline before checkpoint member'
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==name
        assert type(record['bytes']) is int and record['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',record['sha256'])
        target=root/name;partial=target.with_name(target.name+'.part');make_durable_directory(target.parent)
        assert root.resolve() in target.resolve().parents
        assert not target.is_symlink() and not partial.is_symlink()
        if target.exists():
            durable_verify(target,record);continue
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
                partial.replace(target);fsync_directory(target.parent);break
            except Exception:
                if attempt==2:raise
                time.sleep(.5)
        durable_verify(target,record)
    assert time.time()<a.deadline,'Overall deadline before durable acknowledgement'
    result={**manifest,'status':'persistent_checkpoint_verified','persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c',
        'persistent_directory':str(root),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seconds':time.monotonic()-started,'total_bytes':sum(v['bytes'] for v in manifest['files'].values()),
        'transport_manifest_sha256':a.manifest_sha256,'fsync_all_files':True,
        'reopened_final_filename_SHA_all_files':True,'destination_directories_fsynced':True}
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    ack_path=root/ack_name
    with ack_path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_directory(root)
    assert sha(ack_path)==hashlib.sha256(raw).hexdigest()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
