"""Post-terminal durability receipt; no training changes, tensors loaded, or quality."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True);args=parser.parse_args()
    directory=args.directory.resolve();root=args.project_root.resolve();actual=directory/'actual'
    assert directory.is_relative_to(root) and directory.name=='formal2ep'
    receipt=directory/'durable-finalization.json';assert not receipt.exists()
    started=datetime.now(timezone.utc).isoformat();terminal=json.loads((directory/'terminal.json').read_text())
    assert terminal['exit_code']==0 and not terminal['timeout_triggered']
    result=json.loads((actual/'result.json').read_text())
    assert result['status']=='complete_fresh_R1_124x2_32steps' and result['completed_epochs']==2 and result['global_step']==32
    assert len(result['acknowledgements'])==2
    plan=json.loads((directory/'plan.json').read_text())
    assert sha(directory/'formal_train.py')==plan['source_sha256']==result['fingerprint']['source_sha256']
    assert sha(directory/'plan.json')==result['fingerprint']['plan_sha256']
    verified={};directories={actual,directory}
    for epoch in [1,2]:
        epoch_dir=actual/('epoch-%03d'%epoch)
        checkpoint=epoch_dir/'checkpoint.json';metadata=json.loads(checkpoint.read_text())
        ack_path=actual/('epoch-%03d-persistent-ack.json'%epoch);ack=json.loads(ack_path.read_text())
        assert metadata['status']=='complete_epoch' and metadata['completed_epochs']==epoch and metadata['global_step']==epoch*16
        assert ack['status']=='COMPLETE_EPOCH_DURABLE_FSYNC_REOPEN_ALL_SHA_ADAPTER_AND_ADAM_RNG_EXACT'
        assert ack==result['acknowledgements'][epoch-1]
        assert ack['checkpoint']['sha256']==sha(checkpoint) and ack['completed_epochs']==epoch and ack['global_step']==16*epoch
        for name,expected in metadata['files'].items():
            relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
            path=epoch_dir/relative;assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
            with path.open('rb') as stream:os.fsync(stream.fileno())
            verified[str(path.relative_to(directory))]=expected;directories.add(path.parent)
        for path in [checkpoint,ack_path]:
            with path.open('rb') as stream:os.fsync(stream.fileno())
            verified[str(path.relative_to(directory))]=dict(bytes=path.stat().st_size,sha256=sha(path))
        directories.add(epoch_dir)
    for path in [actual/'result.json',directory/'terminal.json']:
        with path.open('rb') as stream:os.fsync(stream.fileno())
        verified[str(path.relative_to(directory))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    cursor=actual
    while True:
        directories.add(cursor)
        if cursor==root:break
        cursor=cursor.parent
        assert cursor.is_relative_to(root)
    order=sorted(directories,key=lambda path:len(path.parts),reverse=True)
    for path in order:sync_directory(path)
    record=dict(status='COMPLETE_FINAL_CHECKPOINTS_ACK_FILES_AND_DIRECTORY_PARENT_CHAIN_FSYNCED',source_sha256=sha(__file__),
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),verified_files=verified,
        fsynced_directories=[str(path) for path in order],scope='Post-terminal filesystem durability supplement only; no training/model/evaluator call',
        training_source_not_changed=True,no_training_retry=True)
    with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for path in order:sync_directory(path)
    print(json.dumps(dict(status=record['status'],receipt=str(receipt),bytes=receipt.stat().st_size,sha256=sha(receipt))))

if __name__=='__main__':main()
