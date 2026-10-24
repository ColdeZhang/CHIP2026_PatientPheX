"""Persist/reopen the three original bytes on .100 before any source removal."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

FILES={'training_state.pt':('38b9877a280c7f300878f93d9f362803f75b011a6cb5fd26eb6bf78c3fd66c88',346667053),
       'checkpoint.json':('f217bdc5bd191fe4319c38c99b46ef7b8d1f505d5f89d8c4d2e21a9425511e87',10774),
       'adapter/adapter_config.json':('c4a478a7dd1aa65613279cdd4250da344ca16df8605de4249f445a1de2e8f2e2',1287)}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    project=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
    out=project/'research/storage-recovery-20261024/full64-epoch002-from-A800'
    assert not out.exists();assert shutil.disk_usage(project).free>1024**3
    base='http://10.5.4.7:51963/'+os.environ['RECOVERY_PREFIX']+'/'
    ready=False
    for attempt in range(20):
        try:
            with urllib.request.urlopen(base+'ready',timeout=1) as response:ready=response.read()==b'READY'
        except OSError:time.sleep(.5)
        if ready:break
    assert ready, 'Read-only source server did not become available; no download begun'
    out.mkdir(parents=True)
    started=time.monotonic();received={}
    for name,(expected,size) in FILES.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part')
        with urllib.request.urlopen(base+name,timeout=25) as response,tmp.open('xb') as f:
            while True:
                data=response.read(4*1024*1024)
                if not data:break
                f.write(data)
            f.flush();os.fsync(f.fileno())
        assert tmp.stat().st_size==size and digest(tmp)==expected
        tmp.rename(p);syncdir(p.parent)
        # Independent reopen after rename and file/directory sync.
        assert p.stat().st_size==size and digest(p)==expected
        received[name]=dict(path=str(p),bytes=size,sha256=expected,reopened_sha256=expected,fsynced=True)
    cursor=out
    while True:
        syncdir(cursor)
        if cursor==project:break
        cursor=cursor.parent
    record=dict(status='ALL3_ORIGINAL_CHECKPOINT_FILES_PERSISTED_REOPENED_EXACT_SHA',
                at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,files=received,
                parent_chain_fsynced_through=str(project),source_not_deleted_by_receiver=True,
                adapter_weight_existing_locator=str(project/'case-retrieval/qwen35-preparation/full64-dev-transfer/adapter-epoch002/adapter_model.safetensors'),
                adapter_weight_sha256='1a703b244b17c1378b383e4d6feefe2486b1a888e78cbd6f9f25c3fa6e46f45b',
                scope='Storage relocation only; no tensor load, training, generation, evaluation or checkpoint metadata rewriting')
    receipt=out/'migration-receipt.json'
    with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    syncdir(out);syncdir(out.parent)
    print(json.dumps(dict(status=record['status'],receipt=str(receipt),bytes=receipt.stat().st_size,sha256=digest(receipt),seconds=record['seconds'])),flush=True)


if __name__=='__main__':main()
