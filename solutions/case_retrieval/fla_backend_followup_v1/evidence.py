"""Evidence before judgement; stdlib-only, no GPU imports."""
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def ref(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def checked(record):
    path=Path(record['path']);actual=ref(path)
    assert all(actual[k]==record[k] for k in ('bytes','sha256'));return path


def jsha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fsync_chain(directory,root):
    directory,root=Path(directory).resolve(),Path(root).resolve()
    assert directory==root or root in directory.parents
    while True:
        descriptor=os.open(str(directory),os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if directory==root:break
        directory=directory.parent


def save(path,value,replace=False):
    path=Path(path);assert replace or not path.exists()
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path);fsync_chain(path.parent,path.parent)


def persist_then_observe(path,writer,observer,budget,metadata):
    """Writer closes durable raw before observer can inspect/raise on NaN."""
    path=Path(path);assert not path.exists()
    with path.open('xb') as handle:
        writer(handle);handle.flush();os.fsync(handle.fileno())
    fsync_chain(path.parent,path.parent)
    raw=ref(path);budget['used_bytes']+=raw['bytes']
    # Record raw provenance even when later observation raises.
    save(path.with_suffix(path.suffix+'.raw.json'),dict(raw=raw,metadata=metadata,raw_bytes_total=budget['used_bytes']))
    assert budget['used_bytes']<=budget['limit_bytes'],'Raw evidence budget exceeded; original preserved'
    observation=observer(path)
    save(path.with_suffix(path.suffix+'.observation.json'),dict(raw=raw,metadata=metadata,observation=observation))
    assert observation['finite'],'Nonfinite evidence retained before failure'
    return raw,observation


def positions(length):
    assert type(length) is int and length>=32
    result=[i*(length-1)//31 for i in range(32)]
    assert len(set(result))==32 and result[0]==0 and result[-1]==length-1
    return result


def softmax(values):
    top=max(values);scaled=[math.exp(x-top) for x in values];total=sum(scaled)
    return [x/total for x in scaled]
