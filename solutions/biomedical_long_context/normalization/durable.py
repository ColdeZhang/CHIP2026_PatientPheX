"""Minimal durable metadata and exact state comparison for the isolated probe."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def ref(path):
    p=Path(path)
    return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))


def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with open(tmp,'w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fsync_dir(p.parent)


def tensor_compare(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a.detach().cpu(),b.detach().cpu())
        return 1
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        return sum(tensor_compare(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        return sum(tensor_compare(x,y) for x,y in zip(a,b))
    assert a==b
    return 0
