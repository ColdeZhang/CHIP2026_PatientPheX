"""Small durable/source helpers for the one SCST engineering experiment."""
import datetime, hashlib, json, os, time
from pathlib import Path

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ref(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
    p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p);return p
def atomic(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);return ref(p)
def exclusive(path,value):
    p=Path(path)
    with p.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def admit(admission,execution):
    assert admission['valid_from_unix']-5<=time.time()<admission['valid_until_unix'],'admission expired'
    assert sha(execution)==admission['execution_sha256']
def ids_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
