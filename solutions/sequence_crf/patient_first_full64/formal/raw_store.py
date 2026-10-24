"""Exclusive fsynced raw evidence publication before any codec interpretation."""
from pathlib import Path
import hashlib,json,os

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_default(value):
 if isinstance(value,(set,frozenset)):return sorted(value,key=repr)
 raise TypeError('Unsupported evidence type: '+type(value).__name__)
def fsync_chain(directory,root):
 directory=Path(directory).resolve();root=Path(root).resolve();directory.relative_to(root)
 while True:
  fd=os.open(str(directory),os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  if directory==root:break
  directory=directory.parent

def publish_bytes(path,raw,root):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(tmp,path);tmp.unlink();fsync_chain(path.parent,root)
 assert path.read_bytes()==raw
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def publish_json(path,value,root):
 return publish_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,default=json_default)+'\n').encode(),root)
def row_path(directory,row_id):return Path(directory)/(hashlib.sha256(row_id.encode()).hexdigest()+'.json')
def persist_raw(directory,record,origin,root):
 return publish_json(row_path(directory,record['row_id']),dict(record=record,origin=origin,raw_before_codec=True),root)
def persist_decision(directory,record,decision,raw_ref,root):
 return publish_json(row_path(directory,record['row_id']),dict(row_id=record['row_id'],decision=decision,raw=raw_ref),root)
