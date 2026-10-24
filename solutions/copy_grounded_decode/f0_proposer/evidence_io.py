"""Original GLiNER JSON bytes with durable, immutable evidence except progress."""
import hashlib,json,os
from pathlib import Path
from raw_store import publish_bytes,fsync_chain

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def binary(path,data):
 return publish_bytes(path,data,os.environ['PATIENTPHEX_DURABILITY_ROOT'])
def write(path,value):
 path=Path(path);raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 if path.name!='progress.json':return binary(path,raw)
 tmp=path.with_suffix(path.suffix+'.pending')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fsync_chain(path.parent,os.environ['PATIENTPHEX_DURABILITY_ROOT'])
