import hashlib
import json
import os
from pathlib import Path

def ref(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def digest(value): return hashlib.sha256(json.dumps(value).encode()).hexdigest()

def write(path,value,*,jsonl=False):
    path=Path(path)
    raw=(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in value)
         if jsonl else json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    with path.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return ref(path)

def load_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def validate_refs(refs):
    for name,item in refs.items():assert ref(item['path'])==item,name
