import hashlib,json,gzip,os
from pathlib import Path

def ref(path):
 p=Path(path);b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def sha(path):return ref(path)['sha256']
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def packed(path,obj):
 b=json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode();p=Path(path);p.write_bytes(gzip.compress(b,compresslevel=1,mtime=0));return {'compressed':ref(p),'decoded_bytes':len(b),'decoded_sha256':hashlib.sha256(b).hexdigest()}
def unpack(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
