"""Assemble byte-exact small parts into an owned new target, no models."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
spec=json.loads(sys.argv[1]);target=R/spec['target'];target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists();data=[]
for s in spec['parts']:
 p=R/s['path'];assert not p.is_symlink();b=p.read_bytes();assert len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256'];data.append(b)
b=b''.join(data);assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
with target.open('xb') as f:f.write(b)
assert hashlib.sha256(target.read_bytes()).hexdigest()==spec['sha256']
print(json.dumps({'path':str(target),'bytes':len(b),'sha256':spec['sha256']}))
