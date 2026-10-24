"""Whole terminal small collection, complete large state remains durable.

Requires real driver/child termination. All small bytes roundtrip; large state
is reopened and fully hashed remotely and referenced, never implied downloaded.
"""
import argparse,hashlib,json,tarfile
from pathlib import Path
from common import atomic,ref,sha
from remote import Remote

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();b=json.loads(Path(a.binding).read_text());e=json.loads(Path(b['execution']['path']).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    api=Remote(b['mount'],b['python'],out/'operations');root=Path(b['remote_root'])
    code="""import pathlib,json,hashlib,os,subprocess,tarfile,time
p=pathlib.Path('.');done=json.loads((p/'driver-complete.json').read_text());assert not pathlib.Path('/proc/'+str(done['child_pid'])).exists();assert not pathlib.Path('/proc/'+str(done['driver_pid'])).exists()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);assert __TARGET_UUID__ not in apps
def ref(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
  os.fsync(f.fileno())
 return dict(bytes=q.stat().st_size,sha256=h.hexdigest())
assert not (p/'collection.tar.gz').exists() and not (p/'collection-inventory.json').exists()
members={};large={}
for q in sorted(p.rglob('*')):
 if q.is_file():
  name=str(q.relative_to(p))
  if name.startswith('terminal/probe-state/'):large[name]=ref(q)
  else:members[name]=ref(q)
inventory=dict(members=members,large_state_retained_not_downloaded=large,child_absent=True,driver_absent=True,target_no_apps=True,at_unix=time.time(),GPU_UUID=__TARGET_UUID__)
with (p/'collection-inventory.json').open('x') as f:json.dump(inventory,f,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
with tarfile.open(p/'collection.tar.gz','w:gz') as tf:
 for name in list(members)+['collection-inventory.json']:tf.add(p/name,arcname=name,recursive=False)
archive=ref(p/'collection.tar.gz');fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(archive=archive,members=len(members)+1,large_state_files=len(large),inventory=ref(p/'collection-inventory.json'),release=dict(at_unix=time.time(),gpu_uuid=__TARGET_UUID__,whole_chain_released=True,child_absent=True,driver_absent=True,target_no_apps=True),original_driver=done)))
""".replace('__TARGET_UUID__',repr(e['gpu_uuid']))
    assembled=api.code(code,root,'assemble',timeout=60);atomic(out/'assembly.json',assembled)
    got=api.get(root/'collection.tar.gz',out/'collection.tar.gz','archive');assert all(got[k]==assembled['archive'][k] for k in ('bytes','sha256'))
    payload=out/'payload';payload.mkdir()
    with tarfile.open(out/'collection.tar.gz','r:gz') as tf:
        members=tf.getmembers();assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        inv=json.load(tf.extractfile('collection-inventory.json'));assert set(m.name for m in members)==set(inv['members'])|{'collection-inventory.json'}
        for m in members:
            data=tf.extractfile(m).read()
            if m.name!='collection-inventory.json':assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==inv['members'][m.name]
            p=payload/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    atomic(out/'release.json',assembled['release'])
    atomic(out/'complete.json',dict(status='COMPLETE_SMALL_ROUNDTRIP_ALL_MEMBERS_SHA_LARGE_STATE_REMOTE_ONLY',archive=got,container_members=len(members),payload_members=len(inv['members']),
        original_driver=assembled['original_driver'],large_state=inv['large_state_retained_not_downloaded'],binding=ref(a.binding),source=ref(__file__),actual_release=ref(out/'release.json')))
    print(json.dumps(dict(status='COMPLETE',members=len(members),large_state=len(inv['large_state_retained_not_downloaded']))))
if __name__=='__main__':main()
