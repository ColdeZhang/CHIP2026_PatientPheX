"""Persist the approved final B output and compact audit supplement, no inference."""
import argparse,base64,hashlib,json,os,shlex,sys,tarfile
from pathlib import Path
PRED='50ef5daa212f3f46abca24545ded61ce04d5c601fc2c30d6d0b734c0bbb07d49'
REVIEW='cd7a6040516af20916746f99af9cdec78a389b7d73dfe1045ee687ef0937310a'
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE='research/official-b-901/final-frozen901-B-20261024'

def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run=a.run.resolve();out=a.out.resolve();out.mkdir(exist_ok=False)
 assembly=run/'actual-v1/assembly.json';v=json.loads(assembly.read_text());terminal=run/'terminal.json';t=json.loads(terminal.read_text());assert v['status']=='COMPLETE_FROZEN901_B_ASSEMBLY_NO_QUALITY' and t['returncode']==0 and not t['timed_out']
 assert v['prediction']['sha256']==PRED and ref(a.review)['sha256']==REVIEW
 paths=[Path(r['path']) for r in v['artifacts'].values()]+[assembly,terminal,Path(v['plan']['path']),Path(v['source']['path']),a.review.resolve(),Path(__file__).resolve(),Path(__file__).with_name('run_assembly_once.py'),Path(__file__).with_name('B-inference-status.json')]
 files={p.relative_to('/workspace').as_posix():p for p in paths};assert all(p.is_file() and not p.is_symlink() for p in files.values())
 members=[dict(path=n,bytes=ref(p)['bytes'],sha256=ref(p)['sha256']) for n,p in sorted(files.items())];archive=out/'frozen901-B-50ef5daa.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
 with archive.open('rb') as f:os.fsync(f.fileno())
 ar=ref(archive);assert ar['bytes']<2*1024*1024
 manifest=out/'manifest.json';save(manifest,dict(scope='Final frozen901 B prediction and compact complete-assembly audit supplement; not a standalone raw-to-output model bundle. Existing B neural archives and model/HPO/Task1/patient dependency chains remain referenced, unchanged and unduplicated.',prediction=v['prediction'],review=ref(a.review),members=members,archive=ar,mount_id=MOUNT,remote_directory=REMOTE,remote_archive=REMOTE+'/'+archive.name,remote_manifest=REMOTE+'/manifest.json',source=ref(__file__),no_model_or_quality_or_official_call=True))
 mr=ref(manifest);save(out/'intent.json',dict(archive=ar,manifest=mr,remote_directory=REMOTE,chunk_bytes=200000))
 sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
 c=Client();c.initialize()
 def call(name,label,**kwargs):
  r=c.request('tools/call',{'name':name,'arguments':dict(mount_id=MOUNT,**kwargs)});save(out/(label+'.json'),r)
  assert not r.get('isError'),'Persist tool error; do not retry blindly'
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 data=archive.read_bytes();parts=[data[i:i+200000] for i in range(0,len(data),200000)]
 for i,b in enumerate(parts):call('write_file','write-part-%03d'%i,path=REMOTE+'/chunk%03d'%i,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)
 call('write_file','write-manifest',path=REMOTE+'/manifest.json',content=manifest.read_text(),encoding='utf8',create_parents=True)
 code='''from pathlib import Path
import json,hashlib,os,tarfile
p=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')/REMOTE;m=p/'manifest.json';assert m.stat().st_size==MANIFEST_BYTES and hashlib.sha256(m.read_bytes()).hexdigest()==MANIFEST_SHA
metadata=json.loads(m.read_text());data=b''.join((p/('chunk%03d'%i)).read_bytes() for i in range(COUNT));assert len(data)==ARCHIVE_BYTES and hashlib.sha256(data).hexdigest()==ARCHIVE_SHA
a=p/ARCHIVE_NAME;assert not a.exists()
with a.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
with m.open('rb') as f:os.fsync(f.fileno())
with tarfile.open(a,'r:gz') as tar:
 rows=tar.getmembers();expected={x['path']:x for x in metadata['members']};assert len(rows)==len(set(x.name for x in rows))==len(expected) and {x.name for x in rows}==set(expected)
 for row in rows:
  assert row.isfile() and not Path(row.name).is_absolute() and '..' not in Path(row.name).parts
  b=tar.extractfile(row).read();r=expected[row.name];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
assert hashlib.sha256(a.read_bytes()).hexdigest()==ARCHIVE_SHA
for q in [p]+list(p.parents):
 fd=os.open(q,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='FINAL_B_SUPPLEMENT_ALL_MEMBER_SHA_FSYNC_REOPEN_PASS',archive_path=str(a),archive_sha256=ARCHIVE_SHA,manifest_path=str(m),manifest_sha256=MANIFEST_SHA,members=len(expected),prediction_sha256=PRED,no_official_request=True)))
'''
 for k,value in [('REMOTE',repr(REMOTE)),('MANIFEST_BYTES',str(mr['bytes'])),('MANIFEST_SHA',repr(mr['sha256'])),('COUNT',str(len(parts))),('ARCHIVE_BYTES',str(ar['bytes'])),('ARCHIVE_SHA',repr(ar['sha256'])),('ARCHIVE_NAME',repr(archive.name)),('PRED',repr(PRED))]:code=code.replace(k,value)
 r=call('run_command','server-verification',command='python3 -c '+shlex.quote(code),timeout_seconds=30);assert r['exit_code']==0;receipt=json.loads(r['stdout']);save(out/'persistent-receipt.json',receipt);print(json.dumps(dict(archive=ar,manifest=mr,receipt=ref(out/'persistent-receipt.json'),remote_manifest=REMOTE+'/manifest.json')))
if __name__=='__main__':main()
