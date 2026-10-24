"""Persist only complete raw/predictions and compact evidence; no gold/model states."""
from pathlib import Path
import json,hashlib,tarfile,sys,shlex,datetime
E=Path(__file__).resolve().parent;B=E/'short-result-backup';B.mkdir(exist_ok=False)
F=E/'formal-preparation/formal';Q=E/'quality-F1';files={}
for p in [F/'responses.jsonl',F/'responses.manifest.json',F/'responses.checkpoint.json',F/'responses.loading.json',F/'responses-freeze.json',F/'inference-plan.json',F/'training-campaign-complete.json',F/'inference-complete.json',F/'inference-gpu-release.json',E/'quality-result.json',E/'RESULT-short-input-frozen.md',E/'terminal-manifest.json',E/'evaluation-preparation/evaluate_medgemma.py',E/'evaluation-preparation/pretraining-amendment.json',E/'formal-preparation/pretraining-amendment.md']:
 files[str(p.relative_to(E))]=p
for p in Q.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl']:files[str(p.relative_to(E))]=p
refs={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())}
manifest=B/'members.json';manifest.write_text(json.dumps(refs,indent=2)+'\n');archive=B/'med-short-terminal.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 tar.add(manifest,arcname='members.json',recursive=False)
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
h=hashlib.sha256(archive.read_bytes()).hexdigest();remote='research/medgemma-relation-control/formal2ep/result-backup/short-'+h[:12];dest=remote+'.tar.gz'
(B/'intent.json').write_text(json.dumps({'archive_sha256':h,'bytes':archive.stat().st_size,'destination':dest,'expanded_directory':remote,'no_gold':True},indent=2)+'\n')
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io;io.M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';io.upload(str(archive),dest)
code='''from pathlib import Path
import json,tarfile,hashlib,os,time
archive=Path(ARCHIVE);assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
root=Path(ROOT);assert not root.exists();root.mkdir(parents=True,mode=0o700)
with archive.open('rb') as f:os.fsync(f.fileno())
with tarfile.open(archive,'r:gz') as tar:
 refs=json.load(tar.extractfile('members.json'));assert set(tar.getnames())==set(refs)|{'members.json'} and len(tar.getmembers())==len(refs)+1
 for item in tar.getmembers():
  p=Path(item.name);assert item.isfile() and not p.is_absolute() and '..' not in p.parts
  data=tar.extractfile(item).read()
  if item.name!='members.json':assert len(data)==refs[item.name]['bytes'] and hashlib.sha256(data).hexdigest()==refs[item.name]['sha256']
  dst=root/item.name;dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
for n,r in refs.items():assert (root/n).stat().st_size==r['bytes'] and hashlib.sha256((root/n).read_bytes()).hexdigest()==r['sha256']
for p in sorted([p for p in root.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True)+[root]+list(root.parents):
 fd=os.open(str(p),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
print(json.dumps(dict(status='persistent_short_Med_raw_and_both_fullpreds_verified',archive_sha256=DIGEST,archive_bytes=archive.stat().st_size,directory=str(root),business_members=len(refs),business_bytes=sum(r['bytes'] for r in refs.values()),all_members_reopened_SHA=True,files_and_directories_fsynced=True,server_epoch=time.time())))
'''.replace('ARCHIVE',repr(dest)).replace('DIGEST',repr(h)).replace('ROOT',repr(remote))
r=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':30});assert r['exit_code']==0;(B/'actual-persistent.json').write_text(json.dumps(r,indent=2)+'\n');print(r['stdout'])
