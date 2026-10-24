from pathlib import Path
import argparse,gzip,hashlib,io,json,os,tarfile
ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
selection=json.loads(a.selection.read_text());assert selection['status']=='FROZEN_EXACT_TERMINAL_SELECTION_NO_QUALITY'
assert a.name==Path(a.name).name and a.name.endswith('.tar.gz')
a.output.mkdir(parents=True,exist_ok=False)
items=[]
for name,ref in selection['members'].items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts and name!='members.json'
 data=Path(ref['path']).read_bytes();assert len(data)==ref['bytes'] and hashlib.sha256(data).hexdigest()==ref['sha256']
 items.append(dict(archive_path=name,source_path=ref['path'],bytes=len(data),sha256=ref['sha256']))
manifest=dict(status='EXACT_IMMUTABLE_CONTENT_MEMBER_LIST',members=items,selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),no_reexecution=True)
raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode();(a.output/'members.json').write_bytes(raw)
archive=a.output/a.name
with archive.open('xb') as output:
 with gzip.GzipFile(fileobj=output,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as tar:
   for item in items:
    data=Path(item['source_path']).read_bytes();header=tarfile.TarInfo(item['archive_path']);header.size=len(data);header.mode=0o600;tar.addfile(header,io.BytesIO(data))
   header=tarfile.TarInfo('members.json');header.size=len(raw);header.mode=0o600;tar.addfile(header,io.BytesIO(raw))
 output.flush();os.fsync(output.fileno())
with tarfile.open(archive) as tar:
 assert len(tar.getnames())==len(items)+1 and len(set(tar.getnames()))==len(items)+1
 for item in items:
  b=tar.extractfile(item['archive_path']).read();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
 assert tar.extractfile('members.json').read()==raw
meta=dict(status='LOCAL_ARCHIVE_FSYNC_ALL_MEMBERS_VERIFIED_NO_REMOTE_WRITE',archive_path=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,member_count=len(items),members_manifest_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest())
with (a.output/'local-archive.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(a.output),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(meta))
