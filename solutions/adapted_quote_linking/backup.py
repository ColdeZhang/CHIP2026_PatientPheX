"""One exclusive minimal CPU artifact backup, exact remote SHA/fsync and actual readback."""
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
from pathlib import Path
from common import ref,save
sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
MOUNT='1593f159-19bc-41ca-8b4c-2e7455f5db99'
PROJECT='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--remote-relative',required=True);a=ap.parse_args();root=Path(a.root).resolve();out=root/'minimal-backup-v1';out.mkdir(exist_ok=False)
 names=['final-review-entry.json']
 for part in ['F1-prepared-v1','F1-normalized-v1','F1-quality-v1']:
  names += [str(p.relative_to(root)) for p in sorted((root/part).rglob('*')) if p.is_file()]
 names += ['F1-query-v1/collection.json'] + [str(p.relative_to(root)) for p in sorted((root/'F1-query-v1/terminal').rglob('*')) if p.is_file()]
 assert all('.gold.' not in n and not n.startswith('/') and '..' not in Path(n).parts for n in names)
 manifest={n:ref(root/n) for n in names};payload=io.BytesIO()
 with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names:z.writestr(n,(root/n).read_bytes())
  z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 raw=payload.getvalue();assert len(raw)<1024*1024;dest=out/'artifacts.zip'
 with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 remote=Path(a.remote_relative);assert not remote.is_absolute() and '..' not in remote.parts and remote.parts[:2]==('research','adapted-quote-linking-v1')
 path=PROJECT+'/'+remote.as_posix();client=Client();client.initialize()
 def call(name,args,tag):
  save(out/(tag+'.intent.json'),dict(tool=name,arguments={k:('[payload omitted]' if k=='content' else v) for k,v in args.items()}))
  r=client.request('tools/call',dict(name=name,arguments=args));assert not r.get('isError'),r
  v=r.get('structuredContent') or json.loads(next(x['text'] for x in r['content'] if x['type']=='text'));save(out/(tag+'.receipt.json'),v);return v
 code='from pathlib import Path\np=Path('+repr(path)+')\nassert not p.exists()\nprint("ABSENT")'
 v=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=20),'inventory');assert v['exit_code']==0
 call('write_file',dict(mount_id=MOUNT,path=remote.as_posix(),content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True),'upload')
 code='''import hashlib,json,os
from pathlib import Path
p=Path(PATH)
with p.open('rb') as f:b=f.read();os.fsync(f.fileno())
q=p.parent
while True:
 fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if q.parent==q:break
 q=q.parent
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),fsync_file_and_ancestors=True)))
'''.replace('PATH',repr(path))
 v=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=20),'remote-verify');assert v['exit_code']==0;remote_ref=json.loads(v['stdout']);assert remote_ref['sha256']==hashlib.sha256(raw).hexdigest() and remote_ref['bytes']==len(raw)
 v=call('read_file',dict(mount_id=MOUNT,path=remote.as_posix(),offset=0,limit=1024*1024),'readback');assert v['encoding']=='base64';back=base64.b64decode(v['content']);assert back==raw
 z=zipfile.ZipFile(io.BytesIO(back));assert set(z.namelist())==set(names)|{'MANIFEST.json'}
 for n,item in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==item['sha256'] and len(z.read(n))==item['bytes']
 save(out/'complete.json',dict(schema='adapted-quote-minimal-backup-v1',source=ref(__file__),local=ref(dest),remote=remote_ref,mount=MOUNT,member_count=len(names),all_members_actual_readback_verified=True,gold_or_model_or_GPU=False,members=manifest))
 print(json.dumps(ref(out/'complete.json')))
if __name__=='__main__':main()
