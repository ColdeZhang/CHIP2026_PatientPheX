"""Recover the original possibly completed backup upload; never write remote bytes."""
import base64,hashlib,json,os,shlex,sys,zipfile,io
from pathlib import Path
from local_common import ref,save
sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
D=Path('/workspace/patientphex-runs/round2-20261023/research/copy-grounded-decode-v1/fallback-v2/minimal-backup-v1')
REL='copy-grounded-decode-v1/backups/F1-fallback-v2-artifacts.zip';REMOTE='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REL
MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def main():
 assert not (D/'complete.json').exists();raw=(D/'artifacts.zip').read_bytes();c=Client();c.initialize()
 def call(name,args,tag):
  r=c.request('tools/call',dict(name=name,arguments=args));save(D/(tag+'.response.json'),r);assert not r.get('isError'),'UNKNOWN tool error preserved; no remote write/retry'
  return r.get('structuredContent') or json.loads(next(x['text'] for x in r['content'] if x['type']=='text'))
 code='''import hashlib,json,os
from pathlib import Path
p=Path(PATH)
with p.open('rb') as f:b=f.read();os.fsync(f.fileno())
q=p.parent
while True:
 fd=os.open(q,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if q.parent==q:break
 q=q.parent
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),file_and_ancestors_fsynced=True)))
'''.replace('PATH',repr(REMOTE))
 v=call('run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=20),'recovery-fsync');assert v['exit_code']==0;remote=json.loads(v['stdout']);assert remote['bytes']==len(raw) and remote['sha256']==hashlib.sha256(raw).hexdigest()
 chunks=[]
 for offset in range(0,len(raw),32768):
  v=call('read_file',dict(mount_id=MOUNT,path=REL,offset=offset,limit=min(32768,len(raw)-offset)),'readback-'+str(offset));assert v['encoding']=='base64';chunk=base64.b64decode(v['content']);assert chunk==raw[offset:offset+32768];chunks.append(chunk)
 assert b''.join(chunks)==raw;z=zipfile.ZipFile(io.BytesIO(raw));members=json.loads(z.read('MANIFEST.json'));assert set(z.namelist())==set(members)|{'MANIFEST.json'}
 for name,item in members.items():b=z.read(name);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
 save(D/'complete.json',dict(schema='copy-grounded-minimal-backup-v1',source=ref(__file__),original_upload_source=ref(Path(__file__).with_name('backup_minimal.py')),original_upload_receipt_UNKNOWN=True,no_remote_rewrite=True,local=ref(D/'artifacts.zip'),remote=remote,mount=MOUNT,member_count=len(members),all_members_actual_readback_verified=True,gold_or_model_or_GPU=False,members=members));print(json.dumps(ref(D/'complete.json')))
if __name__=='__main__':main()
