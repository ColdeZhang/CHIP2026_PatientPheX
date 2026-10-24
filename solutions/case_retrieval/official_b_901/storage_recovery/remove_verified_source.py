"""Only after target receipt readback: remove one exact original state replica."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

PROJECT=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
ROOT=PROJECT/'case-retrieval/qwen35-preparation/full64-relation-cpu-preparation/formal2ep/epoch-002'
EXPECTED='38b9877a280c7f300878f93d9f362803f75b011a6cb5fd26eb6bf78c3fd66c88'
RECEIPT='731ba732e4923177700ec6c97295967dbf01c06aa2e2415de768bb855ffd60ad'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 assert os.environ['TARGET_RECEIPT_READBACK_SHA']==RECEIPT
 p=ROOT/'training_state.pt';s=p.stat();assert p.is_file() and not p.is_symlink()
 assert s.st_uid==os.getuid()==1045 and s.st_size==346667053 and s.st_mtime_ns==1792784171535766226
 assert digest(p)==EXPECTED
 for n,sha in [('checkpoint.json','f217bdc5bd191fe4319c38c99b46ef7b8d1f505d5f89d8c4d2e21a9425511e87'),('adapter/adapter_config.json','c4a478a7dd1aa65613279cdd4250da344ca16df8605de4249f445a1de2e8f2e2')]:assert digest(ROOT/n)==sha
 before=shutil.disk_usage(PROJECT).free;p.unlink();fd=os.open(ROOT,os.O_RDONLY);os.fsync(fd);os.close(fd)
 r=dict(status='ONE_VERIFIED_REPLICA_REMOVED_AFTER_DURABLE_TARGET_RECEIPT_READBACK',at_utc=datetime.now(timezone.utc).isoformat(),before_free_bytes=before,after_free_bytes=shutil.disk_usage(PROJECT).free,deleted=[dict(path=str(p),bytes=s.st_size,sha256=EXPECTED,mtime_ns=s.st_mtime_ns,uid=s.st_uid)],target_receipt_sha256=RECEIPT,target_state='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/storage-recovery-20261024/full64-epoch002-from-A800/training_state.pt',checkpoint_metadata_and_config_preserved=True,no_tensor_or_GPU=True)
 q=ROOT/'storage-relocation-20261024.json'
 with q.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(ROOT,os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(r))
if __name__=='__main__':main()
