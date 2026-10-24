"""Remove only three pinned state replicas after all target receipts readback."""
from datetime import datetime,timezone
import hashlib,json,os,shutil
from pathlib import Path
PROJECT=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
PINNED_PLAN='ab4934b80b46c270415fffb2b891c491d304afa99336173340c8ea51e2b901f5'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def main():
 plan_bytes=os.environ['RECOVERY_PLAN'].encode();assert hashlib.sha256(plan_bytes).hexdigest()==PINNED_PLAN
 plan=json.loads(plan_bytes);bindings=json.loads(os.environ['TARGET_RECEIPTS_READ_BACK']);specs=plan['specifications'];assert len(bindings)==len(specs)==3
 for spec,binding in zip(specs,bindings):
  text=binding['receipt_text'];assert hashlib.sha256(text.encode()).hexdigest()==binding['sha256'];receipt=json.loads(text)
  assert receipt['status']=='THREE_ORIGINAL_FILES_DURABLE_REOPENED'
  assert receipt['source_directory']==spec['source_directory'] and receipt['target_directory']==spec['target_directory']
  assert set(receipt['files'])==set(spec['files'])
  for n,r in spec['files'].items():
   target=receipt['files'][n];assert target['fsynced'] and target['reopened_sha256']==target['sha256']==r['sha256'] and target['bytes']==r['bytes']
   p=Path(r['path']);s=p.stat();assert p==Path(spec['source_directory'])/n and PROJECT/'case-retrieval/qwen35-preparation' in p.parents
   assert p.is_file() and not p.is_symlink() and s.st_uid==r['uid']==os.getuid()==1045
   assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'] and digest(p)==r['sha256']
 before=shutil.disk_usage(PROJECT).free;deleted=[]
 for spec,binding in zip(specs,bindings):
  p=Path(spec['files']['training_state.pt']['path']);p.unlink();syncdir(p.parent)
  record=dict(at_utc=datetime.now(timezone.utc).isoformat(),deleted_source=spec['files']['training_state.pt'],target_state=spec['target_directory']+'/training_state.pt',target_receipt_sha256=binding['sha256'],metadata_and_config_preserved=True)
  q=p.parent/'storage-relocation-20261024.json'
  with q.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  syncdir(p.parent);deleted.append(record)
 out=PROJECT/'research/storage-recovery-20261024';out.mkdir(parents=True,exist_ok=True)
 r=dict(status='THREE_VERIFIED_STATE_REPLICAS_REMOVED_ONLY_AFTER_TARGET_READBACK',at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PINNED_PLAN,before_free_bytes=before,after_free_bytes=shutil.disk_usage(PROJECT).free,deleted=deleted,no_model_or_GPU=True,original_checkpoint_metadata_unchanged=True)
 q=out/'A800-additional-removal.json'
 with q.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 cursor=out
 while True:
  syncdir(cursor)
  if cursor==PROJECT:break
  cursor=cursor.parent
 print(json.dumps(r))
if __name__=='__main__':main()
