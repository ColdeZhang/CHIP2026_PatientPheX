from pathlib import Path
import json,hashlib,sys,tarfile,shlex,io as pyio,datetime
E=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry');F=E/'formal-ack-recovery';Q=F/'paired-evaluation'
B=E/'terminal-backup';B.mkdir(exist_ok=False)
files=[*Q.rglob('*.json'),Q/'original/pred.jsonl',Q/'reverse-first/pred.jsonl',F/'original/responses.jsonl',F/'reverse-first/responses.jsonl',F/'original/responses.manifest.json',F/'reverse-first/responses.manifest.json',F/'all-responses-complete.json',F/'reverse-first/inference-gpu-release.json',F/'terminal-result.json',E/'formal/execution-state.json']
files += [(E/'formal' if a=='original' and ep==1 else F)/a/f'epoch-{ep:03d}/export-ack.json' for a in ['original','reverse-first'] for ep in [1,2]]
records={str(p.relative_to(E)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
manifest={'status':'frozen_terminal_raw_and_full_predictions_no_gold','source_git_checkpoint':'e3f6bdf9fda63961b1d754074b66824c4effe1f8','members':records,'total_bytes':sum(r['bytes'] for r in records.values()),'weights':'Four full states already independently persisted on .100; only ACKs in this archive.'}
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=B/'paired-result-e3f6bdf.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(E)),recursive=False)
 t.add(B/'manifest.json',arcname='backup-manifest.json',recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest();remote='case-retrieval/qwen35-preparation/order-warmstart-stage/terminal-backup/'+archive.name
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
io.M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
pre=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();print("unique terminal backup path available")'),'cwd':'.','timeout_seconds':10})
assert pre['exit_code']==0; (B/'remote-preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
io.upload(str(archive),remote)
code='''from pathlib import Path
import hashlib,json,tarfile,os,datetime
p=Path(REMOTE);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==DIGEST
with p.open('rb') as f:os.fsync(f.fileno())
dfd=os.open(str(p.parent),os.O_RDONLY)
try:os.fsync(dfd)
finally:os.close(dfd)
with tarfile.open(p,'r:gz') as t:
 m=json.load(t.extractfile('backup-manifest.json'));assert len(t.getnames())==len(m['members'])+1
 for n,r in m['members'].items():
  b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
print(json.dumps(dict(status='persistent_archive_all_members_verified',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),members=len(m['members']),uncompressed_payload_bytes=m['total_bytes'],fsync=True,no_gpu=True,no_gold=True)))
'''.replace('REMOTE',repr(remote)).replace('DIGEST',repr(digest))
r=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':20})
(B/'remote-readback.json').write_text(json.dumps(r,indent=2)+'\n');assert r['exit_code']==0
receipt=json.loads(r['stdout']);receipt['mount_id']=io.M;receipt['local_archive']=str(archive);receipt['manifest_sha256']=hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest()
(B/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
