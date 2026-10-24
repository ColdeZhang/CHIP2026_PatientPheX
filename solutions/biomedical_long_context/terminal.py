"""Freeze concise terminal evidence; no model, labels or quality computation."""
import argparse,base64,hashlib,io,json,os,shlex,zipfile
from pathlib import Path
from rpc import call
D=Path('/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024')
S=Path(__file__).resolve().parent
MOUNT='1593f159-19bc-41ca-8b4c-2e7455f5db99'
REL='research/biomedical-long-context/terminal-v1/source-quality.zip'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def prepare():
 assert not (D/'final-review-entry.json').exists()
 fc=D/'formal-v1/collected';nc=D/'normalization-v1/collected';q=D/'quality-v1'
 result=json.loads((q/'result.json').read_text());assert result['new_v2']==1 and not result['retain']
 entry=dict(schema='bioclinical-longcontext-final-review-v1',status='FIXED_PROTECTED_UNION_NEGATIVE_CLOSED',
  quality={p.name:ref(p) for p in sorted(q.iterdir()) if p.is_file()},
  formal={n:ref(fc/n) for n in ('output/complete.json','output/final-state-ACK.json','job/ended.json','output/inputs.json','output/loading.json','output/baseline-span-scores.json')},
  normalization={n:ref(nc/n) for n in ('linked/complete.json','job/ended.json','input-binding.json')},
  recovery={n:ref(D/n) for n in ('formal-v1/collection-complete.json','formal-v1/collection-inventory.json','normalization-v1/collection-complete.json','normalization-v1/collection-inventory.json')},
  source={str(p.relative_to(S)):ref(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.md') and '__pycache__' not in str(p) and 'evidence' not in p.parts},
  result=result,counts_delta_from_cached_metrics=dict(mention=dict(tp=37,fp=173,fn=-37),document=dict(tp=26,fp=94,fn=-26),relation=dict(tp=0,fp=0,fn=0)),
  remote_final_state=json.loads((fc/'output/complete.json').read_text())['checkpoint'],
  limits=['All 32 train documents fit one8192 window; F1 one article needs two overlapping windows (17 total).','Joint backbone/tokenizer/context/passage-loss change; no pure-context causal attribution.','Old strong selector uses train48 excluding F1; new GP uses F2F3 train32.','Final state durable remotely with tensor reopen/restore ACK; not downloaded locally and no trajectory-resume equivalence claim.','Public pretrained MLM upstream raw clinical text is not external manually annotated phenotype labels; per-source upstream preprocessing unavailable.','Source/public/raw and full state recovery depends on referenced archives/assets; terminal small package is not self-contained.'],
  root_separate_veto={'path':'/workspace/patientphex-runs/round2-20261023/research/long-context-span-veto-v1/F1-v1/result.json','owner':'root','same_formal_parent_and_all1313_score_file':True,'not_part_of_this_route_quality_calls':True},
  no_F0_or_originaldev_or_official=True)
 save(D/'final-review-entry.json',entry);save(S/'evidence/final-entry.json',dict(entry=ref(D/'final-review-entry.json'),status=entry['status'],result=ref(q/'result.json')))
 print(json.dumps(ref(D/'final-review-entry.json')))
def backup():
 out=D/'terminal-backup-v1';out.mkdir(exist_ok=True)
 archive=out/'source-quality.zip';manifest=out/'members.json'
 if not archive.exists():
  members={'final-review-entry.json':D/'final-review-entry.json'}
  for p in (D/'quality-v1').iterdir():
   if p.is_file():members['quality-v1/'+p.name]=p
  for p in sorted(S.rglob('*')):
   if p.is_file() and p.suffix in ('.py','.json','.md') and '__pycache__' not in str(p):members['source/'+str(p.relative_to(S))]=p
  for n in ('formal-v1/collection-complete.json','formal-v1/collection-inventory.json','normalization-v1/collection-complete.json','normalization-v1/collection-inventory.json'):
   members[n]=D/n
  refs={n:ref(p) for n,p in members.items()};buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
   for n,p in members.items():z.writestr(n,p.read_bytes())
   z.writestr('MANIFEST.json',json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
  with archive.open('xb') as f:f.write(buf.getvalue());f.flush();os.fsync(f.fileno())
  save(manifest,refs)
 raw=archive.read_bytes();assert len(raw)<1024*1024
 refs=json.loads(manifest.read_text());remote=P+'/'+REL
 # No automatic mutation retry after UNKNOWN: original intent and actual file are recoverable.
 if not (out/'upload-ack.json').exists():
  v=call(out,'write_file',dict(mount_id=MOUNT,path=REL,content=base64.b64encode(raw).decode(),encoding='base64',create_parents=True),'upload-small-terminal');save(out/'upload-ack.json',v)
 code='''import pathlib,hashlib,os,json
p=pathlib.Path(%r)
with p.open('rb') as f:b=f.read();os.fsync(f.fileno())
q=p.parent
while True:
 fd=os.open(q,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 if q.parent==q:break
 q=q.parent
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),file_and_ancestors_fsynced=True)))
'''%remote
 if not (out/'remote-verified.json').exists():
  v=call(out,'run_command',dict(mount_id=MOUNT,cwd='.',command='python3 -c '+shlex.quote(code),timeout_seconds=30),'verify-small-terminal');assert v['exit_code']==0;rr=json.loads(v['stdout']);assert rr['sha256']==hashlib.sha256(raw).hexdigest();save(out/'remote-verified.json',rr)
 parts=out/'readback';parts.mkdir(exist_ok=True)
 for offset in range(0,len(raw),32768):
  p=parts/str(offset);length=min(32768,len(raw)-offset)
  if p.exists():assert p.read_bytes()==raw[offset:offset+length];continue
  v=call(out,'read_file',dict(mount_id=MOUNT,path=REL,offset=offset,limit=length),'readback-'+str(offset));b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode();assert b==raw[offset:offset+length]
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 restored=b''.join((parts/str(o)).read_bytes() for o in range(0,len(raw),32768));assert restored==raw
 with zipfile.ZipFile(io.BytesIO(restored)) as z:
  assert set(z.namelist())==set(refs)|{'MANIFEST.json'}
  for n,v in refs.items():b=z.read(n);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 receipt=dict(status='COMPLETE_REMOTE_ACTUAL_READBACK_ALL_MEMBERS',archive=ref(archive),members=len(refs),container_members=len(refs)+1,mount=MOUNT,remote=json.loads((out/'remote-verified.json').read_text()),new_model_or_quality_calls=0,parents='formal10 and normalization8 member archives reused; no1.8GB model transfer')
 save(out/'complete.json',receipt);save(S/'evidence/terminal-backup.json',dict(receipt=ref(out/'complete.json'),archive=receipt['archive'],members=receipt['members']))
 print(json.dumps(ref(out/'complete.json')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','backup']);a=ap.parse_args();globals()[a.action]()
