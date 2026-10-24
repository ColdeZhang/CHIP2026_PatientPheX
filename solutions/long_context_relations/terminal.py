"""Freeze concise terminal evidence; no model, labels or quality computation."""
import argparse,base64,hashlib,io,json,os,shlex,zipfile
from pathlib import Path
from rpc import call
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1')
S=Path(__file__).resolve().parent
MOUNT='1593f159-19bc-41ca-8b4c-2e7455f5db99'
REL='research/long-context-relations/terminal-v1/source-quality.zip'
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def prepare():
 assert not(D/'final-review-entry.json').exists()
 fc=D/'formal-v1/collected';q=D/'quality-v1';result=json.loads((q/'result.json').read_text())
 assert result['new_v2'] in (0,1) and result['old_v2']==0
 old=json.loads(Path(json.loads((S/'quality-plan.json').read_text())['baseline_metrics']['path']).read_text());new=json.loads((q/'metrics.json').read_text())
 entry=dict(schema='long-context-relations-final-review-v1',status='COMPLETE_FIXED_B4_PROTECTED_UNION',
   quality={p.name:ref(p)for p in sorted(q.iterdir())if p.is_file()},
   formal={str(p.relative_to(fc)):ref(p)for p in sorted(fc.rglob('*'))if p.is_file()},
   recovery={n:ref(D/n)for n in ('formal-v1/collection-complete.json','formal-v1/collection-inventory.json')},
   source={str(p.relative_to(S)):ref(p)for p in sorted(S.rglob('*'))if p.is_file()and p.suffix in ('.py','.json','.md')and '__pycache__'not in str(p)and'evidence'not in p.parts},
   result=result,count_delta_from_cached_metrics={k:{v:new['counts'][k][v]-old['counts'][k][v]for v in ('tp','fp','fn')}for k in old['counts']},
   checkpoint=json.loads((fc/'output/complete.json').read_text())['checkpoint'],
   limits=['Fresh public149M parent, no Task1/probe state reuse.','Joint backbone/tokenizer/globalcontext/context-head change, binary BCE and multiple-mention pooling previously tested.','Original public candidate reachability727/818train positives; no fabricated absent concepts.','Same-span10concept groups20IDs and2conflicting-target groups not distinguished by new representation.','Only F2F3train32 labels used; F1pure input16docs39patients2874pairs.','Original token sequence complete32train/17held windows, no invented across-window encoder attention.','State full model/Adam/scheduler/RNG was reopened/restored and tensor-checked; no trajectory resume equivalence claim.','1.79GB fullstate remote only; small package is not self-contained and uses prior exact assets/data.'],
   no_F0_originaldev_or_official=True)
 save(D/'final-review-entry.json',entry);save(S/'evidence/final-entry.json',dict(entry=ref(D/'final-review-entry.json'),result=ref(q/'result.json'),keep=result['retain']))
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
  for n in ('formal-v1/collection-complete.json','formal-v1/collection-inventory.json'):
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
 receipt=dict(status='COMPLETE_REMOTE_ACTUAL_READBACK_ALL_MEMBERS',archive=ref(archive),members=len(refs),container_members=len(refs)+1,mount=MOUNT,remote=json.loads((out/'remote-verified.json').read_text()),new_model_or_quality_calls=0,parents='formal9-member archive reused; no1.79GB model transfer')
 save(out/'complete.json',receipt);save(S/'evidence/terminal-backup.json',dict(receipt=ref(out/'complete.json'),archive=receipt['archive'],members=receipt['members']))
 print(json.dumps(ref(out/'complete.json')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','backup']);a=ap.parse_args();globals()[a.action]()
