"""Future complete F0 supplement using the already proven same-byte persistence protocol.

No packing before actual complete94, fixed normalization, quality and final review.
No model, parser, evaluator, bank encoding, or gold access.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,stat,time,zipfile
from pathlib import Path
S=Path(__file__).resolve().parents[1]
B=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1')
PRIOR=Path(__file__).with_name('F1_supplement.py')
spec=importlib.util.spec_from_file_location('_sampled_F1_protocol',PRIOR);prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
m=prior.m;m.TARGET='research/grounded-sampled-consensus-v1/F0-endpoint-supplement-20261024'
m.REMOTE_VERIFY=m.REMOTE_VERIFY.replace('grounded-sampled-F1-endpoint-persistent-v1','grounded-sampled-F0-endpoint-persistent-v1')

def read(p):return json.loads(Path(p).read_text())
def checked(x):assert m.ref(x['path'])==x;return Path(x['path'])

def inventory(a):
 entry_path=B/'F0-complete-review-entry.json';review_path=Path(a.review).resolve()
 assert m.ref(entry_path)['sha256']==a.entry_sha256 and m.ref(review_path)['sha256']==a.review_sha256
 entry=read(entry_path);assert entry['actual_generation']['rows']==94 and entry['new_v2'] in (0,1) and entry['old_v2']==0
 assert str(read(review_path)['status']).startswith('PASS')
 entries={'runs/F0-complete-review-entry.json':entry_path,'review/review.json':review_path}
 for relative,item in entry['source_refs'].items():
  path=checked(item);assert path.resolve().is_relative_to(S)
  top=Path(relative).parts[0]
  if top in {'f0','postprocess_F0','normalization_F0','reporting'}:entries['source/'+relative]=path
 for relative,item in entry['refs'].items():
  path=checked(item);assert path.resolve().is_relative_to(B)
  if relative.startswith(('native-F0-cpu-v1/','infer-F0-A30-v1/','normalization-F0-runtime-v1/','F0/')):entries['runs/'+relative]=path
 for folder in ['extraction-v1','normalization-preparation-v1','normalized-v1','quality-F0-v1']:
  for path in sorted((B/'F0'/folder).rglob('*')):
   if path.is_file():entries['runs/'+path.relative_to(B).as_posix()]=path
 quality=read(S/'postprocess_F0/quality-plan.json')
 for name,item in quality['sources']['reused'].items():entries['inherited-source/'+name]=checked(item)
 for name in ['baseline','baseline_metrics','baseline_freeze','baseline_plan']:
  item=quality['inputs'][name];entries['baseline/'+name+Path(item['path']).suffix]=checked(item)
 prep=read(B/'F0/normalization-preparation-v1/manifest.json');phases={}
 names=[('native-F0-cpu-v1','CPU_reported_success'),('infer-F0-A30-v1','inference_reported_success')]
 if prep['necessary_public_query_count']:names.append(('normalization-F0-runtime-v1','query_reported_success'))
 for name,key in names:
  directory=B/name;c=read(directory/'collection.json');assert c['all_actual_SHA_verified'] and c[key]
  if name=='infer-F0-A30-v1':assert c['wrapper']['complete94_actual_members_verified']
  phases[name]=c
  for suffix in ['collection.json','collection-transfer/archive-binding.json','package.json']:entries['runs/'+name+'/'+suffix]=directory/suffix
  entries['runs/'+name+'/member-index.json']=checked(c['member_index']);archive=checked(c['archive']);entries['runs/'+name+'/original-'+archive.name]=archive
  with zipfile.ZipFile(archive) as z:
   data=z.read('member-index.json');assert data==Path(c['member_index']['path']).read_bytes();index=json.loads(data)
   assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(index['files'])|{'member-index.json'}
   assert len(index['files'])==len(c['files'])
   for member,item in index['files'].items():
    raw=z.read(member);assert len(raw)==item['bytes'] and m.digest(raw)==item['sha256']
 for name in ['manifest.json','snapshot-manifest.json','evidence-manifest.json']:
  p=review_path.parent/name
  if p.is_file():entries['review/'+name]=p
 for rel in ['persistence/F1-receipt.json','evidence/F1-actual-restore.json','persistence/F0_supplement.py']:
  entries['source/'+rel]=S/rel
 entries['transport/F1_supplement.py']=PRIOR;entries['transport/original_endpoint_supplement.py']=prior.TRANSPORT
 locators=dict(prior_F1_restored_chain=read(S/'persistence/F1-receipt.json'),prior_F1_actual_restore=read(S/'evidence/F1-actual-restore.json'),original_terminal_archives={n:dict(mount=c['mount'],archive=c['remote_archive']) for n,c in phases.items()},public_assets=read(S/'normalization_F0/target-assets.json'),same_d52_model_and_complete_state='Use actual F1 backup chain, unchanged adapter/base/full-state locator; no weight copies',phase_member_counts={n:len(c['files']) for n,c in phases.items()},no_NN_no_score_no_decode=True)
 for name,p in entries.items():m.safe(name);assert p.is_file() and not p.is_symlink() and p.suffix not in {'.pt','.safetensors','.npy','.bin'}
 return entries,locators,review_path

def pack(a,out):
 assert not (out/'package.json').exists(),'Original package immutable; do not repackage'
 entries,locators,review=inventory(a);payload={};secrets=[v.encode() for k,v in os.environ.items() if len(v)>8 and any(s in k.upper() for s in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
 for name,p in entries.items():
  data=p.read_bytes();assert not any(v in data for v in secrets);payload[name]=data
 manifest=dict(schema='grounded-sampled-F0-endpoint-members-v1',members={n:dict(source_path=str(entries[n].resolve()),bytes=len(b),sha256=m.digest(b)) for n,b in payload.items()},external_resources=locators,original_archives_copied_byteexact=True,exclusions=['prior F1 raw archive duplication','gold','large model/optimizer/RNG/publicbank','transport chatter'],no_NN_no_score_no_decode=True)
 encoded=m.encoded(manifest);assert len(encoded)<=m.CHUNK;m.publish(out/'members-manifest.json',encoded);archive=out/'endpoint-supplement.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,value in sorted(dict(payload,**{'members-manifest.json':encoded}).items()):
   info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=(stat.S_IFREG|0o600)<<16;z.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 with archive.open('rb') as f:os.fsync(f.fileno())
 m.sync(out)
 with zipfile.ZipFile(archive) as z:
  assert len(z.namelist())==len(payload)+1 and set(z.namelist())==set(payload)|{'members-manifest.json'} and z.read('members-manifest.json')==encoded
  for name,value in payload.items():assert z.read(name)==value
 raw=archive.read_bytes();chunks=[]
 for offset in range(0,len(raw),65536):
  b=raw[offset:offset+65536];chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(b),sha256=m.digest(b)))
 p=dict(archive=m.ref(archive),manifest=m.ref(out/'members-manifest.json'),members=len(payload),unpacked_bytes=sum(map(len,payload.values())),target_mount=m.MOUNT,target=m.TARGET,chunks=chunks,source=m.ref(__file__),transport_source=m.ref(prior.TRANSPORT),external_archives=[],external_resources=locators,package_readback_exact=True,all_original_sources_SHA_in_manifest=True,review=m.ref(review))
 m.publish(out/'package.json',m.encoded(p));print(json.dumps(dict(package=m.ref(out/'package.json'),archive=p['archive'],manifest=p['manifest'],content_members=len(payload),ZIP_members=len(payload)+1)))

if __name__=='__main__':
 assert __debug__;ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inventory','pack','deploy']);ap.add_argument('--entry-sha256');ap.add_argument('--review');ap.add_argument('--review-sha256');a=ap.parse_args();os.umask(0o077);prior.guard_gold()
 if a.stage in ['inventory','pack']:assert a.entry_sha256 and a.review and a.review_sha256
 if a.stage=='inventory':
  e,_,_=inventory(a);print(json.dumps(dict(content_members=len(e),bytes=sum(p.stat().st_size for p in e.values()),local_inventory_only=True)));raise SystemExit(0)
 out=B/'endpoint-F0-supplement-v1';out.mkdir(exist_ok=True);owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.stage=='pack':pack(a,out)
 else:
  for p in (out/'deployment').glob('*.backoff.json'):assert time.time()>=read(p)['retry_not_before_unix']+5
  m.deploy(B,out,Path('/workspace/patientphex-tools'))
