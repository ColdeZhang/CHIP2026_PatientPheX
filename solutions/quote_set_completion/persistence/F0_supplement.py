"""One reviewed F0 delta; original F1 model/source package is a locator, not copied."""
import argparse,fcntl,importlib.util,json,os,stat,time,zipfile
from pathlib import Path
S=Path(__file__).resolve().parents[1]
N=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1')
PRIOR=S/'persistence/F1_supplement.py'
spec=importlib.util.spec_from_file_location('_QSC_F1_transport',PRIOR)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
m=p.m
m.TARGET='research/quote-set-completion-v2/F0-supplement-v1'
m.REMOTE_VERIFY=m.REMOTE_VERIFY.replace('quote-set-completion-F1-endpoint-persistent-v1','quote-set-completion-F0-endpoint-persistent-v1')
PHASES={'inference-F0-A30-v1':(108,'inference_reported_success',True),'normalization-F0-v1':(14,'query_reported_success',True)}
REVIEW=None

def read(path):return json.loads(Path(path).read_text())
def checked(v):
 q=Path(v['path']);assert m.ref(q)==v;return q

def inventory():
 entry=N/'F0-complete-review-entry.json'
 assert m.ref(entry)['sha256']=='caa6ed945fa404c0cc24de3ba9319ffe08f9e1805f29a1dfd18342413acd9bfb'
 e=read(entry);assert e['status']=='F0_NONPOSITIVE_METHOD_CLOSED' and not e['result']['keep']
 assert e['result']['new_v2']==1 and e['result']['old_v2']==0 and e['generation']['rows']==47
 entries={'runs/F0-complete-review-entry.json':entry}
 for name,v in e['source_refs'].items():
  q=checked(v);assert q.is_relative_to(S);entries['source/'+name]=q
 for name,v in e['refs'].items():
  q=checked(v);assert q.is_relative_to(N);entries['runs/'+name]=q
 for folder in ['F0-extraction-v1','F0-normalization-preparation-v1','F0-normalized-v1','quality-F0-v1']:
  for q in sorted((N/folder).iterdir()):
   assert q.is_file() and not q.is_symlink();entries['runs/'+q.relative_to(N).as_posix()]=q
 hp=read(N/'F0-input-preparation-v1/manifest.json');assert hp['rows']==47 and hp['known_triples']==204 and hp['ancestry']['disjoint']
 for v in hp['files'].values():
  q=checked(v);entries['runs/'+q.relative_to(N).as_posix()]=q
 collections={}
 for name,(count,success,expected) in PHASES.items():
  D=N/name;c=read(D/'collection.json');assert c['all_actual_SHA_verified'] and len(c['files'])==count and c[success] is expected;collections[name]=c
  for suffix in ['collection.json','collection-transfer/archive-binding.json']:entries['runs/'+name+'/'+suffix]=D/suffix
  q=checked(c['package']);entries['runs/'+name+'/'+q.name]=q
  idxpath=checked(c['member_index']);entries['runs/'+name+'/member-index.json']=idxpath
  archive=checked(c['archive']);entries['runs/'+name+'/original-'+archive.name]=archive
  with zipfile.ZipFile(archive) as z:
   raw=z.read('member-index.json');assert raw==idxpath.read_bytes();index=json.loads(raw)
   assert len(index['files'])==count and len(z.namelist())==count+1 and set(z.namelist())==set(index['files'])|{'member-index.json'}
   for key,v in index['files'].items():
    m.safe(key);b=z.read(key);assert len(b)==v['bytes'] and m.digest(b)==v['sha256']
 for name in ['package.json','persistent-receipt.json','root-actual-restore-v1/complete.json']:
  entries['prior-F1/'+name]=N/'endpoint-F1-supplement-v1'/name
 actual=N/'endpoint-F1-supplement-v1/root-actual-restore-v1/complete.json'
 assert m.ref(actual)['sha256']=='bf565288008727f7d100e3c8ff56ae8dc8451138e63b1f539daf99cd31057a73'
 review=read(REVIEW/'review.json');assert str(review['status']).startswith('PASS')
 manifest=read(REVIEW/'manifest.json')
 for name in ['review.json','manifest.json']:entries['review/'+name]=REVIEW/name
 def safe_refs(value):
  if isinstance(value,dict):
   if {'path','bytes','sha256'}.issubset(value):
    q=Path(value['path'])
    if q.is_relative_to(REVIEW) and q.suffix in {'.py','.json','.md','.txt'} and value['bytes']<100000:
     assert m.ref(q)=={k:value[k] for k in ['path','bytes','sha256']};entries['review/'+q.relative_to(REVIEW).as_posix()]=q
   for v in value.values():safe_refs(v)
  elif isinstance(value,list):
   for v in value:safe_refs(v)
 safe_refs(manifest)
 entries['source/persistence/F0_supplement.py']=Path(__file__)
 entries['transport/F1_supplement.py']=PRIOR
 entries['transport/sampled_F1_supplement.py']=p.PRIOR
 entries['transport/original_endpoint_supplement.py']=p.prior.TRANSPORT
 for key,q in entries.items():
  m.safe(key);assert q.is_file() and not q.is_symlink() and q.suffix not in {'.pt','.safetensors','.npy','.bin'} and not q.name.endswith('.gold.jsonl')
 locators=dict(prior_F1_package=read(N/'endpoint-F1-supplement-v1/package.json'),prior_F1_actual_restore=m.ref(actual),current_actual_archives={k:dict(mount=v['mount'],archive=v['remote_archive'],content_members=len(v['files'])) for k,v in collections.items()},old_model_optimizer_in_prior_ACK=True,no_old_weights_or_F1payload_duplication=True,no_NN_score_decode=True)
 return entries,locators

def pack(out):
    assert not (out/'package.json').exists(),'Original package immutable; do not repackage'
    entries,locators=inventory();payload={}
    secrets=[v.encode() for k,v in os.environ.items() if len(v)>8 and any(s in k.upper() for s in ['TOKEN','PASSWORD','PASSWD','COOKIE','SECRET'])]
    for name,path in entries.items():
        data=path.read_bytes();assert not any(secret in data for secret in secrets);payload[name]=data
    manifest=dict(schema='quote-set-completion-F0-endpoint-members-v1',
        members={name:dict(source_path=str(entries[name].resolve()),bytes=len(data),sha256=m.digest(data)) for name,data in payload.items()},
        external_resources=locators,original_terminal_ZIPs_copied_byteexact=True,
        original_terminal_content_counts={name:value[0] for name,value in PHASES.items()},
        failed_and_partial_originals_preserved_as_failed=True,
        exclusions=['old F1 payloads already restored','conditional full64 CPU readiness','held gold bodies','weights/optimizer/RNG/public bank','transport chatter'],
        no_NN_no_score_no_decode=True)
    encoded=m.encoded(manifest);assert len(encoded)<=m.CHUNK
    m.publish(out/'members-manifest.json',encoded);archive=out/'endpoint-supplement.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in sorted(dict(payload,**{'members-manifest.json':encoded}).items()):
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|0o600)<<16
            z.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with archive.open('rb') as f:os.fsync(f.fileno())
    m.sync(out)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(payload)+1 and set(z.namelist())==set(payload)|{'members-manifest.json'}
        assert z.read('members-manifest.json')==encoded
        for name,value in payload.items():assert z.read(name)==value
    raw=archive.read_bytes();chunks=[]
    for offset in range(0,len(raw),65536):
        piece=raw[offset:offset+65536]
        chunks.append(dict(name=f'chunks/{offset:012d}.bin',offset=offset,bytes=len(piece),sha256=m.digest(piece)))
    package=dict(archive=m.ref(archive),manifest=m.ref(out/'members-manifest.json'),members=len(payload),
        unpacked_bytes=sum(map(len,payload.values())),target_mount=m.MOUNT,target=m.TARGET,chunks=chunks,
        source=m.ref(__file__),transport_source=m.ref(p.prior.TRANSPORT),prior_protocol_source=m.ref(PRIOR),
        external_archives=[],external_resources=locators,package_readback_exact=True,
        all_original_sources_SHA_in_manifest=True,review=m.ref(REVIEW/'review.json'))
    m.publish(out/'package.json',m.encoded(package))
    print(json.dumps(dict(package=m.ref(out/'package.json'),archive=package['archive'],manifest=package['manifest'],
        content_members=len(payload),ZIP_members_including_embedded_manifest=len(payload)+1,chunks=len(chunks))))


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['pack','deploy']);a.add_argument('--review-root');args=a.parse_args()
 assert __debug__;os.umask(0o077);p.prior.guard_gold();out=N/'endpoint-F0-supplement-v1';out.mkdir(exist_ok=True)
 owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if args.stage=='pack':
  assert args.review_root;REVIEW=Path(args.review_root).resolve();assert REVIEW.is_relative_to(N.parents[1]/'validation')
  pack(out)
 else:
  for q in (out/'deployment').glob('*.backoff.json'):assert time.time()>=read(q)['retry_not_before_unix']+5
  m.deploy(N,out,Path('/workspace/patientphex-tools'))
