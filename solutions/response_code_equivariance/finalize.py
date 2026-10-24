"""Verify collected identities, expose one review input and persist one small bundle.

No model, tokenizer, response parser, gold loader or evaluator is called here.
The local checks are an owner integrity check, not an independent quality review.
"""
from pathlib import Path
import argparse,base64,hashlib,io,json,os,shlex,sys,time,zipfile
W=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/response-code-equivariance-v1')
D=B/'gpu-v1';M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def sha(b):return hashlib.sha256(b).hexdigest()
def ref(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=sha(b))
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def check(p,r):
 a=ref(p);assert (a['bytes'],a['sha256'])==(r['bytes'],r['sha256'])
def prepare():
 T=D/'terminal';Q=D/'quality';plan=read(D/'plan.json');manifest=read(D/'terminal-manifest.json')
 for r in manifest['files']:check(T/r['name'],r)
 check(D/'terminal.zip',dict(bytes=manifest['zip_bytes'],sha256=manifest['zip_sha256']))
 with zipfile.ZipFile(D/'terminal.zip') as z:
  assert set(z.namelist())=={r['name'] for r in manifest['files']}
  for r in manifest['files']:
   b=z.read(r['name']);assert len(b)==r['bytes'] and sha(b)==r['sha256']
 s=read(T/'summary.json');w=read(T/'wrapper-complete.json');actual=read(T/'actual-input-and-assets.json')
 assert w['returncode']==0 and not read(T/'gpu-release.json')['target_compute_rows']
 assert s['fresh_rows']==53 and s['cached_rows']==27 and s['expected_rows']==80
 assert s['plan_sha256']==ref(D/'plan.json')['sha256'] and s['source_sha256']==plan['files']['infer_full80.py']['sha256']
 assert s['total_parameters']==8997081600 and actual['all80_actual_ids_equal_prior_native_CPU']
 assert actual['torch']=='2.7.1+cu126' and actual['max_input_tokens']==14570 and actual['input_tokens']==685372
 cache=read(B/'preparation-v1/single-patient-cache-refs.json')
 for item in cache['cache']:
  i=item['original_index']
  for folder,r in item['files'].items():check(T/f'{folder}/{i:03d}.json',r)
 for folder in ['generated-ids','raw-responses','canonical-records']:
  assert {p.name for p in (T/folder).glob('*.json')}=={f'{i:03d}.json' for i in range(80)}
 q=read(Q/'result.json');freeze=read(Q/'common-freeze.json')
 assert q['old_quality_calls']==0 and q['new_quality_calls'] in [1,2]
 assert not q['automatic_F0_dev_official']
 for arm,r in freeze['predictions'].items():check(r['path'],r)
 for result in q['arms'].values():check(result['metrics_ref']['path'],result['metrics_ref'])
 integrity=dict(status='OWNER_SHA_COMPLETENESS_CHECK_PASS_NOT_INDEPENDENT_REVIEW',terminal_members=len(manifest['files']),cache_files_exact=81,generated_shards=80,raw_shards=80,canonical_shards=80,model_calls=53,quality_calls=q['new_quality_calls'],source_checkpoint='876b02b830e272f20171233f15c203fe67e3a2e0',summary=ref(T/'summary.json'),actual=ref(T/'actual-input-and-assets.json'),terminal_manifest=ref(D/'terminal-manifest.json'),quality_result=ref(Q/'result.json'),scope='mapping plus host/runtime development control; no F0/dev/official')
 save(B/'owner-integrity.json',integrity)
 review=dict(status='ACTUAL_COMPLETE_SINGLE_INCREMENTAL_REVIEW_INPUT',owner_integrity=ref(B/'owner-integrity.json'),source_plan=ref(D/'plan.json'),deployment=ref(D/'deployment-manifest.json'),native_terminal=ref(B/'native100-v1/terminal-manifest.json'),cache_refs=ref(B/'preparation-v1/single-patient-cache-refs.json'),terminal_manifest=ref(D/'terminal-manifest.json'),terminal_zip=ref(D/'terminal.zip'),quality_plan=ref(W/'evidence/quality-plan.json'),common_freeze=ref(Q/'common-freeze.json'),quality_result=ref(Q/'result.json'),policy=ref(B/'endpoint-policy.json'),diagnostic_addendum=ref(W/'evidence/endpoint-policy-diagnostic-reporting-addendum.json'),new_quality_calls=q['new_quality_calls'],old_quality_calls=0,review_scope='New 53 generation + 27 cache identities, complete two-endpoint freeze and calls; reuse prior C/native source proofs, no repeated model/tokenizer/parser/evaluation.')
 save(B/'review-input.json',review)
 save(W/'evidence/GPU-terminal-summary.json',dict(summary=s,wrapper=w,release=read(T/'gpu-release.json'),terminal_manifest=ref(D/'terminal-manifest.json'),review_input=ref(B/'review-input.json')))
 save(W/'evidence/quality-result.json',q)
 # Full content stays outside Git. Include input/source transport, terminal and
 # frozen predictions; model weights already live in referenced verified store.
 files={'deployment.tar.gz':D/'deployment.tar.gz','terminal.zip':D/'terminal.zip','native-terminal.zip':B/'native100-v1/terminal.zip'}
 for p in [B/'owner-integrity.json',B/'review-input.json',B/'endpoint-policy.json',B/'preparation-v1/single-patient-cache-refs.json',B/'preparation-v1/row-contract.json',B/'native100-v1/terminal-manifest.json',D/'plan.json',D/'deployment-manifest.json',D/'terminal-manifest.json',D/'job.json',D/'start-intent.json']:
  files['records/'+p.relative_to(B).as_posix()]=p
 for p in Q.rglob('*'):
  if p.is_file():files['quality/'+p.relative_to(Q).as_posix()]=p
 for p in W.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files['source/'+p.relative_to(W).as_posix()]=p
 dest=B/'content-backup-v1.zip'
 with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.writestr(n,p.read_bytes())
 manifest=dict(status='PREPARED_LOCAL_NOT_YET_REMOTE_BACKED_UP',archive=ref(dest),members={n:ref(p) for n,p in files.items()})
 save(B/'content-backup-manifest.json',manifest)
 print(json.dumps(dict(review_input=ref(B/'review-input.json'),archive=ref(dest),members=len(files),quality=q)))

def backup():
 sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
 c=Client();c.initialize();manifest=read(B/'content-backup-manifest.json');archive=Path(manifest['archive']['path']).read_bytes();assert sha(archive)==manifest['archive']['sha256']
 receipts=B/'backup-v1';receipts.mkdir(exist_ok=True);rel='patient-response-code-equivariance/evidence-backup-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel
 def call(name,args,label):
  r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=M,**args)));save(receipts/(label+'-'+str(time.time_ns())+'.json'),r)
  assert not r.get('isError'),'Unknown outcome: inspect existing backup pieces before retry'
  return r.get('structuredContent') or json.loads(r['content'][0]['text'])
 def remote(code,p,label):
  s='P='+repr(p)+'\n'+code;compile(s,'remote-backup.py','exec');r=call('run_command',dict(command='python3 -c '+shlex.quote(s),timeout_seconds=30),label);assert r['exit_code']==0;return json.loads(r['stdout'])
 inv=remote("from pathlib import Path\nimport json,hashlib,os\np=Path(P['root']);p.mkdir(parents=True,exist_ok=True,mode=0o700);assert os.getuid()==p.stat().st_uid==1006 and not p.is_symlink();print(json.dumps({x.name:dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in p.glob('part-*')}))",dict(root=root),'inventory')
 parts=[]
 for offset in range(0,len(archive),32768):
  b=archive[offset:offset+32768];name=f'part-{offset:09d}';r=dict(bytes=len(b),sha256=sha(b));parts.append(dict(name=name,**r))
  if name in inv:assert inv[name]==r;continue
  x=call('write_file',dict(path=rel+'/'+name,content=base64.b64encode(b).decode(),encoding='base64',create_parents=False),name);assert x['size']==len(b) and x['sha256']==sha(b)
 code="""from pathlib import Path
import json,hashlib,os,zipfile,io
p=Path(P['root']);parts=[]
for r in P['parts']:
 b=(p/r['name']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];parts.append(b)
b=b''.join(parts);m=P['manifest'];assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256'];dest=p/'content-backup-v1.zip'
if dest.exists():assert dest.read_bytes()==b
else:
 with dest.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
with dest.open('rb') as f:actual=f.read();os.fsync(f.fileno())
assert actual==b
with zipfile.ZipFile(io.BytesIO(actual)) as z:
 assert set(z.namelist())==set(m['members'])
 for n,r in m['members'].items():
  x=z.read(n);assert len(x)==r['bytes'] and hashlib.sha256(x).hexdigest()==r['sha256']
fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ACTUAL_PERSISTENT_FULL_CONTENT_SHA_FSYNC_REOPEN',archive=str(dest),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),members=len(m['members']),mount_id=P['mount'],at_unix=__import__('time').time())))
"""
 receipt=remote(code,dict(root=root,parts=parts,manifest=manifest,mount=M),'final-verify')
 save(B/'content-backup-receipt.json',receipt);save(W/'evidence/content-backup-receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare-review','backup']);a=ap.parse_args();prepare() if a.action=='prepare-review' else backup()
