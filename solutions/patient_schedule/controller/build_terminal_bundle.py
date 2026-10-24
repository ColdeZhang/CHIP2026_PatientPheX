"""Bundle frozen terminal evidence privately, excluding weights and held gold."""
from pathlib import Path
import argparse,hashlib,json,tarfile
p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C=a.controller;D=a.output;S=Path(__file__).resolve().parents[1];B=C.parent
assert json.loads((C/'campaign-complete.json').read_text())['status']=='FOUR_COMPLETE_EPOCHS_64_STEPS_DURABLE_ACKS_VERIFIED'
assert json.loads((C/'quality/result.json').read_text())['status']=='COMPLETE_FIXED_PATIENT4_F1'
assert json.loads((C/'inference-collected.json').read_text())['all_local_SHA']
D.mkdir(exist_ok=False);paths={}
def add(name,path):
 path=Path(path);assert path.is_file() and not path.is_symlink();assert name not in paths;paths[name]=path
for path in S.rglob('*'):
 if path.is_file() and path.suffix in {'.py','.md','.json'} and '__pycache__' not in path.parts:add('source/'+path.relative_to(S).as_posix(),path)
for n in ['controller-plan.json','campaign.json','campaign-complete.json','GPU-AUTHORIZATION.json','deployment-manifest.json','preflight.json','source-deployment.json','target-deployment.json','local-watch-intent.json','local-watch-events.jsonl','inference-proof.json','inference-plan.json','inference-start-intent.json','inference-job.json','inference-collected.json','inference-terminal-manifest.json','inference-terminal.zip','final-review-entry.json']:
 add('controller/'+n,C/n)
for ep in range(1,5):
 for suffix in ['job','start-intent','state','export-ack','export-verification','source-ack-copy']:
  n=f'epoch-{ep:03d}-{suffix}.json';add('controller/'+n,C/n)
for n in ['launch-intent.json','job.json','worker-ready.json','rendered-source-proof.json','intent.json','events.jsonl','complete.json']:
 add('controller/watch-recovery-1/'+n,C/'watch-recovery-1'/n)
for n in ['terminal.zip','manifest.json','checks.json']:add('training-terminal/'+n,C/'training-terminal'/n)
for path in (C/'quality').rglob('*'):
 if path.is_file():add('quality/'+path.relative_to(C/'quality').as_posix(),path)
for n in ['CPU-contract-result.json','inference-CPU-source.json']:add(n,B/n)
for path in (B/'matched-baseline').rglob('*'):
 if path.is_file() and path.suffix in {'.json','.jsonl','.log'}:add('matched-baseline/'+path.relative_to(B/'matched-baseline').as_posix(),path)
refs={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in paths.items()}
archive=D/'terminal.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for n,p in sorted(paths.items()):t.add(p,arcname=n,recursive=False)
with tarfile.open(archive) as t:
 assert set(t.getnames())==set(refs)
 for n,r in refs.items():
  b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
m=dict(archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),members=refs,scope='private clinical raw responses/predictions in inner zip; no held gold or model tensors; all4 model+optimizer checkpoints separately .100 durable',immutable_original_evidence=True)
(D/'archive-manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(archive=m['archive'],members=len(refs))))
