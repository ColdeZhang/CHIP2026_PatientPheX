"""One private F0 terminal bundle; no checkpoint duplication or held gold."""
from pathlib import Path
import argparse,hashlib,json,os,tarfile
S=Path(__file__).resolve().parent;B=Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-v1');I=B/'F0-full58-v1';C=B/'F0-readiness-v1';D=B/'F0-terminal-backup'
def main():
 assert json.loads((I/'quality/result.json').read_text())['status']=='COMPLETE_FIXED_RESPONSE_CODE_F0';assert (I/'terminal-manifest.json').is_file();D.mkdir(exist_ok=False);paths={}
 def add(n,p):
  p=Path(p);assert p.is_file() and not p.is_symlink() and n not in paths and p.name!='access-token';paths[n]=p
 for p in S.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json','.md','.txt','.diff'] and '__pycache__' not in p.parts:add('source/'+p.relative_to(S).as_posix(),p)
 for n in ['plan.json','cpu-plan.json','structural-result.json','row-contract.json','ancestry.json','held.original.jsonl.gz','held.response-code.jsonl','native-terminal-manifest.json','native-terminal.zip','start-intent.json','start-tool.json','deployment-result.json']:
  add('cpu/'+n,C/n)
 for p in C.glob('recovery-*'):
  if p.is_file() and p.suffix=='.json':add('cpu/'+p.name,p)
 for p in (C/'preparation-history').glob('*'):add('cpu/preparation-history/'+p.name,p)
 for n in ['plan.json','entry.json','deployment-manifest.json','start-intent.json','job.json','observed-job.json','start-tool.json','terminal-manifest.json','terminal.zip','collect-intent.json','collect-manifest-tool.json']:
  if (I/n).is_file():add('inference/'+n,I/n)
 for p in I.glob('deployment-recovery-*.json'):add('inference/'+p.name,p)
 for folder in I.glob('collect-recovery-*'):
  for p in folder.glob('*.json'):
   if not p.name.startswith('part-'):add('inference/'+folder.name+'/'+p.name,p)
 for p in (I/'quality').rglob('*'):
  if p.is_file():add('quality/'+p.relative_to(I/'quality').as_posix(),p)
 add('F0-final-review-entry.json',B/'F0-final-review-entry.json')
 refs={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in paths.items()};archive=D/'terminal.tar.gz'
 with tarfile.open(archive,'x:gz') as t:
  for n,p in sorted(paths.items()):t.add(p,arcname=n,recursive=False)
 with archive.open('rb') as f:os.fsync(f.fileno())
 with tarfile.open(archive) as t:
  assert set(t.getnames())==set(refs)
  for n,r in refs.items():b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 m=dict(archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),members=refs,private_clinical_responses=True,no_held_gold=True,no_model_optimizer_arrays=True,capability_excluded=True)
 with (D/'archive-manifest.json').open('x') as f:json.dump(m,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(archive=m['archive'],members=len(refs))))
if __name__=='__main__':main()
