"""Independent CPU attempt on .100, reusing already verified transport bytes.

This does not restart the A100 task: no A100 CPU start was requested. Only local
source/input extraction and a distinct runtime plan are prepared here.
"""
from pathlib import Path
import argparse,hashlib,json,os,shlex,sys,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
S=Path(__file__).resolve().parent
B=Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-readiness-v1')
D=B/'native100-attempt2'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','deploy']);a=ap.parse_args()
 if a.action=='prepare':
  D.mkdir(exist_ok=False)
  original=json.loads((B/'plan.json').read_text());old=json.loads(Path('/workspace/patientphex-runs/round2-20261023/research/nonthinking-sampled-consensus-full64-20261024/gpu-plan.json').read_text())
  assert not (B/'start-intent.json').exists()
  p={**original,'mount_id':old['mount_id'],'remote':'patient-response-code/full64-cpu-native100-attempt2','python':old['python'],'model':old['model'],'env':{**old['env'],'CUDA_VISIBLE_DEVICES':''},'runtime':{k:old['versions'][k] for k in ['torch','transformers','tokenizers']},'attempt_scope':'Independent CPU on .100 due A100 deployment LUNA; original A100 CPU never started. Not a restart or claimed same-runtime execution.','A100_plan_sha256':sha((B/'plan.json').read_bytes())}
  save(D/'plan.json',p);save(D/'cpu-plan.json',p)
  (D/'structural-result.json').write_bytes((B/'structural-result.json').read_bytes())
  save(D/'preparation.json',dict(status='INDEPENDENT_NATIVE100_CPU_PREPARED_NO_EXECUTION',original_plan_sha256=p['A100_plan_sha256'],runtime=p['runtime'],CPU_seconds=120,TERM_grace=5,no_GPU_or_model=True))
  print(json.dumps(dict(directory=str(D),plan_sha256=sha((D/'plan.json').read_bytes()))));return
 p=json.loads((D/'plan.json').read_text());m=json.loads((B/'deployment-manifest.json').read_text());root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+p['remote'];archive='/home/zhangyuheng/projects/CHIP2026_PatientPheX/patient-response-code/full64-cpu-source-bridge-v1/deployment.tar.gz'
 assert not (D/'deployment-intent.json').exists()
 params=dict(root=root,archive=archive,manifest=m,plan=(D/'plan.json').read_text())
 code='''from pathlib import Path
import hashlib,json,os,tarfile
r=Path(P['root']);assert os.getuid()==1006 and not r.exists();r.mkdir(parents=True,mode=0o700)
a=Path(P['archive']);b=a.read_bytes();m=P['manifest'];assert len(b)==m['archive']['bytes'] and hashlib.sha256(b).hexdigest()==m['archive']['sha256']
with tarfile.open(a) as t:
 assert set(t.getnames())==set(m['files'])
 for name,ref in m['files'].items():
  b=t.extractfile(name).read();assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
  if name in ['plan.json','cpu-plan.json']:b=P['plan'].encode()
  with (r/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  if name.endswith('.py'):compile(b,name,'exec')
fd=os.open(str(r),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status='ACTUAL_ALL_SOURCE_INPUT_TRAINING_ACK_SHA_COMPILE_FSYNC',plan_sha256=hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),files=len(m['files']),uid=os.getuid(),original_data_source_exact=True,no_model_loaded=True)))
'''
 save(D/'deployment-intent.json',dict(at_unix=time.time(),archive_sha256=m['archive']['sha256'],plan_sha256=sha((D/'plan.json').read_bytes()),new_namespace=root,no_original_CPU_start=True))
 client=Client();client.initialize();r=client.request('tools/call',dict(name='run_command',arguments=dict(mount_id=p['mount_id'],command='python3 -c '+shlex.quote('P='+repr(params)+'\n'+code),timeout_seconds=30)))
 save(D/'deployment-tool.json',r);assert not r.get('isError'),str(r)
 x=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert x['exit_code']==0,x.get('stderr');value=json.loads(x['stdout']);assert value['plan_sha256']==sha((D/'plan.json').read_bytes());save(D/'deployment-result.json',value);print(json.dumps(value))
if __name__=='__main__':main()
