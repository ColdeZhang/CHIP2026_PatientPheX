"""Prepare a new local ignored deployment bundle only; never call MCP or launch."""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,tarfile
D=Path(__file__).resolve().parent;OLD=D.parent/'stress_probe'
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=D/'.prepared');a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 old=json.loads((OLD/'source-manifest.json').read_text());plan=json.loads((OLD/'plan.json').read_text())
 assert ref(OLD/'probe.py')['sha256']=='165e9bfcb075d1ec918df4318beabee086957b81aa6d3eb7e4e423d11e6018ee'
 newcpu=json.loads((D/'CPU-result.json').read_text());assert newcpu['status']=='PASS_INDEXED_HEAD_HELPER_CPU_AND_NEGATIVE_CONTRACT_NO_GPU' and not newcpu['CUDA_initialized']
 assert newcpu['source_files']['indexed_head.py']=={k:ref(D/'indexed_head.py')[k] for k in ['bytes','sha256']}
 sourcefiles={name:Path(old['files'][name]['path']) for name in plan['files'] if name not in ['probe.py','launch.py']}
 sourcefiles.update({name:D/name for name in ['probe.py','launch.py','indexed_head.py','cpu_checks.py']})
 sourcefiles['head-CPU-result.json']=D/'CPU-result.json'
 for name,p in sourcefiles.items():
  if name in plan['files'] and name not in ['probe.py','launch.py']:
   assert {k:ref(p)[k] for k in ['bytes','sha256']}==plan['files'][name]
  (a.output/name).write_bytes(p.read_bytes())
 tree=ast.parse((D/'indexed_head.py').read_text());tol=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOLERANCES' for t in x.targets))
 plan.update(status='PREPARED_CPU_ONLY_INDEXED_HEAD_PROBE_NOT_LAUNCHED',remote='ontology-round2/qwen35-fulltext32-head-index-probe-v1',head_contract_tolerances=tol,
  GPU_execution_performed=False,formal_training_changed=False,full_training_authorized=False,
  source_parent_probe=ref(OLD/'probe.py'),source_parent_plan=ref(OLD/'plan.json'),
  local_head_contract='One actual shortest-of-existing12 backbone capture; same frozen hidden/head native A/B/indexed C; restore original Python NumPy Torch CUDA RNG before pressure; statistics before gate',
  pressure_changes='Only Tensor logits_to_keep and external original causal CE. Original12 rows/one update/three accum+max/seed/LoRA/precision unchanged. Additional bounded local-head diagnostic and explicit telemetry only.',
  memory_scope='Local contract peaks saved; bookkeeping peak counters reset before pressure, allocator cache untouched; report both phases and combined maxima.',
  discarded_state=True,new_cpu_contract=ref(D/'CPU-result.json'),local_feasibility_ref='/workspace/patientphex-runs/round2-20261023/research/head-projection-feasibility/result.json')
 plan['files']={name:{k:ref(a.output/name)[k] for k in ['bytes','sha256']} for name in sourcefiles};plan['remote_files']=plan['files'].copy()
 save(D/'plan.json',plan);(a.output/'plan.json').write_bytes((D/'plan.json').read_bytes())
 archive=a.output/'deployment.tar.gz'
 with archive.open('wb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
   with tarfile.open(fileobj=gz,mode='w') as tar:
    for name in sorted(list(sourcefiles)+['plan.json']):
     data=(a.output/name).read_bytes();item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;item.mtime=0;tar.addfile(item,io.BytesIO(data))
 with tarfile.open(archive,'r:gz') as tar:
  assert len(tar.getmembers())==len(sourcefiles)+1
  for member in tar.getmembers():assert member.isfile() and tar.extractfile(member).read()==(a.output/member.name).read_bytes()
 entry={'status':'LOCAL_BUNDLE_READY_NOT_DEPLOYED_NOT_LAUNCHED','mount_id':plan['mount_id'],'remote':plan['remote'],'archive':ref(archive),'plan':ref(D/'plan.json'),'launch_argv':[plan['python'],'-u','launch.py'],'env':plan['env'],'cap_seconds':420,'term_grace_seconds':30,'official_B_priority':True,'requires_parent_decision_before_GPU':True,'no_full_formal_authorization':True}
 save(D/'entry.json',entry)
 manifest={'status':entry['status'],'new_source_directory':str(D),'original_probe':ref(OLD/'probe.py'),'original_formal_train':ref(D.parent/'formal_train/train.py'),'original_formal_controller':ref(D.parent/'formal_train/controller/epoch_driver.py'),'new_sources':{p.name:ref(p) for p in sorted(D.glob('*.py'))},'plan':ref(D/'plan.json'),'CPU_result':ref(D/'CPU-result.json'),'entry':ref(D/'entry.json'),'dependencies':{name:ref(p) for name,p in sourcefiles.items()},'package_member_count':len(sourcefiles)+1,'no_new_model_or_data':True}
 save(D/'source-manifest.json',manifest)
 print(json.dumps({'bundle':ref(archive),'member_count':len(sourcefiles)+1,'plan_sha256':ref(D/'plan.json')['sha256'],'source_manifest_sha256':ref(D/'source-manifest.json')['sha256'],'GPU_started':False}))
if __name__=='__main__':main()
