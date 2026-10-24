from pathlib import Path
import sys,json,hashlib,tarfile,shlex
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup'];import remote_io as io
E=Path(__file__).resolve().parent.parent;T=E/'transfer-a800';W=Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry')
A=Path('/workspace/patientphex-runs/round2-20261023/research/medical-small-model-readiness/assets')
source_mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c';target_mount='69b991ee-183f-4add-af1a-579af92479c9'
assert not (T/'deployment-complete.json').exists()
files={n:E/n for n in ['train_medgemma32.py','medgemma_interface.py','qwen35_smoke.py','qwen35_relation_cost.py','probe_longest.py','infer_medgemma80.py','check_runtime_delta.py']}
files['fetch_assets.py']=T/'fetch_assets.py';files['expected-files.json']=A/'expected-files.json'
for n in ['cpu_complete.json','token-mask-audit.json','run_manifest.json']:files['reference/'+n]=E/'actual-cpu'/n
for n,r in json.loads((E/'data-locators.json').read_text()).items():files[n]=Path(r['path'])
refs={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in files.items()}
(T/'a800-preparation-members.json').write_text(json.dumps(refs,indent=2)+'\n')
archive=T/'a800-preparation.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
server='research/medgemma-relation-control/transfer-a800/serve_assets.py';remote_archive='research/medgemma-relation-control/transfer-a800/a800-preparation.tar.gz';prep='research/medgemma-relation-control/a800-preparation'
plan={'source_mount_id':source_mount,'target_mount_id':target_mount,'source_private_IP':'10.5.2.100','source_server':server,'destination':'/dev/shm/patientphex-medgemma15-1045-20261024/models/medgemma-1.5-4b-it-9bca96ca3e30','preparation':prep,
      'source_model':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30','source_expected':'research/medgemma-relation-control/preparation/expected-files.json',
      'source_server_sha256':hashlib.sha256((T/'serve_assets.py').read_bytes()).hexdigest(),'target_fetcher_sha256':refs['fetch_assets.py']['sha256'],'expected_files_sha256':'690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1',
      'actual_asset_complete_sha256':hashlib.sha256((A/'complete.json').read_bytes()).hexdigest(),'actual_header_sha256':hashlib.sha256((A/'actual-header-inventory.json').read_bytes()).hexdigest(),
      'total_assets_bytes':8639802098,'copy_budget_seconds':600,'TERM_grace_seconds':30,'no_GPU':True,'prior_controller_source_sha256':hashlib.sha256((W/'run_paired_ack_paths.py').read_bytes()).hexdigest(),
      'archive':{'local':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'files':refs}
F=T/'formal';F.mkdir(exist_ok=False);(F/'run-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
io.M=target_mount
check=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import shutil;root=Path(".");assert shutil.disk_usage(root).free>10*1024**2;assert shutil.disk_usage("/dev/shm").free>12*1024**3;shm=Path("/dev/shm/patientphex-medgemma15-1045-20261024");assert not shm.exists();shm.mkdir(mode=0o700);assert not Path('+repr(prep)+').exists();assert not Path('+repr(remote_archive)+').exists();print("A800 SHM/storage and unique namespace ready")'),'cwd':'.','timeout_seconds':10});assert check['exit_code']==0
(T/'target-preflight.json').write_text(json.dumps(check,indent=2)+'\n')
io.upload(str(archive),remote_archive)
code='''from pathlib import Path
import tarfile,hashlib,json
archive=Path(ARCHIVE);refs=REFS;assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
root=Path(PREP);assert not root.exists();root.mkdir(parents=True,mode=0o700)
with tarfile.open(archive,'r:gz') as t:
 assert set(t.getnames())==set(refs) and len(t.getmembers())==len(refs)
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  b=t.extractfile(m).read();r=refs[m.name];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];(root/m.name).parent.mkdir(parents=True,exist_ok=True);(root/m.name).write_bytes(b)
assert all((root/n).stat().st_size==r['bytes'] and hashlib.sha256((root/n).read_bytes()).hexdigest()==r['sha256'] for n,r in refs.items())
print(json.dumps(dict(status='all_source_data_reference_members_verified',files=len(refs),source_archive_sha256=SHA)))
'''.replace('ARCHIVE',repr(remote_archive)).replace('REFS',repr(refs)).replace('SHA',repr(plan['archive']['sha256'])).replace('PREP',repr(prep))
r=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':20});assert r['exit_code']==0;(T/'target-source-actual.json').write_text(json.dumps(r,indent=2)+'\n')
io.M=source_mount
r=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;assert not Path('+repr(server)+').exists();print("source namespace ready")'),'cwd':'.','timeout_seconds':10});assert r['exit_code']==0
io.upload(str(T/'serve_assets.py'),server)
r=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;p=Path('+repr(server)+');s=hashlib.sha256(p.read_bytes()).hexdigest();assert s=='+repr(plan['source_server_sha256'])+';print(json.dumps(dict(server_sha256=s)))'),'cwd':'.','timeout_seconds':10});assert r['exit_code']==0
(T/'deployment-complete.json').write_text(json.dumps({'target_source':json.loads((T/'target-source-actual.json').read_text()),'source_server':r,'no_jobs_started':True},indent=2)+'\n')
print('TRANSFER_PREPARATION_ONLY_COMPLETE')
