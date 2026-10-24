from pathlib import Path
import json,hashlib,io,tarfile,gzip
T=Path(__file__).resolve().parent;C=T.parent;S=C/'formal-sources';D=T/'deployment';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
source=json.loads((S/'source-plan.json').read_text())['refs'];old=json.loads((C/'probe-a800-deployment/run-plan.json').read_text())
namespace='research/medgemma-relation-control/fullcontext-formal-preparation-v1'
files={Path(source[k]['path']).name:Path(source[k]['path']) for k in ['trainer','inference','formal_input_contract','proven_input_contract','append_constructor','legacy_constructor','full_infer','infer_input']}
files['export_checkpoint_server.py']=T/'export_checkpoint_server.py'
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as archive:
 for name,p in sorted(files.items()):
  info=tarfile.TarInfo(name);data=p.read_bytes();info.size=len(data);info.mode=0o600;info.mtime=0;archive.addfile(info,io.BytesIO(data))
bundle=D/'deployment-source-inputs.tar.gz';bundle.write_bytes(gzip.compress(stream.getvalue(),mtime=0));members={n:ref(p) for n,p in sorted(files.items())}
(D/'source-bundle.json').write_text(json.dumps(dict(bundle=ref(bundle),members=members),indent=2)+'\n')
reuse={**old['reused_source_files'],**old['remote_source_files']}
reuse['research/medgemma-relation-control/a800-preparation/meta.prompts.jsonl']={k:source['original_infer'][k] for k in ['bytes','sha256']}
plan=dict(namespace=namespace,mount_id=old['mount_id'],probe_output='/dev/shm/patientphex-medgemma15-1045-20261024/fullcontext-formal2ep',model=old['model'],reused_source_files=reuse,source_bundle_sha256=sha(D/'source-bundle.json'),remote_source_files={namespace+'/'+n:{k:r[k] for k in ['bytes','sha256']} for n,r in members.items()})
(D/'run-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(C/'probe-a800-deployment/deploy_once.py').read_text().replace('deployed_all9_SHA_verified_no_GPU_launch','deployed_all9_fullcontext_formal_SHA_verified_no_GPU_launch')
(D/'deploy_once.py').write_text(s)
print(json.dumps(dict(files=len(files),archive_bytes=bundle.stat().st_size,source_bundle_sha256=plan['source_bundle_sha256'])))
