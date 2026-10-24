"""Bind exactly the completed new two-epoch adapter; never starts GPU or reads gold."""
from pathlib import Path
import datetime,hashlib,io,json,os,tarfile
S=Path(__file__).parent
R=Path('/workspace/patientphex-runs/round2-20261023')
Q=R/'research/qwen35-fulltext32-indexed-formal-v1/controller'
D=R/'research/qwen35-fulltext32-indexed-F1-full80-v1'
OLD=R/'research/qwen35-sft32-full-report-feasibility/full80-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
complete=json.loads((Q/'campaign-complete.json').read_text())
assert complete['status']=='TWO_COMPLETE_EPOCHS_32_STEPS_DURABLE_ACKS_VERIFIED'
cp=json.loads((Q/'controller-plan.json').read_text())
assert complete['campaign']['plan_sha256']==sha(Q/'controller-plan.json')
acks=[]
for e in [1,2]:
 p=Q/('epoch-%03d-export-ack.json'%e);a=json.loads(p.read_text())
 assert a['status']=='persistent_checkpoint_verified' and a['completed_epochs']==e and a['global_step']==16*e
 assert a['fsync_all_files'] and a['reopened_final_filename_SHA_all_files'] and a['destination_directories_fsynced']
 assert sha(p)==complete['epoch%d_ack_sha256'%e]
 verify=json.loads((Q/('epoch-%03d-export-verification.json'%e)).read_text());assert verify['all_files_reopened_SHA'] and verify['ack']==a
 source_copy=json.loads((Q/('epoch-%03d-source-ack-copy.json'%e)).read_text());assert source_copy['sha256']==sha(p)
 acks.append(a)
ready=json.loads((Q/'epoch-002-server-ready.json').read_text());assert ready['manifest']['files']==acks[1]['files']
assert ready['manifest']['completed_epochs']==2 and ready['manifest']['global_step']==32
assert ready['manifest']['trainer_sha256']==sha(S.parent/'train.py')
plan=json.loads((S/'plan-template.json').read_text());assert plan['cached_original_indices']==[] and plan['fresh_rows']==80
plan.update(adapter=cp['output_dir']+'/epoch-002/adapter',adapter_files={n:acks[1]['files']['adapter/'+n]['sha256'] for n in ['adapter_config.json','adapter_model.safetensors']},status='TWO_ACTUAL_DURABLE_EPOCH_ACKS_BOUND_READY_NO_GPU',frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan['actual_training_provenance']={'controller_plan':ref(Q/'controller-plan.json'),'trainer':ref(S.parent/'train.py'),'campaign_complete':ref(Q/'campaign-complete.json'),'epoch1_persistent_ACK':ref(Q/'epoch-001-export-ack.json'),'epoch2_persistent_ACK':ref(Q/'epoch-002-export-ack.json'),'epoch2_full_files':acks[1]['files'],'old_probe_cache_reuse':False}
assert not D.exists();D.mkdir()
for n,r in plan['files'].items():
 src=S/n if (S/n).is_file() else OLD/n
 assert src.stat().st_size==r['bytes'] and sha(src)==r['sha256'],n
 (D/n).write_bytes(src.read_bytes())
for n,src in [('training-campaign-complete.json',Q/'campaign-complete.json'),('training-epoch2-persistent-ACK.json',Q/'epoch-002-export-ack.json')]:
 (D/n).write_bytes(src.read_bytes());plan['files'][n]={'bytes':src.stat().st_size,'sha256':sha(src)}
write(D/'plan.json',plan)
(D/'operations.py').write_bytes((S/'operations.py').read_bytes())
files={n:ref(D/n) for n in list(plan['files'])+['plan.json']}
with tarfile.open(D/'deployment.tar.gz','x:gz') as t:
 for n in files:t.add(D/n,arcname=n,recursive=False)
with tarfile.open(D/'deployment.tar.gz') as t:
 assert set(t.getnames())==set(files)
 for n,r in files.items():b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
write(D/'deployment-manifest.json',dict(archive=ref(D/'deployment.tar.gz'),files=files))
write(D/'entry.json',dict(status='BOUND_NEW_ADAPTER_COMPLETE80_FRESH_NOT_STARTED',plan=ref(D/'plan.json'),source=ref(D/'infer_full80.py'),archive=ref(D/'deployment.tar.gz'),training_complete=ref(Q/'campaign-complete.json'),budget_seconds=1200,TERM_grace_seconds=30,new_model_calls=80,old_cache_reuse=0,no_quality=True))
print(json.dumps(dict(status='BOUND_NO_GPU_NO_GOLD',entry=ref(D/'entry.json'))))
