from pathlib import Path
B=Path(__file__).parent
S=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/patient_first')
for name in ['export_checkpoint_server.py','fetch_checkpoint.py']:
 s=(S/name).read_text()
 # All identity and namespace differences are immutable plan fields.
 lines=s.splitlines();out=[]
 for line in lines:
  if line.startswith(('SOURCE_RUN_ROOT =','PERSISTENT_ROOT =','TRAINER_SHA =','TRAIN_SHA =')):continue
  if line.startswith('RUN_FILES = '):continue
  if line.startswith("             'qwen35_relation_cost.py','patient_first.py','progress.json']"):continue
  out.append(line)
 s='\n'.join(out)+'\n'
 if name.startswith('export'):
  s=s.replace("p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()","p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()\n    plan=json.loads(a.plan.read_text());SOURCE_RUN_ROOT=Path(plan['output_dir']);TRAINER_SHA=plan['trainer_sha256'];TRAIN_SHA=plan['expanded_train_sha256'];RUN_FILES=plan['export_run_files']\n    assert time.time()<a.deadline")
  s=s.replace('import argparse, hashlib, json, re','import argparse, hashlib, json, re, time')
  s=s.replace("=='patientphex-qwen35-meta32-patient-first-lora-v1'","==plan['fingerprint_schema']")
  s=s.replace("sha(root/'train_patient_first.py')","sha(root/plan['export_trainer_basename'])")
  s=s.replace("print(json.dumps({'ready':True","assert time.time()<a.deadline\n    print(json.dumps({'ready':True")
 else:
  s=s.replace("p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()","p.add_argument('--destination',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()\n    plan=json.loads(a.plan.read_text());SOURCE_RUN_ROOT=Path(plan['output_dir']);PERSISTENT_ROOT=Path(plan['persistent_checkpoint_root']);TRAINER_SHA=plan['trainer_sha256'];TRAIN_SHA=plan['expanded_train_sha256'];RUN_FILES=plan['export_run_files']")
  s=s.replace("manifest['files']['_run/train_patient_first.py']['sha256']","manifest['files']['_run/'+plan['export_trainer_basename']]['sha256']")
  s=s.replace("make_durable_directory(root)\n    assert not", "assert os.getuid()==1006\n    make_durable_directory(root)\n    os.chmod(root,0o700)\n    assert not")
 # Describe actual new bounds rather than inherited historical180s comments.
 s=s.replace('patient-first epoch','Qwen35 fulltext epoch').replace('CPU180s','CPU600s maximum within original8400s deadline').replace('absolute180s export deadline','absolute min600/remaining-campaign export deadline')
 (B/name).write_text(s);compile(s,name,'exec')
print('exact H51 fsync/reopen export adapted, no transfer executed')
