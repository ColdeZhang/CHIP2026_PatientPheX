"""Continue exported original epoch1 after a pre-model filename-guard failure.

The original controller/plan/state/failure remain immutable. Only two controller
ack names change. Trainer, checkpoint, optimizer, data and deadline are unchanged.
"""
from pathlib import Path
import datetime,hashlib,importlib.util,json,time
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))

def main():
    plan=json.loads((E/'ack-name-recovery-plan.json').read_text())
    assert sha(__file__)==plan['recovery_source_sha256']
    for r in plan['references'].values():assert ref(r['path'])==r
    old=E/'formal';root=E/'formal-ack-recovery'
    assert not (root/'recovery-launched.json').exists(),'Inspect existing recovery jobs, never duplicate'
    assert (root/'run-plan.json').read_bytes()==(old/'run-plan.json').read_bytes()
    spec=importlib.util.spec_from_file_location('corrected_controller',E/'run_paired_ack_paths.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=module.Run(root);state=json.loads((old/'execution-state-before-ack-recovery.json').read_text());run.deadline=state['absolute_deadline_unix']
    assert run.deadline==1792808951.5793016 and run.plan['cap_seconds']==3900
    run.remaining()
    run.save(root/'recovery-launched.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_state':ref(old/'execution-state-before-ack-recovery.json'),'recovery_plan':ref(E/'ack-name-recovery-plan.json'),
        'absolute_deadline_unix':run.deadline,'only_resume_original_epoch2_then_original80_and_reverse_first_2epochs80':True})
    run.gpu_empty('recovery-gpu-preflight')
    output=run.plan['shm_root']+'/original';oldack=output+'/epoch-001-export-ack.json';newack=output+'/export-ack-epoch-001.json'
    code="from pathlib import Path;import hashlib,json;p=Path("+repr(oldack)+");q=Path("+repr(newack)+");assert p.is_file() and not p.is_symlink() and not q.exists();assert hashlib.sha256(p.read_bytes()).hexdigest()=="+repr(plan['remote_ack_sha256'])+";before=p.read_bytes();p.rename(q);assert q.read_bytes()==before and not p.exists();root=q.parent;matches=list(root.glob('epoch-*'));assert len(matches)==1 and matches[0].name=='epoch-001' and matches[0].is_dir();cp=matches[0]/'checkpoint.json';assert hashlib.sha256(cp.read_bytes()).hexdigest()=="+repr(plan['epoch1_checkpoint_sha256'])+";print(json.dumps(dict(old_ack_removed=True,new_ack=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),epoch_glob=[x.name for x in matches])))"
    run.save(root/'actual-ack-rename.json',run.command(run.plan['mount_id'],code))
    ack=run.train_segment('original',2);run.infer('original',ack)
    for epoch in [1,2]:ack=run.train_segment('reverse-first',epoch)
    run.infer('reverse-first',ack)
    run.save(root/'all-responses-complete.json',{'status':'both_full_newstage_arms_and_durable_checkpoints_complete_before_quality',
        'seconds_from_original_start':time.time()-state['started_unix'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_deadline_unix':run.deadline,'original_epoch1_manifest':ref(old/'original/epoch-001/run_manifest.json'),
        'original_epoch1_persistent_ack':ref(old/'original/epoch-001/export-ack.json'),'recovery_plan':ref(E/'ack-name-recovery-plan.json'),'no_quality_evaluation':True})
    print('BOTH_ARMS_COMPLETE_AFTER_ACK_NAME_RECOVERY_NO_QUALITY_YET',flush=True)

if __name__=='__main__':main()
