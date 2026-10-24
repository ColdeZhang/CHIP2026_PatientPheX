"""Bind old epoch1 and recovered epochs into the unchanged paired F1 endpoint."""
from pathlib import Path
import hashlib,json,subprocess,sys
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(p.read_text())
def main():
    root=E/'formal-ack-recovery';old=E/'formal';plan=read(root/'run-plan.json')
    assert (root/'all-responses-complete.json').is_file()
    oldcp=read(old/'original/epoch-001/checkpoint.json');newcp=read(root/'original/epoch-002/checkpoint.json')
    assert oldcp['fingerprint']==newcp['fingerprint'] and oldcp['completed_epochs']==1 and newcp['completed_epochs']==2
    assert oldcp['global_step']==16 and newcp['global_step']==32
    history={'original_epoch1_checkpoint':old/'original/epoch-001/checkpoint.json',
        'original_epoch1_manifest':old/'original/epoch-001/run_manifest.json',
        'original_epoch1_persistent_ack':old/'original/epoch-001/export-ack.json',
        'original_epoch2_persistent_ack':root/'original/epoch-002/export-ack.json',
        'reverse_first_epoch1_persistent_ack':root/'reverse-first/epoch-001/export-ack.json',
        'reverse_first_epoch2_persistent_ack':root/'reverse-first/epoch-002/export-ack.json',
        'original_deadline_state_archive':old/'execution-state-before-ack-recovery.json',
        'minimal_recovery_plan':E/'ack-name-recovery-plan.json',
        'actual_ack_rename':root/'actual-ack-rename.json','both_responses_complete':root/'all-responses-complete.json'}
    for name,path in history.items():
        if name.endswith('persistent_ack'):assert read(path)['status']=='persistent_checkpoint_verified'
    helper=E/'freeze_evaluation_plan.py'
    assert ref(helper)==plan['references']['freeze_evaluation_plan']
    subprocess.run([sys.executable,str(helper),'--directory',str(root)],check=True)
    target=root/'paired-evaluation/run-plan.json';evaluation=read(target)
    evaluation['execution_history']={k:ref(v) for k,v in history.items()}
    evaluation['history_materializer_source_sha256']=sha(Path(__file__))
    evaluation['same_original_deadline_and_no_extra_updates']=True
    # Endpoint whitelist is unchanged. These additional no-label references
    # make the split original/recovery execution chain explicit for review.
    target.write_text(json.dumps(evaluation,indent=2)+'\n')
    print(json.dumps({'endpoint_plan':ref(target),'no_gold_access_or_quality_evaluation':True}))
if __name__=='__main__':main()
