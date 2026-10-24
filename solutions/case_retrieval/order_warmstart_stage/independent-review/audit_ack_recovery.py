#!/usr/bin/env python3
"""Narrow acknowledgement filename and same-stage recovery source addendum."""
import ast,datetime,hashlib,json,pathlib
R=pathlib.Path('/workspace/patientphex-runs/round2-20261023')
E=R/'research/candidate-order-warmstart-entry'; O=R/'validation/candidate-order-warmstart-entry-review'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
ref=lambda p:dict(path=str(p),bytes=pathlib.Path(p).stat().st_size,sha256=sha(p))
def main():
    plan=read(E/'ack-name-recovery-plan.json'); records=list(plan['references'].values())+[ref(E/'ack-name-recovery-plan.json'),ref(E/'recover_ack_name.py')]
    out=O/'ack-recovery-snapshots';out.mkdir(exist_ok=True);snaps=[]
    for i,record in enumerate(records):
        assert ref(record['path'])==record
        data=pathlib.Path(record['path']).read_bytes(); p=out/f'{i:02d}-{pathlib.Path(record["path"]).name}'
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
        snaps.append(dict(original=record,snapshot=ref(p)))
    old=(E/'run_paired.py').read_text();new=(E/'run_paired_ack_paths.py').read_text()
    assert sha(E/'run_paired.py')=='586620a36721d138f29e9432474eb0a2af0e2510c0d37d277fc83feff9a208c5'
    assert sha(E/'run_paired_ack_paths.py')=='9afc2dc07fe5fe9b044c4dce6eaec9fbb69bd089af238ff34930bddf99a427f5'
    expected=old.replace("f'/epoch-{epoch:03d}-export-ack.json'","f'/export-ack-epoch-{epoch:03d}.json'").replace("output+'/epoch-001-export-ack.json'","output+'/export-ack-epoch-001.json'")
    assert new==expected
    assert sha(E/'train_order_warmstart.py')=='93301f6070f624672d8331b8fda7304089a2af44b95c83424513ef4895873ab4'
    fixture=read(E/'ack-name-cpu-contract.json')
    assert fixture['passed'] and len(fixture['checks'])==3 and fixture['no_MCP_GPU_model_gold']
    assert fixture['fixture_sha256']==sha(E/'test_ack_name_contract.py')
    assert fixture['corrected_controller_sha256']==sha(E/'run_paired_ack_paths.py')
    assert fixture['trainer_source_sha256']==sha(E/'train_order_warmstart.py')
    source=(E/'recover_ack_name.py').read_text();assert sha(E/'recover_ack_name.py')==plan['recovery_source_sha256']
    for name in ['recover_ack_name.py','run_paired_ack_paths.py','test_ack_name_contract.py']:
        compile(ast.parse((E/name).read_text()),name,'exec')
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='execute' for n in ast.walk(tree))
    assert "old/'execution-state-before-ack-recovery.json'" in source
    state=read(E/'formal/execution-state-before-ack-recovery.json')
    assert sha(E/'formal/execution-state-before-ack-recovery.json')=='a17919df6e4afded28e3a1cae97d89869c954f626db5bec000bb6c55db21663f'
    assert state['absolute_deadline_unix']==plan['same_original_deadline_unix']==1792808951.5793016
    assert state['absolute_deadline_unix']-state['started_unix']==3900
    assert (E/'formal-ack-recovery/run-plan.json').read_bytes()==(E/'formal/run-plan.json').read_bytes()
    assert sha(E/'formal/run-plan.json')=='3d0acfe84d09dc897efd1d96966ab294fbeec4fbaec46a22d711bb960e3af727'
    assert "run.deadline=state['absolute_deadline_unix']" in source and 'run.remaining()' in source
    assert "ack=run.train_segment('original',2);run.infer('original',ack)" in source
    assert "for epoch in [1,2]:ack=run.train_segment('reverse-first',epoch)" in source
    assert "run.save(root/'recovery-launched.json'" in source and source.index("run.save(root/'recovery-launched.json'")<source.index("run.gpu_empty('recovery-gpu-preflight')")
    assert "assert p.is_file() and not p.is_symlink() and not q.exists()" in source
    assert "p.rename(q);assert q.read_bytes()==before and not p.exists()" in source
    assert "matches=list(root.glob('epoch-*'));assert len(matches)==1 and matches[0].name=='epoch-001' and matches[0].is_dir()" in source
    assert "plan['remote_ack_sha256']" in source and "plan['epoch1_checkpoint_sha256']" in source
    ack=read(E/'formal/original/epoch-001/export-ack.json')
    assert sha(E/'formal/original/epoch-001/export-ack.json')==plan['remote_ack_sha256']
    assert ack['checkpoint_sha256']==plan['epoch1_checkpoint_sha256']==sha(E/'formal/original/epoch-001/checkpoint.json')
    assert ack['total_bytes']==sum(x['bytes'] for x in ack['files'].values())==519966182
    assert ack['completed_epochs']==1 and ack['global_step']==16 and ack['fsync_all_files']
    assert ack['persistent_mount_id']=='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
    failure=(E/'formal/original/train-002/stderr.log').read_text()
    assert "validate_checkpoint(output,checkpoint,fingerprint)" in failure and 'Resume requires latest complete epoch checkpoint' in failure
    report=dict(status='PASS_NARROW_ACK_FILENAME_RECOVERY_SOURCE',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      scope='Only two acknowledgement basename changes, original filename failure, immutable original stage state and recovery of the same stage/deadline; prior numerical training/evaluation contracts reused.',
      prior_entry=ref(O/'review.json'),prior_boundary_addendum=ref(O/'final-boundary-addendum.json'),
      recovery_plan=ref(E/'ack-name-recovery-plan.json'),corrected_controller=ref(E/'run_paired_ack_paths.py'),recovery_source=ref(E/'recover_ack_name.py'),trainer=ref(E/'train_order_warmstart.py'),
      evidence=dict(exact_two_basename_replacements=True,actual_frozen_guard_CPU_cases=fixture['checks'],fixture=ref(E/'ack-name-cpu-contract.json'),fixture_not_repeated=True,
        prior_stage_state_archive=ref(E/'formal/execution-state-before-ack-recovery.json'),original_plan_exact_byte_copy=True,original_deadline_unix=state['absolute_deadline_unix'],no_execute_or_new_budget=True,
        durable_epoch1_ack=ref(E/'formal/original/epoch-001/export-ack.json'),exported_bytes=519966182,checkpoint_sha256=ack['checkpoint_sha256'],failed_attempt=ref(E/'formal/original/train-002/stderr.log'),
        recovery_intent_before_remote_mutation=True,ack_actual_SHA_and_no_target_guard=True,rename_preserves_bytes=True,latest_checkpoint_directory_guard_preserved=True),
      decision='The failure is a filename enumeration collision before model load and epoch2 updates. The narrow correction is sufficient; resume original epoch2 from the already exported own-stage checkpoint within the unchanged absolute deadline. No quality result inferred.',
      limitations=['This is a source/receipt review, not a new GPU run or claim that recovery has completed.','Final paired evaluation must bind old formal original epoch1 receipts plus new recovery outputs explicitly.','Training source, optimizer/RNG resume contract, data, supervision, two-arm schedule and final paired freeze-before-gold policy are unchanged.'],
      snapshots=snaps,reviewer_source=ref(__file__),no_MCP_GPU_or_gold=True)
    p=O/'ack-filename-recovery-addendum.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(p)))
if __name__=='__main__':main()
