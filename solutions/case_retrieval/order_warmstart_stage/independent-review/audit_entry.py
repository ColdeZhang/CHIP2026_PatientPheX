#!/usr/bin/env python3
"""Final offline incremental entry review; no gold, model, remote calls or launch."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import math

R=Path('/workspace/patientphex-runs/round2-20261023')
E=R/'research/candidate-order-warmstart-entry'
O=R/'validation/candidate-order-warmstart-entry-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
ref=lambda p:dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))

def main():
    planpath=E/'formal/run-plan.json'
    assert sha(planpath)=='2dadd63a7e771e1a5a107b94424a656cbf1158669ef318fa7e690ed300790c31'
    plan=read(planpath)
    for record in plan['references'].values():assert ref(record['path'])==record
    assert ref(plan['evaluation_template']['path'])==plan['evaluation_template']
    template=read(plan['evaluation_template']['path'])
    for record in template['references'].values():
        assert Path(record['path'])!=Path(template['gold_path'])
        assert ref(record['path'])==record
    assert plan['controller_sha256']==sha(E/'run_paired.py')=='b78468eefab32a7067ae8c04a3fb6eb50d594373ecc238a609e16afa775d4acb'
    assert template['control_sha256']==sha(E/'evaluate_paired.py')=='563da3a19527b28e96e07418eb2fee063762301faf9492630292d61615e8a79d'
    assert template['trainer_sha256']==sha(E/'train_order_warmstart.py')=='93301f6070f624672d8331b8fda7304089a2af44b95c83424513ef4895873ab4'
    assert plan['cap_seconds']==3900 and plan['grace_seconds']==30
    assert plan['policy']['arms']==['original','reverse-first']
    assert plan['mount_id']=='69b991ee-183f-4add-af1a-579af92479c9'
    assert plan['persistent_mount_id']=='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
    assert plan['env']['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']=='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
    assert plan['env']['PYTHONPATH'].endswith('/a800-migration/runtime')
    assert plan['fetcher_sha256']==sha(E/'fetch_checkpoint.py')
    assert sum(plan['policy']['time_estimate_seconds'].values())==3900
    assert plan['model_source_sha256']==template['references']['model_source']['sha256']
    for local,remote in [('train_order_warmstart.py','train_order_warmstart.py'),('qwen35_smoke.py','qwen35_smoke.py'),('qwen35_relation_cost.py','qwen35_relation_cost.py'),('export_checkpoint_server.py','export_checkpoint_server.py')]:
        assert plan['remote_source_sha256'][plan['remote_root']+'/'+remote]==sha(E/local)
    prompts=[json.loads(x) for x in Path(template['references']['prompts']['path']).read_text().splitlines()]
    assert len(prompts)==80 and {x['row_id']:str(x['pmc_id']) for x in prompts}==plan['row_documents']
    assert len({str(x['pmc_id']) for x in prompts})==16
    assert all('answer' not in x and all(m['role'] in {'system','user'} for m in x['messages']) for x in prompts)
    cpu=read(E/'cpu-runtime-complete.json')
    for name,record in cpu['files'].items():
        p=E/'actual-cpu'/name;assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
    c=read(E/'actual-cpu/cpu_complete.json');m=read(E/'actual-cpu/run_manifest.json');fp=m['fingerprint']
    assert c['source_sha256']==template['trainer_sha256'] and c['manifest_sha256']==sha(E/'actual-cpu/run_manifest.json')
    assert c['cuda_initialized'] is False and c['not_training'] is True
    assert c['adapter_parameters']==43278336 and c['total_parameters']==8997081600
    assert c['rows_per_variant']==124 and c['documents']==32 and c['max_tokens']==5523
    assert c['full_tokens_by_variant']=={'O':402594,'R':402596} and c['supervised_tokens_per_variant']==7170
    coverage=read(E/'actual-cpu/meta-coverage.json')
    assert coverage['all_parameter_devices']==['meta']
    assert coverage['base_parameters']==8953803264 and coverage['adapter_parameters']==43278336
    causal=read(E/'actual-cpu/causal-loss-cpu-check.json')
    assert causal['actual_loss']==causal['manual_shifted_ce'] and causal['ignored_gradients_all_zero'] and causal['masked_logits_change_loss_unchanged']
    prior=read(R/'research/candidate-order-training-feasibility/row-inverse-and-token-audit.json')
    for variant,key in [('O','original'),('R','reversed')]:
        current=read(E/f'actual-cpu/token-mask-audit-{variant}.json')
        expected=[dict(row_id=t['original_row_id']+(':reverse-candidates' if variant=='R' else ''),full_tokens=t[key]['full_tokens'],prefix_tokens=t[key]['prefix_tokens'],supervised_tokens=t[key]['supervised_tokens']) for t in prior]
        assert current['rows']==expected
        assert current['encoded_sha256']==fp['encoded_variant_sha256'][variant]
    packages=fp['environment']['packages']
    assert packages['transformers']=='5.20.2' and packages['peft']=='0.21.2'
    assert all(packages[k] is None for k in ['fla-core','flash-linear-attention','causal-conv1d'])
    assert fp['warm_start']['adapter_files']['adapter_model.safetensors']=='4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'
    job=read(E/'cpu-runtime-job.json')
    assert job['env']['CUDA_VISIBLE_DEVICES']=='' and job['GPU'] is False
    assert job['cap_seconds']==120 and job['grace_seconds']==15
    assert read(E/'cpu-runtime-monitor.json')['running'] is False
    for p in [E/'run_paired.py',E/'evaluate_paired.py',E/'freeze_evaluation_plan.py',E/'prepare_run.py',E/'deploy_fetcher.py']:
        compile(ast.parse(p.read_text()),str(p),'exec')
    control=(E/'run_paired.py').read_text()
    assert "'model_source_sha256':p['model_source_sha256']" in control
    assert control.count("'start_process'")==1 and 'duration=self.remaining()' in control
    assert "'--kill-after=30s'" in control and 'self.deadline=started+3900' in control
    assert "for arm in ['original','reverse-first']" in control and 'for epoch in [1,2]' in control
    assert 'finally:' in control and "'stop_process'" in control
    # Run only pure receipt/retention guards from the final endpoint source.
    tree=ast.parse((E/'evaluate_paired.py').read_text())
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    ns=dict(Path=Path,PurePosixPath=PurePosixPath,MOUNT_ID=plan['persistent_mount_id'],math=math,METRICS=('Score','menF1','docF1','micF1','macF1'))
    exec(compile(ast.Module(body=[funcs[n] for n in ['require','verify_export','delta','keep_suggestion']],type_ignores=[]),'<frozen-final-pure-guards>','exec'),ns)
    files={n:'1'*64 for n in ['training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json']}
    cp=dict(files=files);refs=dict(checkpoint=dict(sha256='2'*64),training_manifest=dict(sha256='3'*64))
    inventory={n:dict(bytes=7,sha256=s) for n,s in files.items()}
    inventory['checkpoint.json']=dict(bytes=7,sha256='2'*64);inventory['_run/run_manifest.json']=dict(bytes=7,sha256='3'*64)
    receipt=dict(status='persistent_checkpoint_verified',persistent_mount_id=plan['persistent_mount_id'],fsync_all_files=True,completed_epochs=2,global_step=32,source_checkpoint_directory='/dev/shm/own/original/epoch-002',persistent_directory=plan['persistent_root']+'/original/epoch-002',checkpoint_sha256='2'*64,run_manifest_sha256='3'*64,files=inventory)
    ns['verify_export'](receipt,cp,refs);guardcases=['actual_relative_prefix_pass']
    for bad in ['/absolute/checkpoint','elsewhere/checkpoint',plan['persistent_root']+'/../escape',plan['persistent_root']]:
        r=copy.deepcopy(receipt);r['persistent_directory']=bad
        try:ns['verify_export'](r,cp,refs)
        except ValueError:guardcases.append('reject_'+bad)
        else:raise AssertionError(bad)
    r=copy.deepcopy(receipt);r['persistent_mount_id']='other'
    try:ns['verify_export'](r,cp,refs)
    except ValueError:guardcases.append('wrong_mount_rejected')
    else:raise AssertionError('wrong mount')
    assert ns['keep_suggestion']({'Score':.61,'macF1':.5},{'Score':.6,'macF1':.5})
    assert not ns['keep_suggestion']({'Score':.6,'macF1':.6},{'Score':.6,'macF1':.5})
    assert not ns['keep_suggestion']({'Score':.61,'macF1':.49},{'Score':.6,'macF1':.5})
    assert not (E/'formal/execution-state.json').exists()
    assert not (E/'fetcher-deployment.json').exists()
    references=[planpath,E/'evaluation-template.json',E/'run_paired.py',E/'evaluate_paired.py',E/'freeze_evaluation_plan.py',E/'prepare_run.py',E/'deploy_fetcher.py',E/'cpu-runtime-complete.json',E/'cpu-runtime-job.json',E/'cpu-runtime-monitor.json',E/'deployment-actual.json',E/'evaluate_paired_interface.md',O/'data-and-schedule.json',O/'trainer-and-ack-source.json',Path(__file__)]
    references += [E/'actual-cpu'/name for name in cpu['files']]
    result=dict(status='PASS_MATCHED_WARMSTART_STAGE_COMPLETE_ENTRY_OFFLINE_NO_LAUNCH',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Source/data/supervision/matched-budget/durable-export/paired-evaluation entry; no new training, inference or quality result.',
        decision='No source-contract blocker remains. Root MCP pause still applies; entry PASS does not assert deployment, current free GPU, training or remote backup.',
        formal_plan_sha256=sha(planpath),endpoint_source_sha256=sha(E/'evaluate_paired.py'),
        data=dict(train_documents=32,rows=124,patient_concept_pairs=2506,positives=594,excluded_documents=32,F1_prompts=80,F1_pairs=1557,F1_entities=1189,reverse_semantics_and_exposure_independently_rebuilt=True),
        training=dict(warm_adapter_sha256='4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff',old_epochs=2,new_epochs=2,arms={'original':['O','O'],'reverse-first':['R','O']},optimizer_calls_each=32,examples_each=248,supervised_tokens_each=14340,total_tokens={'original':805188,'reverse-first':805190},fresh_Adam_scheduler_and_RNG=True,exact_old_optimizer_resume=False,same_sample_order_and_start_seed=True,unchanged_old_full_JSON_loss=True),
        actual_CPU=dict(job_id=job['job']['id'],started_at=job['job']['started_at'],native_seconds=c['elapsed_seconds'],CUDA_initialized=False,all_parameter_devices=['meta'],base_parameters=8953803264,adapter_parameters=43278336,total_parameters=8997081600,per_row_native_token_lengths_and_supervision_match_both_frozen_variants=True,actual_warm_adapter_SHA_verified_by_CPU_source=True,actual_model_weights_loaded=False),
        controller=dict(shared_seconds=3900,TERM_grace_seconds=30,sequential_one_A800=True,training_segments=4,full_persistent_exports=4,original80_inferences=2,complete_epoch_export_and_actual_readback_before_resume=True,checkpoint_contains=['adapter','Adam','scheduler','Python RNG','NumPy RNG','Torch RNG','CUDA RNG','run manifest','source','progress'],mutable_start_not_retried=True,HTTP_server_cleanup_in_finally=True,remaining_budget_checked_before_every_new_start=True,read_only_endpoint_checks_after_deadline_do_not_start_new_jobs=True,source_and_empty_GPU_rechecked_before_start=True),
        endpoint=dict(only_final_epoch2=True,both_complete80_and_fullpreds_before_first_gold=True,parser_and_v2_unchanged=True,cached_old_metrics_only=True,one_v2_per_new_arm=True,old_nonrelation_fields_and1189_entities_exact=True,no_union_threshold_search_migration_or_official=True,retention='Score strictly increases with nondecreasing macF1; compare R/O to O/O and both to cached old32; advisory only.'),
        corrections_before_launch=['Aligned evaluator export receipt to fixed .100 mount plus safe relative prefix.','Added model_source_sha256 to controller inference plan.'],
        reviewer_final_export_guard_cases=guardcases,
        limitations=['Entry readiness is not a completed training result, validated durable production export, or Score >= .8.','Root paused new MCP actions after global outage; tiny .100 fetcher deployment, fresh GPU/storage preflight, source push/backup and all new-stage execution remain pending.','Known F1 is repeatedly observed development; comparison to old also changes optimization stage and training amount. O/O vs R/O is the matched order contrast.','Old preparation fresh-base/forbid-continuation flags describe the historical baseline; the explicitly selected new stage reuses its data provenance and adapter values, with fresh optimizer/RNG.','Same initial seed does not imply identical later dropout masks with changed token lengths.','Runtime/token/meta and owner transport fixtures were verified from source-bound actual receipts, not rerun by reviewer; full warm-model forward and complete-checkpoint restore are not asserted.','Global execution budget includes training/export/inference. Current-invocation trainer timings must not be mistaken for full-stage elapsed.','No server backup is claimed while MCP is unavailable.'],
        references=[ref(p) for p in references],no_gold_read=True,no_model_or_tokenizer_import=True,no_network_or_GPU=True)
    (O/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],report=ref(O/'review.json'))))

if __name__=='__main__':main()
