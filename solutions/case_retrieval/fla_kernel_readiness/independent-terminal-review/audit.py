from pathlib import Path
import ast, datetime, hashlib, json, math, statistics

R=Path('/workspace/patientphex-runs/round2-20261023');B=R/'research/fla-kernel-readiness';A=B/'attempt2';N=A/'gpu-numeric-attempt2';O=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ref=lambda p:dict(path=str(Path(p)),bytes=Path(p).stat().st_size,sha256=sha(p))
plan=read(B/'probe-plan.json');manifest=read(B/'probe-source-manifest.json');repair=read(B/'attempt2-repair-manifest.json')
assert sha(B/'probe-plan.json')==repair['original_plan_sha256']=='f16cc09a5cfc8b0dddd8555e93e39a1ce56dd8538f3b51b6ca2c99f1c83267d8'
assert sha(B/'gpu_probe.py')==repair['original_probe_sha256']=='b8aca0d289e70623d57e537e23751017c2b6cd260a6ab2cba6a740997bf1e01e'
for n,v in manifest['files'].items():assert sha(B/n)==v['sha256'] and (B/n).stat().st_size==v['bytes']
for n,h in repair['new_sources'].items():assert sha(B/n)==h
cpu=read(B/'attempt2-cache-cpu-contract.json');assert cpu['passed'] and not cpu['cuda_initialized']
assert cpu['actual_dynamic_cache_layers']==32 and cpu['actual_field_container']=='dict[int,Tensor|None]'
assert all(cpu['fail_closed_cases'].values()) and cpu['all_layer_keys_shape_dtype_preserved'] and cpu['snapshot_detached_copy']
wrapper=(B/'gpu_probe_attempt2.py').read_text();tree=ast.parse(wrapper)
assigns=[n.targets[0].attr for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='original']
assert sorted(assigns)==['compare_states','states']
capture=(B/'cache_snapshot_attempt2.py').read_text()
assert 'value.detach().cpu().clone()' in capture and "if not result:raise ValueError" in capture
assert 'if a.layout!=b.layout:raise AssertionError' in capture
gate=read(B/'gpu-import-gate.json');assert gate['passed'] and not gate['cuda_initialized_before'] and gate['cuda_initialized_after'] and gate['elapsed_seconds']<20
assert gate['modeling_sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
binding=read(N/'actual-bindings.json')
for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:
    assert any(x['module'].startswith('fla.ops.gated_delta_rule.') for x in binding['fused'][name])
    assert all(x['module']=='transformers.models.qwen3_5.modeling_qwen3_5' for x in binding['native'][name])
for v in binding['conv'].values():assert not any(x['module'].startswith('causal_conv1d') for x in v)
collection=read(A/'collection.json')
for name,v in collection['files'].items():assert sha(A/name)==v['sha256'] and (A/name).stat().st_size==v['bytes']
actual=read(A/'terminal.json');metrics=read(N/'metrics.json');timing=read(N/'timings.json');th=plan['thresholds']
assert actual['failed_comparison']=='synthetic-0/logits' and actual['metrics']==metrics['synthetic-0/logits']
assert plan['prompt_lengths']==[128,255,256,513] and plan['decode_steps']==8
assert th['logits_relative_l2']==.01 and th['logits_max_abs']==.25 and th['logits_mean_kl']==.001
assert th['gradient_relative_l2']==th['cache_relative_l2']==.02
for name in ['block_output','block_decode']:
    v=metrics[name];assert v['passed'] and v['relative_l2']<.01
    assert math.isclose(v['relative_l2'],v['error_l2']/v['reference_l2'],rel_tol=1e-12)
grads={k:v for k,v in metrics.items() if k.startswith('block_grad/')}
assert len(grads)==9 and all(v['passed'] and v['relative_l2']<=.02 for v in grads.values())
cache=[metrics['block_cache/'+str(i)] for i in range(9)]
assert len({v['structure_sha256'] for v in cache})==1
for c in cache:
    assert c['state_tensor_count']==2
    assert set(c['tensor_metrics'])=={'0/conv_states/int:0','0/recurrent_states/int:0'}
    assert all(v['passed'] and v['relative_l2']<=.02 for v in c['tensor_metrics'].values())
failed=metrics['synthetic-0/logits']
assert failed['shape']==[1,128,248320] and failed['passed'] is False
assert math.isclose(failed['relative_l2'],failed['error_l2']/failed['reference_l2'],rel_tol=1e-12)
assert failed['relative_l2']>.01 and failed['max_abs']>.25 and failed['mean_kl']<.001
assert not any(k.startswith(('synthetic-1','synthetic-2','synthetic-3','lora/')) for k in metrics)
assert not (N/'summary.json').exists()
assert all(len(timing[k])==1 for k in ['prefill/128/native','prefill/128/fused','decode/128/native','decode/128/fused'])
assert not any(k.startswith('lora_step') for k in timing)
assert len(timing['block_native_warm_forward'])==len(timing['block_fused_warm_forward'])==3
loading=read(N/'loading-info.json');assert not any(loading.values())
assert 'AssertionError: synthetic-0/logits' in (A/'attempt2.stderr.log').read_text()
firststderr=(B/'attempt1/numeric.stderr.log').read_text()
assert "AttributeError: 'int' object has no attribute 'detach'" in firststderr
first_timing=read(B/'attempt1/gpu-numeric-v1/timings.json')
assert list(first_timing)==['block_native_cold_fb']
launch=read(A/'attempt2-launch-record.json');job=read(B/'gpu-attempt2-job.json')['structuredContent'];terminal=read(A/'process-terminal.json');release=read(A/'gpu-release.json')
assert job['id']==terminal['id']=='proc_68ecdbc1-61f2-41dd-94e7-0c049a3ef79c' and not terminal['running']
assert launch['deadline_unix']==repair['original_numeric_deadline_unix']==1792804070.1162088
assert launch['kill_deadline_unix']==launch['deadline_unix']+30
assert math.isclose(launch['remaining_seconds'],launch['deadline_unix']-launch['started_unix'],abs_tol=.001)
assert launch['remaining_seconds']<600 and launch['repair_backup_commit']=='5257677ef5d648ba492fb5e721bd6b99b4630707'
assert launch['argv'][3]=='385.018s' and launch['fresh_gpu_processes']==''
assert release['allocated_process_absent'] and release['allocated_gpu_zero_memory']
assert release['allocated_uuid']=='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539' and release['allocated_uuid'] not in release['compute_apps']
release_time=datetime.datetime.fromisoformat(release['checked_utc']);assert release_time.timestamp()<launch['deadline_unix']
assert actual['remote_exit_code'] is None and not actual['new_quality_evaluation'] and not actual['third_attempt']

report=dict(status='PASS_AUDIT_OF_FAILED_FLA_FULL_MODEL_NUMERIC_CONTRACT_NO_EXPANSION',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Incremental frozen code, instrumentation-repair, binding, measured contract and resource-terminal review. No new GPU, logits generation or competition evaluation.',
    dispatch=dict(import_gate_seconds=gate['elapsed_seconds'],both_GDN_actual_FLA=True,both_conv_reference=True),
    attempt1=dict(failure='cache instrumentation iterated integer dictionary key as Tensor',fused_numeric_compared=False,quality_failure=False),
    repair=dict(only_cache_sampler_and_structure_comparison_changed=True,original_source_plan_thresholds_exact=True,actual_CPU_mixed_container_and_fail_closed_contract_receipts_pass=True,original_absolute_deadline_preserved=True),
    passed_scope=dict(block_forward_relative_L2=metrics['block_output']['relative_l2'],block_decode_relative_L2=metrics['block_decode']['relative_l2'],gradient_parameters=9,max_gradient_relative_L2=max(v['relative_l2'] for v in grads.values()),cache_snapshots=9,state_tensors_per_snapshot=2),
    failed_scope=dict(first_full_model_prompt_tokens=128,logits_relative_L2=failed['relative_l2'],logits_max_abs=failed['max_abs'],mean_KL=failed['mean_kl'],L2_and_max_abs_fail=True,KL_pass=True),
    unavailable=['full-model 128 teacherforced/cache/greedy numerical or token comparisons','255/256/513 full-model controls','three-repeat alternating full-model timing','LoRA gradients and optimizer update comparison','raw full logits for centered-error reanalysis'],
    timing=dict(block_fused_cold_forward_backward_seconds=timing['block_fused_cold_fb'][0],includes_cold_JIT=True,block_native_warm_median=statistics.median(timing['block_native_warm_forward']),block_fused_warm_median=statistics.median(timing['block_fused_warm_forward']),full_model_load_seconds=timing['load_model'][0],full_model_speed_conclusion=False),
    terminal=dict(managed_running_false=True,managed_exit_code_unavailable=True,GPU_empty=True,release_utc=release['checked_utc'],original_deadline_unix=launch['deadline_unix'],attempt2_remaining_budget_seconds=launch['remaining_seconds']),
    decision='Keep the reference runtime; no kernel rollout or third retry from this result.',
    limitations=['Verified saved numerical receipts and frozen code; raw large tensors were not retained and no reviewer NN rerun was performed.','Mean KL passing does not override the predeclared combined logit contract. No causal attribution to constant shifts or task-quality degradation.','The first infer call executed additional teacherforced/greedy paths internally but failed before saving/comparing them, so they are not counted as passed.'],
    references=[ref(p) for p in [B/'probe-plan.json',B/'probe-source-manifest.json',B/'gpu_probe.py',B/'gpu-import-gate.json',B/'attempt2-repair-manifest.json',B/'cache_snapshot_attempt2.py',B/'gpu_probe_attempt2.py',B/'attempt2-cache-cpu-contract.json',B/'attempt1/process-terminal.json',B/'attempt1/numeric.stderr.log',B/'attempt1/gpu-numeric-v1/timings.json',A/'terminal.json',N/'metrics.json',N/'timings.json',N/'actual-bindings.json',N/'loading-info.json',A/'failure-traceback.txt',A/'collection.json',A/'attempt2-launch-record.json',A/'process-terminal.json',A/'gpu-release.json',B/'GPU_RESULT.md',Path(__file__)]]
)
(O/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(ref(O/'review.json')))
