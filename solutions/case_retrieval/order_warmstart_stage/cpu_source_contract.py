"""Pure standard-library contracts; no torch/tokenizer/model/data-quality execution."""
from pathlib import Path
import ast, copy, datetime, hashlib, json, random, sys, tempfile, time

B=Path(__file__).resolve().parent
R=Path('/workspace/patientphex-runs/round2-20261023')
P=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion'
F=R/'research/candidate-order-training-feasibility'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
ref=lambda p:dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
start=time.monotonic()
old_tree=ast.parse((B/'upstream-trainer.py').read_text())
new_tree=ast.parse((B/'train_order_warmstart.py').read_text())
compile(new_tree,str(B/'train_order_warmstart.py'),'exec')
oldf={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
newf={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
dump=lambda n:ast.dump(n,include_attributes=False)
unchanged=['training_rows','verify_sources_and_data','model_files','validate_checkpoint','batch_plan','verify_fingerprint','validate_stop_epoch','epoch_outcome','collate']
assert all(dump(oldf[n])==dump(newf[n]) for n in unchanged)
names=['batch_plan','verify_fingerprint','validate_checkpoint','validate_stage_transition','verify_export_ack','order_variant','semantic_configuration']
ns={'Path':Path,'json':json,'random':random,'sha256':sha,'CONFIG':{'same':'sentinel'}}
exec(compile(ast.Module(body=[newf[n] for n in names],type_ignores=[]),'<selected-pure-functions>','exec'),ns)

checkpoint=load(P/'formal2ep/epoch-002/checkpoint.json');manifest=load(P/'formal2ep/run_manifest.json')
preparation=load(P/'preparation-manifest.json');prior=load(F/'summary.json')
latest=load(R/'case_retrieval/qwen35-preparation/thinking-sampling-followup/responses.manifest.json')
assert checkpoint['fingerprint']==manifest['fingerprint']
assert checkpoint['completed_epochs']==2 and checkpoint['global_step']==32 and checkpoint['smoke_only'] is False
assert checkpoint['fingerprint']['training_document_ids']==preparation['training_document_ids']
assert not set(preparation['training_document_ids']) & set(preparation['excluded_document_ids'])
assert preparation['no_F1_or_outer0_in_any_generator_or_SFT_training']
assert checkpoint['files']['adapter/adapter_model.safetensors']==latest['fingerprint']['adapter_files']['adapter_model.safetensors']=='4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'
assert checkpoint['files']['adapter/adapter_config.json']==latest['fingerprint']['adapter_files']['adapter_config.json']
assert checkpoint['fingerprint']['base_files']['actual_files']==latest['fingerprint']['model_files']
try:
    ns['validate_checkpoint'](P/'formal2ep',P/'formal2ep/epoch-002',checkpoint['fingerprint'])
    raise AssertionError('Old completed checkpoint wrongly accepted as own resume')
except ValueError as e:
    assert 'unfinished formal124x2-meta32' in str(e)

transition=ns['validate_stage_transition']
for args in [('cpu',None,None,None),('train',None,1,None),('train','new/epoch-001',2,'export-ack.json')]:transition(*args)
bad=[('train',None,None,None),('train',None,2,None),('train','old/epoch-002',2,None),('train','new/epoch-001',None,'ack'),('cpu','new/epoch-001',None,None)]
for args in bad:
    try:transition(*args);raise AssertionError('Invalid transition accepted')
    except ValueError:pass
from types import SimpleNamespace
a=SimpleNamespace(model='fixed',order_arm='original',resume=None,stop_after_epoch=1,export_ack=None)
b=SimpleNamespace(model='fixed',order_arm='original',resume='new/epoch-001',stop_after_epoch=2,export_ack='epoch1-ack')
assert ns['semantic_configuration'](a)==ns['semantic_configuration'](b)
b.order_arm='reverse-first';assert ns['semantic_configuration'](a)!=ns['semantic_configuration'](b)

audit=load(F/'row-inverse-and-token-audit.json');assert len(audit)==124
assert all(x['inverse_all_fields_exact'] and x['target_inverse_byteexact'] for x in audit)
calls=[]
for e in range(2):
    groups=ns['batch_plan'](124,1,8,20261023,e)
    assert [len(g) for g in groups]==[8]*15+[4]
    assert sorted(i for g in groups for row in g for i in row)==list(range(124))
    for gi,g in enumerate(groups):
        ids=[row[0] for row in g];step=e*16+gi
        lr=1e-4*(step/2 if step<2 else max(0,(32-step)/30))
        variant=ns['order_variant']('reverse-first',e)
        calls.append({'optimizer_call':step+1,'new_stage_epoch':e+1,'sample_ids':[audit[i]['original_row_id'] for i in ids],
            'rows':len(ids),'control_variant':'O','augmented_variant':variant,'lr_used':lr,
            'control_tokens':sum(audit[i]['original']['full_tokens'] for i in ids),
            'augmented_tokens':sum(audit[i]['reversed' if variant=='R' else 'original']['full_tokens'] for i in ids),
            'control_supervised_tokens':sum(audit[i]['original']['supervised_tokens'] for i in ids),
            'augmented_supervised_tokens':sum(audit[i]['reversed' if variant=='R' else 'original']['supervised_tokens'] for i in ids)})
assert len(calls)==32 and sum(x['rows'] for x in calls)==248
assert sum(x['control_tokens'] for x in calls)==805188 and sum(x['augmented_tokens'] for x in calls)==805190
assert sum(x['control_supervised_tokens'] for x in calls)==sum(x['augmented_supervised_tokens'] for x in calls)==14340
stats=load(P/'json-value-weighted-cpu-preparation/training-stats-diagnostic/summary.json')['arms']['old32_JSON_CE']
for epoch in stats['epochs'].values():
    for row in epoch['recorded_step_losses']:
        assert abs(calls[row['step']-1]['lr_used']-row['learning_rate'])<1e-15
(B/'matched-32-call-schedule.json').write_text(json.dumps(calls,indent=2)+'\n')

def has_load(node):
    return any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='torch' and x.func.attr=='load' for x in ast.walk(node))
oldload=next(n for n in ast.walk(oldf['run']) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='checkpoint' and has_load(ast.Module(body=n.body,type_ignores=[])))
newload=next(n for n in ast.walk(newf['run']) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='checkpoint' and has_load(ast.Module(body=n.body,type_ignores=[])))
assert dump(ast.Module(body=oldload.body,type_ignores=[]))==dump(ast.Module(body=newload.body,type_ignores=[]))
assert not has_load(ast.Module(body=newload.orelse,type_ignores=[]))
oldloss=next(n for n in ast.walk(oldf['run']) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='indices')
newloss=next(n for n in ast.walk(newf['run']) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='indices')
assert dump(oldloss)==dump(newloss)

ack_tests=[]
with tempfile.TemporaryDirectory(dir=B,prefix='cpu-ack-fixture-') as tmp:
    root=Path(tmp);cp=root/'epoch-001';(cp/'adapter').mkdir(parents=True)
    for name,payload in [('adapter/config.json',b'config'),('adapter/model.bin',b'model'),('training_state.pt',b'synthetic state')]:
        (cp/name).write_bytes(payload)
    (root/'run_manifest.json').write_text('{"synthetic":true}\n')
    (root/'progress.json').write_text('{"completed_epochs":1}\n')
    meta={'files':{str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()},'completed_epochs':1,'global_step':16}
    (cp/'checkpoint.json').write_text(json.dumps(meta))
    files={n:{'bytes':(cp/n).stat().st_size,'sha256':s} for n,s in meta['files'].items()}
    files['checkpoint.json']={'bytes':(cp/'checkpoint.json').stat().st_size,'sha256':sha(cp/'checkpoint.json')}
    for n in ['run_manifest.json','progress.json']:files['_run/'+n]={'bytes':(root/n).stat().st_size,'sha256':sha(root/n)}
    ack={'status':'persistent_checkpoint_verified','checkpoint_sha256':sha(cp/'checkpoint.json'),'completed_epochs':1,'global_step':16,
         'source_checkpoint_directory':str(cp),'persistent_mount_id':'5a9955da-5240-41cc-9e52-05d3d8da8a3c','persistent_directory':'synthetic-only/no-upload',
         'files':files,'run_manifest_sha256':sha(root/'run_manifest.json')}
    ap=root/'ack.json';ap.write_text(json.dumps(ack));ns['verify_export_ack'](cp,meta,ap);ack_tests.append('complete_pass')
    for label,mutate in [
        ('wrong_checkpoint_sha',lambda x:x.update(checkpoint_sha256='0'*64)),
        ('wrong_epoch',lambda x:x.update(completed_epochs=2)),
        ('wrong_step',lambda x:x.update(global_step=32)),
        ('wrong_mount',lambda x:x.update(persistent_mount_id='other')),
        ('wrong_source_directory',lambda x:x.update(source_checkpoint_directory=str(root/'other'))),
        ('missing_state',lambda x:x['files'].pop('training_state.pt')),
        ('wrong_size',lambda x:x['files']['training_state.pt'].update(bytes=0)),
        ('wrong_manifest',lambda x:x.update(run_manifest_sha256='0'*64)),
        ('missing_run_manifest',lambda x:x['files'].pop('_run/run_manifest.json')),
        ('unsafe_extra',lambda x:x['files'].update({'_run/../outside':{'bytes':1,'sha256':'0'*64}})),
    ]:
        badack=copy.deepcopy(ack);mutate(badack);ap.write_text(json.dumps(badack))
        try:ns['verify_export_ack'](cp,meta,ap);raise AssertionError(label+' incorrectly accepted')
        except ValueError:ack_tests.append(label+'_rejected')

assert 'torch' not in sys.modules
references={str(p.name):ref(p) for p in [B/'train_order_warmstart.py',B/'derive_trainer.py',B/'upstream-trainer.py',B/'trainer.diff',B/'matched-32-call-schedule.json',P/'formal2ep/epoch-002/checkpoint.json',P/'formal2ep/run_manifest.json',P/'preparation-manifest.json',F/'summary.json',F/'row-inverse-and-token-audit.json']}
result={'status':'PASS_LOCAL_SOURCE_AND_SCHEDULE_CONTRACTS_NOT_RUNTIME_OR_QUALITY','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cpu_seconds':time.monotonic()-start,'references':references,'unchanged_function_AST':unchanged,
    'same_actual_old32_adapter_and_base_inventory_as_latest_inference':True,'old32_training_and_excluded_document_sets_exact':True,
    'old_preparation_fresh_baseline_flags_unchanged':{'fresh_base_required':preparation['fresh_base_required'],'old_adapter_continuation_forbidden':preparation['old_adapter_continuation_forbidden']},
    'old_completed_checkpoint_rejected_by_original_resume':True,'new_resume_control_arguments_do_not_change_semantic_configuration':True,
    'old_checkpoint_optimizer_scheduler_RNG_resume_body_AST_exact':True,'old_loss_microbatch_body_AST_exact':True,
    'warm_start_branch_has_no_torch_load':True,'initial_stop1_export_then_resume_stop2_required':True,'export_ack_contracts':ack_tests,
    'schedule':{'new_epochs':2,'calls':32,'nonzero_lr_calls':31,'exposures':248,'groups':[8]*15+[4],
        'control':'O/O','augmentation':'R/O','original_sample_ID_per_call_shared':True,'control_tokens':805188,'augmented_tokens':805190,'supervised_tokens_each':14340,
        'warmup_steps':2,'peak_lr':.0001,'final_used_lr':calls[-1]['lr_used'],'after_final_scheduler_lr':0.0,'recorded_old32_LR_observations_match':True},
    'limitations':['No model/Peft/optimizer objects loaded; actual warm-start branch still needs authorized runtime CPU/source checks.',
        'Common seed/RNG reset is identical at new-stage start; changed per-row token shapes can change later dropout RNG consumption, so do not claim identical stochastic masks.',
        'No fresh4epoch or exact old optimizer continuation: old2epoch weights plus a new2epoch optimizer stage.',
        'Complete-epoch exit/export/resume adds two extra base loads across paired arms; owner measures/budgets separately.',
        'Final elapsed_seconds belongs to final invocation, not both new epochs; owner aggregates both segments.',
        'Metadata/source references were verified locally; no fresh remote weights or storage checks here.'],
    'no_gpu_or_network':True,'no_tokenizer_or_model_import':True,'no_F1_gold_or_score':True,'source_sha256':sha(__file__)}
(B/'cpu-source-contract.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':result['status'],'source_sha256':sha(B/'train_order_warmstart.py'),'cpu_seconds':result['cpu_seconds'],'ack_cases':len(ack_tests)}))
