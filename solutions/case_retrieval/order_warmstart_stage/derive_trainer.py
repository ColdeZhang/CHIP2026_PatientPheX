"""Derive a narrow warm-start/order-stage entry from the frozen old32 trainer."""
from pathlib import Path
import difflib, hashlib, json

B = Path(__file__).resolve().parent
SOURCE = Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-training-feasibility/source/original_trainer.py')
old = SOURCE.read_text()
assert hashlib.sha256(old.encode()).hexdigest() == '08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
new = old

def replace(a, b):
    global new
    assert new.count(a) == 1, (a[:100], new.count(a))
    new = new.replace(a, b)

start = new.index('"""')
end = new.index('"""', start + 3) + 3
new = new[:start] + '''"""New two-epoch stage from the same verified old32 epoch2 adapter.
Fresh AdamW/scheduler/RNG; O/O versus R/O. Not an exact old-stage resume.
Each complete new epoch is exported before own-stage resume is allowed.
No model quality evaluation is performed here.
"""''' + new[end:]
replace("sha256=helpers.sha", """WARM_CHECKPOINT_SHA='5cecc79c3286089ad413b94ec1c52c0f78e84df2e3fc8b161ca2168f6a9e0120'
WARM_ADAPTER_SHA='4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'
WARM_CONFIG_SHA='96564441328e6759568e60c49ea87db432b9ec45cfd57001ef81640d4ff9c4f9'
REVERSE_SHA='34778f082d1057cab4646a3ef259fd0f15f4e48f13944ad5216bbde042e609b8'
UPSTREAM_TRAINER_SHA='08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
sha256=helpers.sha""")
replace("if k not in {'resume','stop_after_epoch'}", "if k not in {'resume','stop_after_epoch','export_ack'}")
insert = '''
def validate_stage_transition(mode, resume, stop_after_epoch, export_ack):
    if mode=='cpu':
        if resume or stop_after_epoch is not None or export_ack:
            raise ValueError('CPU preparation cannot advance/resume the stage')
        return
    if not resume:
        if stop_after_epoch!=1 or export_ack:
            raise ValueError('Initial stage must stop after epoch1 before persistent export')
    elif stop_after_epoch!=2 or not export_ack:
        raise ValueError('Own-stage epoch2 requires explicit stop2 and export receipt')

def order_variant(arm, epoch):
    if arm not in {'original','reverse-first'} or epoch not in {0,1}:
        raise ValueError('Unknown two-epoch order stage')
    return 'R' if arm=='reverse-first' and epoch==0 else 'O'

def verify_warm_start(args, training_ids, excluded_ids):
    checkpoint=Path(args.warm_start_checkpoint)
    if sha256(checkpoint/'checkpoint.json')!=WARM_CHECKPOINT_SHA:
        raise ValueError('Require exact completed old32 checkpoint metadata')
    meta=json.loads((checkpoint/'checkpoint.json').read_text());fp=meta['fingerprint']
    if meta['completed_epochs']!=2 or meta['global_step']!=32 or meta['smoke_only'] is not False:
        raise ValueError('Warm-start source is not the complete old32 epoch2')
    if fp['source_sha256']!=UPSTREAM_TRAINER_SHA or fp['train_sha256']!=TRAIN_SHA or fp['excluded_input_sha256']!=EXCLUDED_SHA:
        raise ValueError('Old32 source/data fingerprint differs')
    if fp['training_document_ids']!=training_ids or fp['excluded_document_ids']!=excluded_ids:
        raise ValueError('Warm-start provenance is not identical strict train32')
    expected={'adapter_model.safetensors':WARM_ADAPTER_SHA,'adapter_config.json':WARM_CONFIG_SHA}
    for name,digest in expected.items():
        if meta['files']['adapter/'+name]!=digest or sha256(checkpoint/'adapter'/name)!=digest:
            raise ValueError('Actual warm-start adapter file differs: '+name)
    cfg=json.loads((checkpoint/'adapter/adapter_config.json').read_text())
    if cfg['r']!=16 or cfg['lora_alpha']!=32 or cfg['lora_dropout']!=.05 or set(cfg['target_modules'])!=set(protocol.TARGETS) or cfg.get('bias')!='none':
        raise ValueError('Warm-start actual LoRA configuration differs')
    return {'kind':'new_stage_adapter_warm_start_fresh_optimizer_RNG',
        'source_checkpoint_sha256':WARM_CHECKPOINT_SHA,'adapter_files':expected,
        'source_epochs':2,'source_optimizer_calls':32,'old_optimizer_loaded':False,
        'old_training_state_opened':False,'not_exact_old_stage_resume':True}

def reverse_rows(args, original, training_ids, excluded_ids):
    if sha256(args.reverse_train)!=REVERSE_SHA:
        raise ValueError('Frozen reversible R124 source differs')
    rows,train_ids,heldout_ids=training_rows(args.reverse_train,args.excluded_input)
    if len(rows)!=124 or train_ids!=training_ids or heldout_ids!=excluded_ids:
        raise ValueError('R124 document isolation differs')
    for old,row in zip(original,rows):
        if row['row_id']!=old['row_id']+':reverse-candidates' or row['pmc_id']!=old['pmc_id'] or row['patient_ids']!=old['patient_ids']:
            raise ValueError('R124 positional original-ID correspondence differs')
    return rows

def verify_export_ack(checkpoint, metadata, ack_path):
    if not ack_path:
        raise ValueError('Own-stage resume requires completed persistent export receipt')
    ack=json.loads(Path(ack_path).read_text())
    if ack.get('status')!='persistent_checkpoint_verified':
        raise ValueError('Missing verified persistent export receipt')
    if ack.get('checkpoint_sha256')!=sha256(checkpoint/'checkpoint.json') or ack.get('completed_epochs')!=metadata['completed_epochs'] or ack.get('global_step')!=metadata['global_step']:
        raise ValueError('Export receipt does not bind this exact complete checkpoint')
    if Path(ack.get('source_checkpoint_directory','')).resolve()!=checkpoint.resolve():
        raise ValueError('Export receipt source checkpoint directory differs')
    if ack.get('persistent_mount_id')!='5a9955da-5240-41cc-9e52-05d3d8da8a3c' or not isinstance(ack.get('persistent_directory'),str) or not ack['persistent_directory'] or '..' in Path(ack['persistent_directory']).parts:
        raise ValueError('Persistent export destination differs or is unsafe')
    if ack.get('run_manifest_sha256')!=sha256(checkpoint.parent/'run_manifest.json'):
        raise ValueError('Persistent export run manifest differs')
    expected=dict(metadata['files']);expected['checkpoint.json']=sha256(checkpoint/'checkpoint.json')
    receipt_files=ack.get('files',{})
    if not set(expected).issubset(receipt_files) or '_run/run_manifest.json' not in receipt_files:
        raise ValueError('Persistent export inventory is not complete')
    for name,actual in receipt_files.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe persistent export inventory member')
        if name in expected:
            path=checkpoint/name;digest=expected[name]
        else:
            if len(relative.parts)<2 or relative.parts[0]!='_run':
                raise ValueError('Unexpected export member outside checkpoint/run support')
            path=checkpoint.parent/Path(*relative.parts[1:])
            if not path.is_file() or path.is_symlink():
                raise ValueError('Export support must be a regular nonsymlink file')
            digest=sha256(path)
        if not path.is_file() or path.is_symlink() or actual!={'sha256':digest,'bytes':path.stat().st_size}:
            raise ValueError('Persistent export size/SHA differs: '+name)
    return {'path':str(Path(ack_path).resolve()),'sha256':sha256(ack_path),
        'persistent_mount_id':ack['persistent_mount_id'],'persistent_directory':ack['persistent_directory']}

'''
replace('\ndef model_files(model_dir,source_manifest,require_weights):', insert + '\ndef model_files(model_dir,source_manifest,require_weights):')
replace('def encode_all(rows,tokenizer,output):', "def encode_all(rows,tokenizer,output,variant='O'):")
replace("    if len(encoded)!=124 or max(lengths)!=5523 or sum(lengths)!=402594 or any(n>CONFIG['max_seq_length'] for n in lengths):", "    expected_tokens={'O':402594,'R':402596}[variant]\n    if len(encoded)!=124 or max(lengths)!=5523 or sum(lengths)!=expected_tokens or any(n>CONFIG['max_seq_length'] for n in lengths):")
replace("write_json(output/'token-mask-audit.json'", "write_json(output/('token-mask-audit-'+variant+'.json')")
replace("        status('source_and_data');rows,training_ids,excluded_ids=verify_sources_and_data(args)", """        status('source_and_data');rows,training_ids,excluded_ids=verify_sources_and_data(args)
        warm_start=verify_warm_start(args,training_ids,excluded_ids)
        reversed_rows=reverse_rows(args,rows,training_ids,excluded_ids)
        validate_stage_transition(args.mode,args.resume,args.stop_after_epoch,args.export_ack)""")
replace("        status('token_mask_audit');encoded=encode_all(rows,tokenizer,output)", """        status('token_mask_audit');encoded_by_variant={'O':encode_all(rows,tokenizer,output,'O'),
            'R':encode_all(reversed_rows,tokenizer,output,'R')}
        encoded=encoded_by_variant['O']""")
replace("'schema':'patientphex-qwen35-meta32-relation-lora-v1'", "'schema':'patientphex-qwen35-meta32-order-warmstart-stage-v1'")
replace("            'encoded_sha256':helpers.json_sha(encoded)}", """            'encoded_sha256':helpers.json_sha(encoded),
            'encoded_variant_sha256':{k:helpers.json_sha(v) for k,v in encoded_by_variant.items()},
            'reverse_train_sha256':REVERSE_SHA,'warm_start':warm_start,
            'order_arm':args.order_arm,'epoch_variants':[order_variant(args.order_arm,e) for e in range(2)],
            'upstream_old32_trainer_sha256':UPSTREAM_TRAINER_SHA}""")
replace("'fresh_base_on_initial_run':True,'validation_labels_used':False}", """'fresh_base_on_initial_run':False,'fresh_optimizer_on_initial_run':True,'validation_labels_used':False,
            'old_stage_completed_epochs':2,'new_stage_epochs':2,'warm_start':warm_start,
            'epoch_tokens':[402596 if order_variant(args.order_arm,e)=='R' else 402594 for e in range(2)],
            'epoch_variants':[order_variant(args.order_arm,e) for e in range(2)]}""")
replace("                'rows':124,'documents':32,'max_tokens':5523,'full_tokens':402594,'supervised_tokens':7170,", "                'rows_per_variant':124,'documents':32,'max_tokens':5523,'full_tokens_by_variant':{'O':402594,'R':402596},'supervised_tokens_per_variant':7170,")
replace("        if checkpoint:metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])", """        export_receipt=None
        if checkpoint:
            metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
            export_receipt=verify_export_ack(checkpoint,metadata,args.export_ack)""")
replace("'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid()}", "'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid(),'persistent_export_receipt':export_receipt,'warm_start':warm_start}")
replace("        else:model,named,coverage=protocol.attach_adapter(base,torch,peft)", """        else:
            model=PeftModel.from_pretrained(base,Path(args.warm_start_checkpoint)/'adapter',is_trainable=True)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            coverage={'warm_started_verified_old32_adapter':warm_start,'target_names':protocol.TARGETS}
        adapter_config=model.peft_config['default']
        if adapter_config.r!=16 or adapter_config.lora_alpha!=32 or adapter_config.lora_dropout!=.05 or adapter_config.inference_mode:
            raise RuntimeError('Actual trainable warm-start/resume LoRA configuration differs')
        if any(p.dtype!=torch.float32 for p in named.values()):
            raise RuntimeError('Require actual FP32 LoRA parameters as in old stage')""")
replace("        model.train();first_update_checked=False;status('formal_training')", """        else:
            # Adapter construction/loading may consume RNG. Reset the new stage only
            # after loading; own-stage resume instead restores all saved RNG above.
            random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
            if optimizer.state or scheduler.last_epoch!=0 or optimizer.param_groups[0]['lr']!=0.:
                raise RuntimeError('New stage must start with empty Adam and fresh warmup schedule')
            write_json(output/'new-stage-initialization.json',{'warm_start':warm_start,'optimizer_state_entries':0,
                'scheduler_last_epoch':scheduler.last_epoch,'initial_lr':optimizer.param_groups[0]['lr'],
                'new_stage_seed':CONFIG['seed'],'old_training_state_opened':False})
        model.train();first_update_checked=False;status('formal_training')""")
replace("        for epoch in range(completed_epochs,2):\n            groups=", "        for epoch in range(completed_epochs,2):\n            encoded=encoded_by_variant[order_variant(args.order_arm,epoch)]\n            groups=")
replace("'fresh_run':not bool(checkpoint)", "'fresh_optimizer_stage':not bool(checkpoint),'warm_started_adapter':True")
replace("'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False})", "'source_sha256':sha256(__file__),'result_type':'new_warm_start_stage_training_only_no_evaluation_score','old_stage_epochs':2,'new_stage_epochs':2,'order_arm':args.order_arm,'warm_start':warm_start,'new_resume_numerical_equivalence_tested':False})")
replace("'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,", "'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,'elapsed_scope':'current_invocation_only_not_full_stage','started_completed_epochs':resumed_epochs,'full_stage_elapsed_seconds':None,")
replace("    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)", """    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    parser.add_argument('--warm-start-checkpoint',required=True);parser.add_argument('--reverse-train',required=True)
    parser.add_argument('--order-arm',choices=['original','reverse-first'],required=True)
    parser.add_argument('--export-ack')""")
replace("    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):", "    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None or args.export_ack):")
compile(new, 'train_order_warmstart.py', 'exec')
(B/'train_order_warmstart.py').write_text(new)
(B/'upstream-trainer.py').write_text(old)
(B/'trainer.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='old32-trainer',tofile='new-order-warmstart-stage')))
print(json.dumps({'source_sha256':hashlib.sha256(new.encode()).hexdigest(),'diff_bytes':(B/'trainer.diff').stat().st_size}))
