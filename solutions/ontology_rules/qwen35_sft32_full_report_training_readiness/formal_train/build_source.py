from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).parent;T=B.parent;R=T.parents[1]
old=R/'case_retrieval/qwen35-preparation/sft-meta1-fusion/train_qwen35_meta32.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
s=old.read_text()
def edit(a,b):
 global s
 if a not in s:raise AssertionError(a[:180])
 s=s.replace(a,b)
edit('Length6144','Length13824')
edit('CPU mode verifies data/template/loss/meta without weights or CUDA. Training mode','CPU boundary contracts live in cpu_contract.py; this train-only entry')
edit('requires an allocated empty GPU and exact official files. No validation labels,','requires an allocated empty GPU and exact official files. No validation labels,')
edit('import qwen35_relation_cost as protocol','import qwen35_relation_cost as protocol\nimport runtime_contract as contract')
edit("'max_seq_length':6144","'max_seq_length':13824")
edit("TRAIN_SHA='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'","ORIGINAL_TRAIN_SHA='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'\nTRAIN_SHA='9e80e54125d3d46479acb9a3720e8b5f4c13ffacb5f65e82d3d9d10ac839d25e'\nENCODED_SHA='0755d5fc1e21675e85b96dc09311a4587e9c7b93a0c0d728360f76c6b2d7a980'\nFULLTEXT_PREPARATION_SHA='c489a9eeb2bc17462e26b23e7e7cf18950c8d0b66ba125bb2da0cfb1deefd14b'\nNATIVE_RECEIPT_SHA='6a82b6a63c88812a1ab1fdc86d92fb5af9e4b53a4606712790257ea6a8f2acfa'\nNATIVE_AUDIT_SHA='38afa9b9df2685b074346aa285c2e93e33e81d43764c694763e89a5f6f585935'\nSELECTED_LOSS_SHA='dfd0759fd43a181f7fbc4faa339a61a5495b3b1e6cadfae462f59a3d752d3cf9'\nNATIVE_LOSS_SHA='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'\nMODEL_SOURCE_SHA='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'\nRUNTIME_CONTRACT_SHA='"+hashlib.sha256((B/'runtime_contract.py').read_bytes()).hexdigest()+"'")
edit("if k not in {'resume','stop_after_epoch'}","if k not in {'resume','stop_after_epoch','resume_export_ack'}")
edit("preparation['training_sha256']!=TRAIN_SHA","preparation['training_sha256']!=ORIGINAL_TRAIN_SHA")
edit('    return rows,train_ids,excluded_ids','''    if sha256(args.fulltext_preparation)!=FULLTEXT_PREPARATION_SHA or sha256(args.native_receipt)!=NATIVE_RECEIPT_SHA or sha256(args.native_row_audit)!=NATIVE_AUDIT_SHA:
        raise ValueError('Frozen fulltext construction/native receipt/audit differs')
    fulltext=json.loads(Path(args.fulltext_preparation).read_text());native=json.loads(Path(args.native_receipt).read_text())
    if (fulltext['original_train']['sha256']!=ORIGINAL_TRAIN_SHA or fulltext['expanded_train']['sha256']!=TRAIN_SHA
        or fulltext['training_document_ids']!=train_ids or fulltext['excluded_document_ids_metadata_only']!=excluded_ids
        or not fulltext['answers_utf8_preserved_all124'] or not fulltext['all_metadata_CID_PID_local_evidence_preserved']
        or not fulltext['all_passage_text_reconstruction_exact'] or native['expanded_encoded_sha256']!=ENCODED_SHA
        or native['expanded']['supervised_tokens']!=7170 or not native['all_assistant_supervision_IDs_unchanged']):
        raise ValueError('Fulltext changes original target/metadata or native identity')
    if sha256(contract.__file__)!=RUNTIME_CONTRACT_SHA:raise ValueError('Runtime contract source differs')
    return rows,train_ids,excluded_ids''')
edit('def encode_all(rows,tokenizer,output):','def encode_all(rows,tokenizer,output,native_row_audit):')
edit('max(lengths)!=5523 or sum(lengths)!=402594','max(lengths)!=13549 or sum(lengths)!=1009181')
edit("    write_json(output/'token-mask-audit.json'",'''    native=json.loads(Path(native_row_audit).read_text())
    if helpers.json_sha(encoded)!=ENCODED_SHA or len(native)!=124:raise ValueError('Actual fulltext encoded124 differs from previously native-verified IDs')
    for index,(row,expected) in enumerate(zip(encoded,native)):
        actual=expected['expanded']
        if (row['row_id']!=expected['row_id'] or helpers.json_sha(row['input_ids'])!=actual['input_ids_sha256']
            or helpers.json_sha(row['labels'])!=actual['labels_sha256'] or row['supervised_tokens']!=actual['supervised_tokens']):
            raise ValueError('Exact input/mask contract differs at row '+str(index))
    write_json(output/'token-mask-audit.json' ''')
edit("max(p.name for p in output.glob('epoch-*'))","max(p.name for p in contract.complete_epoch_directories(output))")
edit("    if args.mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''","    if args.mode!='train':raise ValueError('CPU source checks use cpu_contract.py; this runner is train-only')\n    contract.check_deadline(args.absolute_deadline_unix)\n    if args.absolute_deadline_unix-time.time()>8400:raise ValueError('Absolute deadline exceeds authorized total budget')\n    if os.environ.get('PYTORCH_ALLOC_CONF') or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):raise ValueError('Allocator configuration differs')")
edit("    output=Path(args.output_dir)","    output=Path(args.output_dir)\n    if str(output)!='/dev/shm/patientphex-qwen35-fulltext32-1045-20261024/formal2ep':raise ValueError('Full state must stay in own fixed SHM namespace')\n    if not args.resume and args.resume_export_ack:raise ValueError('Fresh run cannot consume any old ACK')\n    if args.resume and not args.resume_export_ack:raise ValueError('Resume requires exact durable full-state ACK')")
edit("torch.backends.cudnn.benchmark=False","torch.backends.cudnn.benchmark=False\n        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True")
edit("        status('base_file_fingerprints')",'''        if torch.__version__!='2.7.1+cu126':raise RuntimeError('Exact A800 Torch runtime differs')
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        if sha256(q.__file__)!=MODEL_SOURCE_SHA or importlib.util.find_spec('fla') is not None:raise RuntimeError('Native Qwen runtime differs')
        if any(getattr(q,n).__module__!=q.__name__ or inspect.unwrap(getattr(q,n)).__module__!=q.__name__ for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']):raise RuntimeError('Native GDN dispatch differs')
        status('base_file_fingerprints')''')
edit("encode_all(rows,tokenizer,output)","encode_all(rows,tokenizer,output,args.native_row_audit)")
edit("        protocol.causal_loss_probe(torch,output)","        # Native/selected-CE numeric contract already frozen; do not repeat it per epoch.")
edit("'schema':'patientphex-qwen35-meta32-relation-lora-v1'","'schema':'patientphex-qwen35-meta32-fulltext-selectedCE-lora-v1'")
edit("'encoded_sha256':helpers.json_sha(encoded)}","'encoded_sha256':helpers.json_sha(encoded),'fulltext_preparation_sha256':FULLTEXT_PREPARATION_SHA,'native_receipt_sha256':NATIVE_RECEIPT_SHA,'selected_loss_sha256':SELECTED_LOSS_SHA,'runtime_contract_sha256':RUNTIME_CONTRACT_SHA}")
edit("'max_tokens':5523,'total_tokens':402594","'max_tokens':13549,'total_tokens':1009181")
edit("'resume_unit':'complete epoch only; partial epoch work must replay'","'resume_unit':'complete acknowledged epoch only; no automatic partial epoch replay'")
edit("[('train_qwen35_meta32.py',__file__),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol))]","[('train.py',__file__),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol)),('runtime_contract.py',contract.__file__)]")
start=s.index("        if args.mode=='cpu':\n            status('meta_full_hybrid_coverage')");end=s.index('        checkpoint=Path(args.resume)',start);s=s[:start]+s[end:]
edit("        if checkpoint:metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])",'''        if checkpoint:
            metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
            ack_receipt=contract.validate_export_ack(args.resume_export_ack,output,checkpoint,metadata,sha256(__file__))
            write_json(output/'resume-export-ack-verified.json',ack_receipt)
        contract.check_deadline(args.absolute_deadline_unix)''')
edit("        base.config.use_cache=False",'''        import selected_loss
        import transformers.loss.loss_utils as native_loss
        if sha256(selected_loss.__file__)!=SELECTED_LOSS_SHA or sha256(native_loss.__file__)!=NATIVE_LOSS_SHA:raise RuntimeError('Selected/native CE source changed')
        if base.loss_function is not native_loss.ForCausalLMLoss:raise RuntimeError('Native loss dispatch differs')
        base.loss_function=selected_loss.selected_causal_loss
        if base.loss_function is not selected_loss.selected_causal_loss:raise RuntimeError('Instance loss hook did not bind')
        (output/'selected_loss.py').write_bytes(Path(selected_loss.__file__).read_bytes())
        base.config.use_cache=False''')
edit("            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])",'''            contract.validate_state_structure(state,resumed_epochs,list(named))
            if contract.tree_digest({n:p for n,p in named.items()},torch)!=metadata['trainable_parameter_digest']:raise ValueError('Resumed adapter tensor values differ')
            expected_state_digest=contract.tree_digest(state,torch)
            if expected_state_digest!=metadata['serialized_state_digest']:raise ValueError('Serialized full-state fingerprint differs')
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])''')
edit("            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])",'''            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            actual_state={'completed_epochs':completed_epochs,'global_step':global_step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'optimizer_parameter_names':list(named)}
            restored_digest=contract.tree_digest(actual_state,torch)
            if restored_digest!=expected_state_digest:raise ValueError('Restored Adam/scheduler/all RNG/global-step state differs')
            write_json(output/'resume-state-verified.json',{'status':'PASS_EXACT_LOGICAL_FULL_STATE_RESTORE','completed_epochs':completed_epochs,'global_step':global_step,
                'serialized_state_digest':expected_state_digest,'restored_state_digest':restored_digest,'trainable_parameter_digest':metadata['trainable_parameter_digest'],
                'optimizer_parameter_count':len(optimizer.state),'scheduler_last_epoch':scheduler.last_epoch,'learning_rate':optimizer.param_groups[0]['lr'],
                'all_RNG_restored':True,'GPU_numerical_resume_equivalence_claimed':False})
            # Release CPU deserialization copies; optimizer keeps its normal GPU state.
            del state,actual_state''')
edit("            for group in groups:\n                optimizer", "            for group in groups:\n                contract.check_deadline(args.absolute_deadline_unix)\n                optimizer")
edit("            status('save_epoch_checkpoint')", "            contract.check_deadline(args.absolute_deadline_unix)\n            status('save_epoch_checkpoint')")
edit("'cuda_rng':torch.cuda.get_rng_state_all()}\n            torch.save", "'cuda_rng':torch.cuda.get_rng_state_all(),'optimizer_parameter_names':list(named)}\n            torch.save")
edit("'new_resume_numerical_equivalence_tested':False})\n            staging.replace(destination)","'new_resume_numerical_equivalence_tested':False,\n                'serialized_state_digest':contract.tree_digest(state,torch),'trainable_parameter_digest':contract.tree_digest({n:p for n,p in named.items()},torch)})\n            local_receipt=contract.local_publish_checkpoint(staging,destination,output)\n            write_json(output/f'local-checkpoint-epoch-{epoch+1:03d}.json',local_receipt)\n            contract.check_deadline(args.absolute_deadline_unix)")
edit("'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False}","'result_type':'training_only_no_evaluation_score_pending_second_epoch_persistent_ACK','new_resume_numerical_equivalence_tested':False,'persistent_export_ack_still_required':True}")
edit("parser.add_argument('--mode',choices=['cpu','train'],required=True)","parser.add_argument('--mode',choices=['train'],required=True)")
edit("['train','excluded-input','data-manifest','model','model-source-manifest','output-dir']","['train','excluded-input','data-manifest','fulltext-preparation','native-receipt','native-row-audit','model','model-source-manifest','output-dir']")
edit("parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)","parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)\n    parser.add_argument('--resume-export-ack');parser.add_argument('--absolute-deadline-unix',type=float,required=True)")
edit("    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):parser.error('CPU preparation does not pause/resume training')", "    if (args.resume is None and args.stop_after_epoch!=1) or (args.resume is not None and args.stop_after_epoch is not None):parser.error('Fresh epoch1 must stop for ACK; epoch2 resumes complete state once')")
compile(s,'train.py','exec');(B/'train.py').write_text(s)
for name in ['qwen35_smoke.py','qwen35_relation_cost.py','selected_loss.py','model-source-manifest.json']:
    shutil.copyfile(T/'stress-probe-v1'/name,B/name)
print(json.dumps({'source_sha256':hashlib.sha256((B/'train.py').read_bytes()).hexdigest(),'bytes':len(s.encode()),'runtime_contract_sha256':hashlib.sha256((B/'runtime_contract.py').read_bytes()).hexdigest()}))
