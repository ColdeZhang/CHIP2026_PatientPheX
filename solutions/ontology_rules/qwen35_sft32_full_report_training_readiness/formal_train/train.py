#!/usr/bin/env python3
"""Fixed Qwen3.5 relation LoRA:124 meta32 OOF rows,2 epochs,complete-epoch resume.

CPU boundary contracts live in cpu_contract.py; this train-only entry
requires an allocated empty GPU and exact official files. No validation labels,
inference, quantization, packing, smoke adapter or alternative task is supported.
Fresh base is required; only this exact meta32 run may resume complete epochs.
Resume state is saved/restored; numerical GPU resume equivalence is not claimed.
Length13824 is only a rejection cap: batches pad to their actual maximum, never to cap.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import qwen35_smoke as helpers
import qwen35_relation_cost as protocol
import runtime_contract as contract

HELPER_SHA='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
PROTOCOL_SHA='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
OLD_TRAINER_SHA='2a25b83d5d9e3dcc3f57cc8e2ff1aaf519a41a709e4ff289b2b26db9fab51423'
SOURCE_MANIFEST_SHA='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
CONFIG={'max_seq_length':13824,'batch_size':1,'gradient_accumulation':8,'epochs':2,
        'learning_rate':1e-4,'weight_decay':.01,'warmup_ratio':.05,'max_grad_norm':1.,
        'rank':16,'alpha':32,'dropout':.05,'seed':20261023,'targets':protocol.TARGETS,
        'dtype':'bfloat16','attention':'sdpa','quantization':False,'packing':False,
        'gradient_checkpointing':{'use_reentrant':False}}
ORIGINAL_TRAIN_SHA='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
TRAIN_SHA='9e80e54125d3d46479acb9a3720e8b5f4c13ffacb5f65e82d3d9d10ac839d25e'
ENCODED_SHA='0755d5fc1e21675e85b96dc09311a4587e9c7b93a0c0d728360f76c6b2d7a980'
FULLTEXT_PREPARATION_SHA='c489a9eeb2bc17462e26b23e7e7cf18950c8d0b66ba125bb2da0cfb1deefd14b'
NATIVE_RECEIPT_SHA='6a82b6a63c88812a1ab1fdc86d92fb5af9e4b53a4606712790257ea6a8f2acfa'
NATIVE_AUDIT_SHA='38afa9b9df2685b074346aa285c2e93e33e81d43764c694763e89a5f6f585935'
SELECTED_LOSS_SHA='dfd0759fd43a181f7fbc4faa339a61a5495b3b1e6cadfae462f59a3d752d3cf9'
NATIVE_LOSS_SHA='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'
MODEL_SOURCE_SHA='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
RUNTIME_CONTRACT_SHA='3f0c2427a0556790b7169561c7e8b1479a2d21dc9b8c91f712e4aee038a241b9'
EXCLUDED_SHA='2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
PREPARATION_SHA='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
COPIED_TRAINER_SHA='9d2e0eeb34e6bfa76ea90cbf6f9ffad996ab527e4746bb936e4f3cbc58724b8b'
sha256=helpers.sha
write_json=helpers.write


def semantic_configuration(args):
    return {'fixed':CONFIG,'paths_and_device':{k:v for k,v in vars(args).items()
            if k not in {'resume','stop_after_epoch','resume_export_ack'}}}


def training_rows(train, excluded):
    data, heldout = helpers.rows(train), helpers.rows(excluded)
    if not data or not heldout:
        raise ValueError('Training and excluded input must be nonempty')
    heldout_ids = [str(x['pmc_id']) for x in heldout]
    if len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError('Duplicate excluded document')
    if any('entities' in x or 'association' in x for x in heldout):
        raise ValueError('Excluded input must not contain label fields')
    ids = [x.get('row_id') for x in data]
    if any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Training row IDs must be unique strings')
    document_ids = sorted({str(x['pmc_id']) for x in data})
    if len(document_ids) != 32 or set(document_ids) & set(heldout_ids):
        raise ValueError('Require the actual 32-document training set, disjoint from excluded input')
    for row in data:
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
            raise ValueError('Messages must end with user: '+row['row_id'])
        if any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant'}
               or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Require text-only messages: '+row['row_id'])
        if not isinstance(row.get('answer'), str) or not isinstance(json.loads(row['answer']), (list,dict)):
            raise ValueError('Answer must be a JSON array/object: '+row['row_id'])
    return data, document_ids, sorted(heldout_ids)


def verify_sources_and_data(args):
    if sha256(inspect.getfile(helpers))!=HELPER_SHA or sha256(inspect.getfile(protocol))!=PROTOCOL_SHA:
        raise ValueError('Frozen Qwen3.5 helper/protocol source differs')
    if sha256(args.train)!=TRAIN_SHA or sha256(args.excluded_input)!=EXCLUDED_SHA:
        raise ValueError('Require exact meta32 two16-to16 OOF124 train and stripped outer0+metaF1 input')
    if sha256(args.model_source_manifest)!=SOURCE_MANIFEST_SHA:
        raise ValueError('Official15-file fixed source manifest differs')
    rows,train_ids,excluded_ids=training_rows(args.train,args.excluded_input)
    if len(rows)!=124 or len(train_ids)!=32 or len(excluded_ids)!=32:
        raise ValueError('Require124 rows/32train documents/32excluded input documents')
    if sha256(args.data_manifest)!=PREPARATION_SHA:
        raise ValueError('Frozen meta32 candidate provenance differs')
    preparation=json.loads(Path(args.data_manifest).read_text())
    if (preparation['training_sha256']!=ORIGINAL_TRAIN_SHA or preparation['excluded_input_sha256']!=EXCLUDED_SHA
        or preparation['training_document_ids']!=train_ids or preparation['excluded_document_ids']!=excluded_ids
        or not preparation['all_target_documents_excluded_from_own_generators']):
        raise ValueError('Meta32 provenance identity/isolation differs')
    if sha256(args.fulltext_preparation)!=FULLTEXT_PREPARATION_SHA or sha256(args.native_receipt)!=NATIVE_RECEIPT_SHA or sha256(args.native_row_audit)!=NATIVE_AUDIT_SHA:
        raise ValueError('Frozen fulltext construction/native receipt/audit differs')
    fulltext=json.loads(Path(args.fulltext_preparation).read_text());native=json.loads(Path(args.native_receipt).read_text())
    if (fulltext['original_train']['sha256']!=ORIGINAL_TRAIN_SHA or fulltext['expanded_train']['sha256']!=TRAIN_SHA
        or fulltext['training_document_ids']!=train_ids or fulltext['excluded_document_ids_metadata_only']!=excluded_ids
        or not fulltext['answers_utf8_preserved_all124'] or not fulltext['all_metadata_CID_PID_local_evidence_preserved']
        or not fulltext['all_passage_text_reconstruction_exact'] or native['expanded_encoded_sha256']!=ENCODED_SHA
        or native['expanded']['supervised_tokens']!=7170 or not native['all_assistant_supervision_IDs_unchanged']):
        raise ValueError('Fulltext changes original target/metadata or native identity')
    if sha256(contract.__file__)!=RUNTIME_CONTRACT_SHA:raise ValueError('Runtime contract source differs')
    return rows,train_ids,excluded_ids


def model_files(model_dir,source_manifest,require_weights):
    source=json.loads(Path(source_manifest).read_text())
    if source.get('model')!='Qwen/Qwen3.5-9B' or source.get('hf_revision')!=helpers.REVISION or len(source['files'])!=15:
        raise ValueError('Unexpected official model source/revision/file count')
    expected={x['name']:x for x in source['files']}
    if len(expected)!=15 or any(Path(n).name!=n for n in expected):
        raise ValueError('Duplicate or nonlocal official filename')
    names=sorted(expected if require_weights else helpers.CPU_FILES)
    actual={}
    for name in names:
        path=Path(model_dir)/name
        actual[name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        if actual[name]!={'bytes':expected[name]['size'],'sha256':expected[name]['sha256']}:
            raise ValueError('Actual official file differs: '+name)
    return {'model':source['model'],'hf_revision':source['hf_revision'],'source_manifest_sha256':sha256(source_manifest),
            'actual_files':actual,'all15_files_verified':require_weights}


def encode_all(rows,tokenizer,output,native_row_audit):
    encoded=[helpers.encode(row,tokenizer) for row in rows]
    lengths=[len(e['input_ids']) for e in encoded]
    if len(encoded)!=124 or max(lengths)!=13549 or sum(lengths)!=1009181 or any(n>CONFIG['max_seq_length'] for n in lengths):
        raise ValueError('Frozen124 token accounting differs; no truncation permitted')
    if sum(e['supervised_tokens'] for e in encoded)!=7170:
        raise ValueError('Frozen supervised-token accounting differs')
    native=json.loads(Path(native_row_audit).read_text())
    if helpers.json_sha(encoded)!=ENCODED_SHA or len(native)!=124:raise ValueError('Actual fulltext encoded124 differs from previously native-verified IDs')
    for index,(row,expected) in enumerate(zip(encoded,native)):
        actual=expected['expanded']
        if (row['row_id']!=expected['row_id'] or helpers.json_sha(row['input_ids'])!=actual['input_ids_sha256']
            or helpers.json_sha(row['labels'])!=actual['labels_sha256'] or row['supervised_tokens']!=actual['supervised_tokens']):
            raise ValueError('Exact input/mask contract differs at row '+str(index))
    write_json(output/'token-mask-audit.json' ,{'rows':[{'row_id':e['row_id'],'full_tokens':len(e['input_ids']),
        'prefix_tokens':e['prefix_tokens'],'supervised_tokens':e['supervised_tokens']} for e in encoded],
        'rows_count':124,'total_full_tokens':sum(lengths),'max_full_tokens':max(lengths),
        'supervised_tokens':7170,'truncated_rows':0,'prefix_mask_eos_passed':True,
        'encoded_sha256':helpers.json_sha(encoded),'fulltext_preparation_sha256':FULLTEXT_PREPARATION_SHA,'native_receipt_sha256':NATIVE_RECEIPT_SHA,'selected_loss_sha256':SELECTED_LOSS_SHA,'runtime_contract_sha256':RUNTIME_CONTRACT_SHA})
    return encoded


def validate_checkpoint(output,checkpoint,fingerprint):
    if checkpoint.resolve().parent!=output.resolve():raise ValueError('Resume checkpoint must belong to original output')
    if any(output.glob('.epoch-*.incomplete')):raise ValueError('Inspect incomplete checkpoint staging before resume')
    metadata=json.loads((checkpoint/'checkpoint.json').read_text())
    verify_fingerprint(metadata['fingerprint'],fingerprint)
    completed=int(metadata['completed_epochs']);expected=output/f'epoch-{completed:03d}'
    if checkpoint.resolve()!=expected.resolve() or checkpoint.name!=max(p.name for p in contract.complete_epoch_directories(output)):
        raise ValueError('Resume requires latest complete epoch checkpoint')
    if not 1<=completed<2 or metadata['global_step']!=16*completed or metadata.get('smoke_only'):
        raise ValueError('Checkpoint is not an unfinished formal124x2-meta32 run')
    for name,digest in metadata['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe checkpoint member')
        if sha256(checkpoint/relative)!=digest:raise ValueError('Checkpoint SHA mismatch: '+name)
    if not {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}.issubset(metadata['files']):
        raise ValueError('Incomplete training state/adapter checkpoint')
    return metadata

def batch_plan(size, batch_size, accumulation, seed, epoch, smoke=False, lengths=None):
    if min(size, batch_size, accumulation) < 1:
        raise ValueError("Dataset and batch sizes must be positive")
    if smoke:
        if lengths is None or len(lengths) != size:
            raise ValueError("Smoke requires all sample lengths")
        return [[[max(range(size), key=lambda i: lengths[i])]]]
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, size, batch_size)]
    return [batches[i:i + accumulation] for i in range(0, len(batches), accumulation)]

def verify_fingerprint(expected, actual):
    if expected != actual:
        raise ValueError("Exact resume fingerprint mismatch; use a new output directory")

def validate_stop_epoch(stop_after_epoch, epochs, completed_epochs=0, smoke=False):
    if stop_after_epoch is None:
        return
    if smoke:
        raise ValueError("--stop-after-epoch is not supported with --smoke")
    if not 1 <= stop_after_epoch <= epochs:
        raise ValueError("--stop-after-epoch must be between 1 and --epochs")
    if stop_after_epoch <= completed_epochs:
        raise ValueError("--stop-after-epoch must be after the resumed completed epoch")

def epoch_outcome(completed_epochs, epochs, stop_after_epoch):
    if not 1 <= completed_epochs <= epochs:
        raise ValueError("Completed epoch outside configured training schedule")
    if stop_after_epoch is not None and completed_epochs > stop_after_epoch:
        raise ValueError("Training advanced beyond the requested stop epoch")
    if completed_epochs == epochs:
        return "complete"
    return "paused" if completed_epochs == stop_after_epoch else "running"

def collate(examples, pad_token_id, torch):
    width = max(len(x["input_ids"]) for x in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device="cuda:0"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda:0"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda:0")}

def run(args):
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.mode!='train':raise ValueError('CPU source checks use cpu_contract.py; this runner is train-only')
    contract.check_deadline(args.absolute_deadline_unix)
    if args.absolute_deadline_unix-time.time()>8400:raise ValueError('Absolute deadline exceeds authorized total budget')
    if os.environ.get('PYTORCH_ALLOC_CONF') or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):raise ValueError('Allocator configuration differs')
    output=Path(args.output_dir)
    if str(output)!='/dev/shm/patientphex-qwen35-fulltext32-1045-20261024/formal2ep':raise ValueError('Full state must stay in own fixed SHM namespace')
    if not args.resume and args.resume_export_ack:raise ValueError('Fresh run cannot consume any old ACK')
    if args.resume and not args.resume_export_ack:raise ValueError('Resume requires exact durable full-state ACK')
    if output.exists() and any(output.iterdir()) and not args.resume:raise ValueError('Fresh invocation needs an empty output; use explicit own-run resume')
    if args.resume and (args.mode!='train' or not (output/'run_manifest.json').is_file()):raise ValueError('Resume requires original formal run_manifest')
    output.mkdir(parents=True,exist_ok=True)
    started=time.time();stage='source_and_data';control=None;control_path=None
    def status(name):
        nonlocal stage
        stage=name;print(json.dumps({'stage':name,'mode':args.mode}),flush=True)
    try:
        status('source_and_data');rows,training_ids,excluded_ids=verify_sources_and_data(args)
        import numpy as np
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM,get_linear_schedule_with_warmup
        from peft import PeftModel
        if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':raise RuntimeError('Require isolated TF5.20.2/PEFT0.21.2')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        if torch.__version__!='2.7.1+cu126':raise RuntimeError('Exact A800 Torch runtime differs')
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        if sha256(q.__file__)!=MODEL_SOURCE_SHA or importlib.util.find_spec('fla') is not None:raise RuntimeError('Native Qwen runtime differs')
        if any(getattr(q,n).__module__!=q.__name__ or inspect.unwrap(getattr(q,n)).__module__!=q.__name__ for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']):raise RuntimeError('Native GDN dispatch differs')
        status('base_file_fingerprints')
        base_files=model_files(args.model,args.model_source_manifest,args.mode=='train')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='right')
        if tokenizer.pad_token_id is None:raise ValueError('Expected original tokenizer pad token')
        status('token_mask_audit');encoded=encode_all(rows,tokenizer,output,args.native_row_audit)
        # Native/selected-CE numeric contract already frozen; do not repeat it per epoch.
        text_config=Qwen3_5TextConfig.from_pretrained(args.model,local_files_only=True)
        text_config._attn_implementation='sdpa';text_config.use_cache=False
        packages={}
        for name in ['torch','transformers','peft','accelerate','numpy','tokenizers','huggingface-hub','safetensors','fla-core','flash-linear-attention','causal-conv1d']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,
            'torch_cuda':torch.version.cuda,'modeling_source_sha256':sha256(inspect.getsourcefile(Qwen3_5ForCausalLM)),
            'peft_source_sha256':sha256(inspect.getsourcefile(PeftModel))}
        if args.mode=='train':
            status('gpu_preflight')
            if not args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Sole CUDA_VISIBLE_DEVICES must equal allocated UUID')
            check=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
            write_json(output/'gpu-preflight.json',{'uuid':args.gpu_uuid,'processes':check.stdout})
            if any(line.split(',')[0].strip()==args.gpu_uuid for line in check.stdout.splitlines()):raise RuntimeError('Allocated GPU already occupied; refuse to share')
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():raise RuntimeError('Require one BF16 CUDA device')
            environment.update(gpu_name=torch.cuda.get_device_name(0),gpu_uuid=args.gpu_uuid,gpu_capability=list(torch.cuda.get_device_capability(0)))
            torch.cuda.manual_seed_all(CONFIG['seed'])
        fingerprint={'schema':'patientphex-qwen35-meta32-fulltext-selectedCE-lora-v1','configuration':semantic_configuration(args),
            'source_sha256':sha256(__file__),'helper_sha256':HELPER_SHA,'protocol_sha256':PROTOCOL_SHA,
            'copied_epoch_loop_source_sha256':OLD_TRAINER_SHA,'copied204_trainer_sha256':COPIED_TRAINER_SHA,
            'meta32_preparation_sha256':PREPARATION_SHA,'train_sha256':TRAIN_SHA,
            'excluded_input_sha256':EXCLUDED_SHA,'training_document_ids':training_ids,'excluded_document_ids':excluded_ids,
            'base_files':base_files,'environment':environment,'text_config':text_config.to_dict(),
            'encoded_sha256':helpers.json_sha(encoded),'fulltext_preparation_sha256':FULLTEXT_PREPARATION_SHA,'native_receipt_sha256':NATIVE_RECEIPT_SHA,'selected_loss_sha256':SELECTED_LOSS_SHA,'runtime_contract_sha256':RUNTIME_CONTRACT_SHA}
        # JSON round-trip normalizes config integer keys/tuples before comparing
        # a newly computed fingerprint with its persisted checkpoint form.
        fingerprint=json.loads(json.dumps(fingerprint,default=helpers.json_default,sort_keys=True))
        manifest_path=output/'run_manifest.json'
        if args.resume:verify_fingerprint(json.loads(manifest_path.read_text())['fingerprint'],fingerprint)
        manifest={'fingerprint':fingerprint,'examples':124,'max_tokens':13549,'total_tokens':1009181,'supervised_tokens':7170,
            'steps_per_epoch':16,'epochs':2,'total_optimizer_steps':32,'warmup_steps':2,'smoke_only':False,
            'resume_unit':'complete acknowledged epoch only; no automatic partial epoch replay','new_gpu_resume_numerical_equivalence_tested':False,
            'fresh_base_on_initial_run':True,'validation_labels_used':False}
        write_json(manifest_path,manifest)
        for name,source in [('train.py',__file__),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol)),('runtime_contract.py',contract.__file__)]:
            (output/name).write_bytes(Path(source).read_bytes())
        checkpoint=Path(args.resume) if args.resume else None;resumed_epochs=0;metadata=None
        if checkpoint:
            metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
            ack_receipt=contract.validate_export_ack(args.resume_export_ack,output,checkpoint,metadata,sha256(__file__))
            write_json(output/'resume-export-ack-verified.json',ack_receipt)
        contract.check_deadline(args.absolute_deadline_unix)
        validate_stop_epoch(args.stop_after_epoch,2,resumed_epochs)
        if (output/'training_complete.json').exists():raise ValueError('Run is already marked complete; refusing train/resume')
        controls=output/'run_controls';controls.mkdir(exist_ok=True);control_path=controls/f'{time.time_ns()}-{os.getpid()}.json'
        control={'resume':args.resume,'stop_after_epoch':args.stop_after_epoch,'started_completed_epochs':resumed_epochs,
            'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid()}
        write_json(control_path,control);(output/'paused.json').unlink(missing_ok=True)
        status('load_complete_text_base')
        base,loading=Qwen3_5ForCausalLM.from_pretrained(args.model,config=text_config,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
        write_json(output/'base-loading-info.json',loading)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):raise RuntimeError('Incomplete or mismatched base weights')
        if any(not key.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for key in loading.get('unexpected_keys',[])):raise RuntimeError('Unexpected text checkpoint keys')
        if type(base).__name__!='Qwen3_5ForCausalLM' or any('visual' in n for n,_ in base.named_modules()):raise RuntimeError('Wrong model class')
        if sum(p.numel() for p in base.parameters())!=protocol.BASE_PARAMETERS:raise RuntimeError('Exact base parameter count differs')
        import selected_loss
        import transformers.loss.loss_utils as native_loss
        if sha256(selected_loss.__file__)!=SELECTED_LOSS_SHA or sha256(native_loss.__file__)!=NATIVE_LOSS_SHA:raise RuntimeError('Selected/native CE source changed')
        if base.loss_function is not native_loss.ForCausalLMLoss:raise RuntimeError('Native loss dispatch differs')
        base.loss_function=selected_loss.selected_causal_loss
        if base.loss_function is not selected_loss.selected_causal_loss:raise RuntimeError('Instance loss hook did not bind')
        (output/'selected_loss.py').write_bytes(Path(selected_loss.__file__).read_bytes())
        base.config.use_cache=False
        if checkpoint:
            model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=True)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            coverage={'resumed_verified_epoch_adapter':str(checkpoint),'target_names':protocol.TARGETS}
        else:model,named,coverage=protocol.attach_adapter(base,torch,peft)
        if not named or any('.lora_A.' not in n and '.lora_B.' not in n for n in named):raise RuntimeError('Only LoRA may train')
        if sum(p.numel() for p in named.values())!=protocol.ADAPTER_PARAMETERS or sum(p.numel() for p in model.parameters())!=protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS or not sum(p.numel() for p in model.parameters())<10_000_000_000:raise RuntimeError('Exact adapter/total parameter count differs')
        if {protocol.family(n) for n in named}!={'DeltaNet','full_attention','MLP'}:raise RuntimeError('Missing hybrid adapter family')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        if not model.is_gradient_checkpointing:raise RuntimeError('Gradient checkpointing not enabled')
        write_json(output/'model-coverage.json',coverage)
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        scheduler=get_linear_schedule_with_warmup(optimizer,2,32)
        completed_epochs=global_step=0
        if checkpoint:
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            completed_epochs,global_step=state['completed_epochs'],state['global_step']
            if completed_epochs!=resumed_epochs or global_step!=16*completed_epochs:raise ValueError('Serialized epoch/step differs from checkpoint metadata')
            contract.validate_state_structure(state,resumed_epochs,list(named))
            if contract.tree_digest({n:p for n,p in named.items()},torch)!=metadata['trainable_parameter_digest']:raise ValueError('Resumed adapter tensor values differ')
            expected_state_digest=contract.tree_digest(state,torch)
            if expected_state_digest!=metadata['serialized_state_digest']:raise ValueError('Serialized full-state fingerprint differs')
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            if scheduler.last_epoch!=global_step:raise ValueError('Restored scheduler step differs')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
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
            del state,actual_state
        model.train();first_update_checked=False;status('formal_training')
        for epoch in range(completed_epochs,2):
            groups=batch_plan(len(encoded),1,8,CONFIG['seed'],epoch)
            if len(groups)!=16 or sum(len(g) for g in groups)!=124:raise RuntimeError('Frozen epoch batch/step accounting differs')
            loss_sum=0.;epoch_started=time.time()
            for group in groups:
                contract.check_deadline(args.absolute_deadline_unix)
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:probes[protocol.family(name)]=(name,p,p.detach().clone())
                for indices in group:
                    batch=collate([encoded[i] for i in indices],tokenizer.pad_token_id,torch)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
                    (loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)
                grad_norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['max_grad_norm'])
                if not bool(torch.isfinite(grad_norm)) or float(grad_norm)==0.:raise RuntimeError('Nonfinite/zero LoRA gradient')
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('Frozen base received a gradient')
                optimizer.step();scheduler.step();global_step+=1
                if probes:
                    if set(probes)!={'DeltaNet','full_attention','MLP'} or any(torch.equal(before,p.detach()) for _,p,before in probes.values()):raise RuntimeError('First nonzero-LR step did not change all three B-family probes')
                    write_json(output/'first-nonzero-update.json',{'step':global_step,'learning_rate':learning_rate,
                        'gradient_norm_before_clip':float(grad_norm),'changed_B_probes':{k:v[0] for k,v in probes.items()},'fresh_run':not bool(checkpoint)})
                    first_update_checked=True
                loss_sum+=group_loss
                if global_step%5==0:print(json.dumps({'epoch':epoch+1,'step':global_step,'loss':group_loss,'learning_rate':learning_rate,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}),flush=True)
            if not first_update_checked:raise RuntimeError('No nonzero-LR adapter update was verified')
            contract.check_deadline(args.absolute_deadline_unix)
            status('save_epoch_checkpoint')
            destination=output/f'epoch-{epoch+1:03d}';staging=output/f'.epoch-{epoch+1:03d}.incomplete'
            if destination.exists() or staging.exists():raise ValueError('Checkpoint destination/staging exists; no overwrite')
            staging.mkdir();model.save_pretrained(staging/'adapter',safe_serialization=True)
            if not all((staging/'adapter'/n).is_file() for n in ['adapter_model.safetensors','adapter_config.json']):raise RuntimeError('Adapter safe weights/config missing')
            state={'completed_epochs':epoch+1,'global_step':global_step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'optimizer_parameter_names':list(named)}
            torch.save(state,staging/'training_state.pt')
            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            write_json(staging/'checkpoint.json',{'fingerprint':fingerprint,'files':files,'completed_epochs':epoch+1,'global_step':global_step,'smoke_only':False,'new_resume_numerical_equivalence_tested':False,
                'serialized_state_digest':contract.tree_digest(state,torch),'trainable_parameter_digest':contract.tree_digest({n:p for n,p in named.items()},torch)})
            local_receipt=contract.local_publish_checkpoint(staging,destination,output)
            write_json(output/f'local-checkpoint-epoch-{epoch+1:03d}.json',local_receipt)
            contract.check_deadline(args.absolute_deadline_unix)
            write_json(output/'progress.json',{'completed_epochs':epoch+1,'global_step':global_step,'mean_accumulation_group_loss':loss_sum/len(groups),'epoch_seconds':time.time()-epoch_started,'checkpoint':str(destination),'training_only':True})
            outcome=epoch_outcome(epoch+1,2,args.stop_after_epoch);control.update(status=outcome,completed_epochs=epoch+1,global_step=global_step,checkpoint=str(destination),updated_unix_seconds=time.time());write_json(control_path,control)
            print(json.dumps({'status':outcome,'epoch':epoch+1,'global_step':global_step,'checkpoint':str(destination)}),flush=True)
            if outcome=='paused':
                write_json(output/'paused.json',{'completed_epochs':epoch+1,'configured_epochs':2,'global_step':global_step,'checkpoint':str(destination),'control_record':str(control_path),'result_type':'paused_complete_epoch_not_training_complete'});return
            status('formal_training')
        (output/'paused.json').unlink(missing_ok=True)
        write_json(output/'training_complete.json',{'smoke_only':False,'epochs':2,'global_step':global_step,'base_parameters':protocol.BASE_PARAMETERS,
            'trainable_adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),
            'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score_pending_second_epoch_persistent_ACK','new_resume_numerical_equivalence_tested':False,'persistent_export_ack_still_required':True})
        print(json.dumps({'complete':'training','epochs':2,'steps':global_step}),flush=True)
    except Exception as error:
        if control is not None:
            control.update(status='failed',failed_stage=stage,exception=type(error).__name__,message=str(error));write_json(control_path,control)
        write_json(output/f'failure-{time.time_ns()}.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),
            'source_sha256':sha256(__file__),'elapsed_seconds':time.time()-started,'training_completed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['train'],required=True)
    for name in ['train','excluded-input','data-manifest','fulltext-preparation','native-receipt','native-row-audit','model','model-source-manifest','output-dir']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    parser.add_argument('--resume-export-ack');parser.add_argument('--absolute-deadline-unix',type=float,required=True)
    args=parser.parse_args()
    validate_stop_epoch(args.stop_after_epoch,2)
    if (args.resume is None and args.stop_after_epoch!=1) or (args.resume is not None and args.stop_after_epoch is not None):parser.error('Fresh epoch1 must stop for ACK; epoch2 resumes complete state once')
    run(args)

if __name__=='__main__':main()
