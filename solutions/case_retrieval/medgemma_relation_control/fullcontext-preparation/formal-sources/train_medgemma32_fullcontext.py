#!/usr/bin/env python3
"""MedGemma1.5-4B append-only full-context control: same124 targets, fresh2 epochs.

CPU mode verifies data/template/loss/meta without weights or CUDA. Training mode
requires an allocated empty GPU and exact official files. No validation labels,
inference, quantization, packing, smoke adapter or alternative task is supported.
Fresh base is required; only this exact meta32 run may resume complete epochs.
Resume state is saved/restored; numerical GPU resume equivalence is not claimed.
Length14000 is a new full-context rejection cap; original trainer8192 remains unchanged: batches pad to their actual maximum, never to cap.
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
import medgemma_interface as interface
import train_medgemma32 as original_training
import fullcontext_training_contract as fullcontext

HELPER_SHA='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
PROTOCOL_SHA='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
OLD_TRAINER_SHA='2a25b83d5d9e3dcc3f57cc8e2ff1aaf519a41a709e4ff289b2b26db9fab51423'
SOURCE_MANIFEST_SHA='690e0829d0036901491dbca2f9f094cfecff4190eb023e03ad8a8cc611bdbfb1'
INTERFACE_SHA='aecef7eaab0565ab98ce76c034f01b543be3342b00a84d4bae71c36d614d55be'
CONFIG={'max_seq_length':14000,'batch_size':1,'gradient_accumulation':8,'epochs':2,
        'learning_rate':1e-4,'weight_decay':.01,'warmup_ratio':.05,'max_grad_norm':1.,
        'rank':16,'alpha':32,'dropout':.05,'seed':20261023,'targets':interface.TARGETS,
        'dtype':'bfloat16','attention':'sdpa','quantization':False,'packing':False,
        'gradient_checkpointing':{'use_reentrant':False}}
INFERENCE_SHA=fullcontext.FULL_INFER_SHA
TRAIN_SHA=fullcontext.proven.FULL_TRAIN_SHA
EXCLUDED_SHA='2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
PREPARATION_SHA='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
COPIED_TRAINER_SHA='9d2e0eeb34e6bfa76ea90cbf6f9ffad996ab527e4746bb936e4f3cbc58724b8b'
sha256=helpers.sha
write_json=helpers.write


def semantic_configuration(args):
    return {'fixed':CONFIG,'paths_and_device':{k:v for k,v in vars(args).items()
            if k not in {'resume','stop_after_epoch','cpu_reference'}}}


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
    return fullcontext.verify_training_sources(args,original_training,interface)


def model_files(model_dir,source_manifest,require_weights):
    return interface.model_files(model_dir,source_manifest,require_weights)


def encode_all(rows,tokenizer,output):
    encoded=[interface.encode(row,tokenizer) for row in rows]
    lengths=[len(e['input_ids']) for e in encoded]
    if len(encoded)!=124 or any(n>CONFIG['max_seq_length'] for n in lengths):
        raise ValueError('Require all124 rows within rejection cap; no truncation permitted')
    record={'rows':[{'row_id':e['row_id'],'full_tokens':len(e['input_ids']),
        'prefix_tokens':e['prefix_tokens'],'supervised_tokens':e['supervised_tokens']} for e in encoded],
        'rows_count':124,'total_full_tokens':sum(lengths),'max_full_tokens':max(lengths),
        'supervised_tokens':sum(e['supervised_tokens'] for e in encoded),'truncated_rows':0,
        'prefix_mask_eos_passed':True,'encoded_sha256':helpers.json_sha(encoded),
        'source_sha256':sha256(__file__),'interface_sha256':INTERFACE_SHA,
        'model_source_manifest_sha256':SOURCE_MANIFEST_SHA,'train_sha256':TRAIN_SHA}
    write_json(output/'token-mask-audit.json',record)
    return encoded


def validate_checkpoint(output,checkpoint,fingerprint):
    if checkpoint.resolve().parent!=output.resolve():raise ValueError('Resume checkpoint must belong to original output')
    if any(output.glob('.epoch-*.incomplete')):raise ValueError('Inspect incomplete checkpoint staging before resume')
    metadata=json.loads((checkpoint/'checkpoint.json').read_text())
    verify_fingerprint(metadata['fingerprint'],fingerprint)
    completed=int(metadata['completed_epochs']);expected=output/f'epoch-{completed:03d}'
    if checkpoint.resolve()!=expected.resolve() or checkpoint.name!=max(p.name for p in output.glob('epoch-*') if p.is_dir()):
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
    if args.mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    output=Path(args.output_dir)
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
        from transformers import AutoTokenizer,AutoConfig,Gemma3ForConditionalGeneration,get_linear_schedule_with_warmup
        from peft import PeftModel
        if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':raise RuntimeError('Require isolated TF5.20.2/PEFT0.21.2')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
        torch.backends.cudnn.benchmark=False
        status('base_file_fingerprints')
        base_files=model_files(args.model,args.model_source_manifest,args.mode=='train')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='right')
        if tokenizer.pad_token_id is None:raise ValueError('Expected original tokenizer pad token')
        status('token_mask_audit');encoded=encode_all(rows,tokenizer,output)
        if sha256(args.inference_prompts)!=INFERENCE_SHA:raise ValueError('Require fixed append-only full80 F1 input-only prompts')
        infer_rows=helpers.rows(args.inference_prompts);infer_ids=[r['row_id'] for r in infer_rows]
        if len(infer_rows)!=80 or len(set(infer_ids))!=80 or len({str(r['pmc_id']) for r in infer_rows})!=16:raise ValueError('Require unique80 F1 rows/16docs')
        if any('answer' in r or 'entities' in r or 'association' in r or str(r['pmc_id']) not in excluded_ids for r in infer_rows):raise ValueError('Inference inputs must be label-free and excluded from training')
        infer_encoded=[(r,interface.render_prefix(r,tokenizer)[1]) for r in infer_rows]
        fullcontext.verify_actual_inference_encoding(args,infer_encoded)
        if max(len(ids) for _,ids in infer_encoded)>14000:raise ValueError('No inference prompt truncation permitted')
        write_json(output/'inference-input-audit.json',{'input_sha256':INFERENCE_SHA,'rows':80,'documents':16,
            'total_tokens':sum(len(ids) for _,ids in infer_encoded),'max_tokens':max(len(ids) for _,ids in infer_encoded),
            'candidate_count':sum(len(r['code_map']) for r in infer_rows),'patient_candidate_pairs':sum(len(r['code_map'])*len(r['patient_ids']) for r in infer_rows),
            'token_ids_sha256':helpers.json_sha([{'row_id':r['row_id'],'input_ids':ids} for r,ids in infer_encoded]),
            'per_row':[{'row_id':r['row_id'],'tokens':len(ids)} for r,ids in infer_encoded],'no_gold':True,'no_truncation':True})
        schedule=[batch_plan(124,1,8,CONFIG['seed'],epoch) for epoch in range(2)]
        if any(len(groups)!=16 or [len(g) for g in groups]!=[8]*15+[4] for groups in schedule):raise ValueError('Exact original124x2 accumulation differs')
        write_json(output/'training-schedule.json',{'epochs':[[[rows[i]['row_id'] for i in batch] for g in groups for batch in g] for groups in schedule],
            'groups':[[[rows[i]['row_id'] for batch in g for i in batch] for g in groups] for groups in schedule],
            'optimizer_calls':32,'exposures':248,'per_epoch_calls':16,'tail_group_samples':4,'seed':CONFIG['seed']})
        protocol.causal_loss_probe(torch,output)
        native_greedy=interface.greedy_config(transformers);native_greedy.validate()
        write_json(output/'greedy-generation-config.json',native_greedy.to_dict())
        text_config=AutoConfig.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        if text_config.model_type!='gemma3' or text_config.text_config.model_type!='gemma3_text':raise ValueError('Require native Gemma3 multimodal config/text module')
        text_config._attn_implementation='sdpa';text_config.text_config._attn_implementation='sdpa';text_config.text_config.use_cache=False
        if text_config.text_config.use_bidirectional_attention or text_config.text_config.final_logit_softcapping is not None:raise ValueError('Unexpected native causal text/softcap config')
        if args.mode=='train':
            fullcontext.verify_actual_training_encoding(args,encoded,output,helpers)
        packages={}
        for name in ['torch','transformers','peft','accelerate','numpy','tokenizers','huggingface-hub','safetensors','fla-core','flash-linear-attention','causal-conv1d']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,
            'torch_cuda':torch.version.cuda,'modeling_source_sha256':sha256(inspect.getsourcefile(Gemma3ForConditionalGeneration)),
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
        fingerprint={'schema':'patientphex-medgemma15-fullbundle-text-relation-fullcontext-lora-v1','configuration':semantic_configuration(args),
            'source_sha256':sha256(__file__),'helper_sha256':HELPER_SHA,'protocol_sha256':PROTOCOL_SHA,'interface_sha256':INTERFACE_SHA,
            'copied_epoch_loop_source_sha256':OLD_TRAINER_SHA,'copied204_trainer_sha256':COPIED_TRAINER_SHA,
            'meta32_preparation_sha256':PREPARATION_SHA,'train_sha256':TRAIN_SHA,'inference_input_sha256':INFERENCE_SHA,
            'append_only_provenance':fullcontext.provenance(),
            'excluded_input_sha256':EXCLUDED_SHA,'training_document_ids':training_ids,'excluded_document_ids':excluded_ids,
            'base_files':base_files,'environment':environment,'text_config':text_config.to_dict(),
            'encoded_sha256':helpers.json_sha(encoded)}
        # JSON round-trip normalizes config integer keys/tuples before comparing
        # a newly computed fingerprint with its persisted checkpoint form.
        fingerprint=json.loads(json.dumps(fingerprint,default=helpers.json_default,sort_keys=True))
        manifest_path=output/'run_manifest.json'
        if args.resume:verify_fingerprint(json.loads(manifest_path.read_text())['fingerprint'],fingerprint)
        manifest={'fingerprint':fingerprint,'examples':124,'max_tokens':max(len(e['input_ids']) for e in encoded),'total_tokens':sum(len(e['input_ids']) for e in encoded),'supervised_tokens':sum(e['supervised_tokens'] for e in encoded),
            'steps_per_epoch':16,'epochs':2,'total_optimizer_steps':32,'warmup_steps':2,'smoke_only':False,
            'resume_unit':'complete epoch only; partial epoch work must replay','new_gpu_resume_numerical_equivalence_tested':False,
            'fresh_base_on_initial_run':True,'validation_labels_used':False}
        write_json(manifest_path,manifest)
        for name,source in [('train_medgemma32_fullcontext.py',__file__),('train_medgemma32.py',inspect.getfile(original_training)),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol)),('medgemma_interface.py',inspect.getfile(interface))]+fullcontext.source_snapshot():
            (output/name).write_bytes(Path(source).read_bytes())
        if args.mode=='cpu':
            status('meta_full_bundle_language_only_coverage')
            with torch.device('meta'):
                base=Gemma3ForConditionalGeneration(text_config).to(dtype=torch.bfloat16)
                model,named,coverage=interface.attach_adapter(base,torch,peft)
            if coverage['all_parameter_devices']!=['meta'] or torch.cuda.is_initialized():raise RuntimeError('CPU audit initialized non-meta/GPU parameters')
            write_json(output/'meta-coverage.json',coverage)
            write_json(output/'cpu_complete.json',{'source_sha256':sha256(__file__),'manifest_sha256':sha256(manifest_path),
                'rows':124,'documents':32,'max_tokens':manifest['max_tokens'],'full_tokens':manifest['total_tokens'],'supervised_tokens':manifest['supervised_tokens'],
                'adapter_parameters':interface.ADAPTER_PARAMETERS,'total_parameters':interface.BASE_PARAMETERS+interface.ADAPTER_PARAMETERS,
                'cuda_initialized':False,'elapsed_seconds':time.time()-started,'not_training':True})
            print(json.dumps({'complete':'cpu','cuda_initialized':False,'source_sha256':sha256(__file__)}),flush=True);return
        checkpoint=Path(args.resume) if args.resume else None;resumed_epochs=0;metadata=None
        if checkpoint:metadata=validate_checkpoint(output,checkpoint,fingerprint);resumed_epochs=int(metadata['completed_epochs'])
        validate_stop_epoch(args.stop_after_epoch,2,resumed_epochs)
        if (output/'training_complete.json').exists():raise ValueError('Run is already marked complete; refusing train/resume')
        controls=output/'run_controls';controls.mkdir(exist_ok=True);control_path=controls/f'{time.time_ns()}-{os.getpid()}.json'
        control={'resume':args.resume,'stop_after_epoch':args.stop_after_epoch,'started_completed_epochs':resumed_epochs,
            'configured_epochs':2,'status':'running','started_unix_seconds':time.time(),'pid':os.getpid()}
        write_json(control_path,control);(output/'paused.json').unlink(missing_ok=True)
        status('load_complete_native_bundle_text_only_forward')
        base,loading=Gemma3ForConditionalGeneration.from_pretrained(args.model,config=text_config,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',generation_config=native_greedy,local_files_only=True,trust_remote_code=False,output_loading_info=True)
        write_json(output/'base-loading-info.json',loading)
        if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):raise RuntimeError('Require complete native full bundle with no unaccounted checkpoint keys')
        if type(base).__name__!='Gemma3ForConditionalGeneration':raise RuntimeError('Wrong native full bundle class')
        if sum(p.numel() for p in base.parameters())!=interface.BASE_PARAMETERS:raise RuntimeError('Exact native full bundle parameter count differs')
        base.config.text_config.use_cache=False
        vision_guards=interface.forbid_vision_forward(base)
        if checkpoint:
            model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=True)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            coverage={'resumed_verified_epoch_adapter':str(checkpoint),'target_names':interface.TARGETS}
        else:model,named,coverage=interface.attach_adapter(base,torch,peft)
        if not named or any('.lora_A.' not in n and '.lora_B.' not in n for n in named):raise RuntimeError('Only LoRA may train')
        if sum(p.numel() for p in named.values())!=interface.ADAPTER_PARAMETERS or sum(p.numel() for p in model.parameters())!=interface.BASE_PARAMETERS+interface.ADAPTER_PARAMETERS or not sum(p.numel() for p in model.parameters())<10_000_000_000:raise RuntimeError('Exact adapter/total parameter count differs')
        if {interface.family(n) for n in named}!={'attention','MLP'}:raise RuntimeError('Missing language adapter family')
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
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            if scheduler.last_epoch!=global_step:raise ValueError('Restored scheduler step differs')
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        model.train();first_update_checked=False;status('formal_training')
        for epoch in range(completed_epochs,2):
            groups=batch_plan(len(encoded),1,8,CONFIG['seed'],epoch)
            if len(groups)!=16 or sum(len(g) for g in groups)!=124:raise RuntimeError('Frozen epoch batch/step accounting differs')
            loss_sum=0.;epoch_started=time.time()
            for group in groups:
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and interface.family(name) not in probes:probes[interface.family(name)]=(name,p,p.detach().clone())
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
                    if set(probes)!={'attention','MLP'} or any(torch.equal(before,p.detach()) for _,p,before in probes.values()):raise RuntimeError('First nonzero-LR step did not change both language B-family probes')
                    write_json(output/'first-nonzero-update.json',{'step':global_step,'learning_rate':learning_rate,
                        'gradient_norm_before_clip':float(grad_norm),'changed_B_probes':{k:v[0] for k,v in probes.items()},'fresh_run':not bool(checkpoint)})
                    first_update_checked=True
                loss_sum+=group_loss
                if global_step%5==0:print(json.dumps({'epoch':epoch+1,'step':global_step,'loss':group_loss,'learning_rate':learning_rate,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}),flush=True)
            if not first_update_checked:raise RuntimeError('No nonzero-LR adapter update was verified')
            status('save_epoch_checkpoint')
            destination=output/f'epoch-{epoch+1:03d}';staging=output/f'.epoch-{epoch+1:03d}.incomplete'
            if destination.exists() or staging.exists():raise ValueError('Checkpoint destination/staging exists; no overwrite')
            staging.mkdir();model.save_pretrained(staging/'adapter',safe_serialization=True)
            for notice in ['LICENSE.HAI-DEF.html','PROHIBITED-USE-POLICY.HAI-DEF.html','NOTICE.txt']:(staging/'adapter'/notice).write_bytes((Path(args.model)/notice).read_bytes())
            (staging/'adapter'/'MODIFICATIONS.txt').write_text('Text-language LoRA adapted on PatientPheX strict training32; base weights unchanged. Source and training scope are recorded in checkpoint.json.\n')
            if not all((staging/'adapter'/n).is_file() for n in ['adapter_model.safetensors','adapter_config.json']):raise RuntimeError('Adapter safe weights/config missing')
            state={'completed_epochs':epoch+1,'global_step':global_step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,staging/'training_state.pt')
            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            write_json(staging/'checkpoint.json',{'fingerprint':fingerprint,'files':files,'completed_epochs':epoch+1,'global_step':global_step,'smoke_only':False,'new_resume_numerical_equivalence_tested':False})
            for artifact in staging.rglob('*'):
                if artifact.is_file():
                    with artifact.open('rb') as saved:os.fsync(saved.fileno())
            for folder in [staging/'adapter',staging]:
                fd=os.open(str(folder),os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            staging.replace(destination)
            fd=os.open(str(output),os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            write_json(output/'progress.json',{'completed_epochs':epoch+1,'global_step':global_step,'mean_accumulation_group_loss':loss_sum/len(groups),'epoch_seconds':time.time()-epoch_started,'checkpoint':str(destination),'training_only':True})
            outcome=epoch_outcome(epoch+1,2,args.stop_after_epoch);control.update(status=outcome,completed_epochs=epoch+1,global_step=global_step,checkpoint=str(destination),updated_unix_seconds=time.time());write_json(control_path,control)
            print(json.dumps({'status':outcome,'epoch':epoch+1,'global_step':global_step,'checkpoint':str(destination)}),flush=True)
            if outcome=='paused':
                write_json(output/'paused.json',{'completed_epochs':epoch+1,'configured_epochs':2,'global_step':global_step,'checkpoint':str(destination),'control_record':str(control_path),'result_type':'paused_complete_epoch_not_training_complete'});return
            status('formal_training')
        (output/'paused.json').unlink(missing_ok=True)
        write_json(output/'training_complete.json',{'smoke_only':False,'epochs':2,'global_step':global_step,'base_parameters':interface.BASE_PARAMETERS,
            'trainable_adapter_parameters':interface.ADAPTER_PARAMETERS,'total_parameters':interface.BASE_PARAMETERS+interface.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),
            'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False})
        print(json.dumps({'complete':'training','epochs':2,'steps':global_step}),flush=True)
    except Exception as error:
        if control is not None:
            control.update(status='failed',failed_stage=stage,exception=type(error).__name__,message=str(error));write_json(control_path,control)
        write_json(output/f'failure-{time.time_ns()}.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),
            'source_sha256':sha256(__file__),'elapsed_seconds':time.time()-started,'training_completed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','train'],required=True)
    for name in ['train','original-train','train-input','excluded-input','data-manifest','fullcontext-manifest','model','model-source-manifest','output-dir','inference-prompts','original-inference-prompts','inference-input','cpu-row-audit']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--cpu-reference',required=True)
    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    validate_stop_epoch(args.stop_after_epoch,2)
    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):parser.error('CPU preparation does not pause/resume training')
    run(args)

if __name__=='__main__':main()
