#!/usr/bin/env python3
"""Fresh strict train32 JSON value2 SFT:124 rows,2 epochs,32 calls.
Original inputs/JSON/LoRA/batches retained; reviewed token-weighted CE only. No truncation. Final-only complete
checkpoint or recoverable partial at an optimizer boundary before2100s cap.
No old adapter/probe/resume is accepted by this initial-run entry.
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
import loss as weighted

HELPER_SHA='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
PROTOCOL_SHA='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
OLD_TRAINER_SHA='2a25b83d5d9e3dcc3f57cc8e2ff1aaf519a41a709e4ff289b2b26db9fab51423'
SOURCE_MANIFEST_SHA='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
CONFIG={'max_seq_length':6144,'batch_size':1,'gradient_accumulation':8,'epochs':2,
        'learning_rate':1e-4,'weight_decay':.01,'warmup_ratio':.05,'max_grad_norm':1.,
        'rank':16,'alpha':32,'dropout':.05,'seed':20261023,'targets':protocol.TARGETS,
        'dtype':'bfloat16','attention':'sdpa','quantization':False,'packing':False,
        'gradient_checkpointing':{'use_reentrant':False}}
TRAIN_SHA='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
EXCLUDED_SHA='2786daf5d7a09fea9d3c526b64f5cf972c0d7298e3a2a29d87d479055b4e5c82'
PREPARATION_SHA='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
WEIGHTS_SHA='342e2d258b84d5e7a898245eb83d94ae187629748c209e360212216a8f587acb'
LOSS_SHA='af2627e7823bcbf5051ba974554d8371bfafdd319309c12cb0245fd3b30c959e'
ORIGINAL_META32_TRAINER_SHA='08fc142c6b9a5d78603c2a1ad7545b733d66dff12f86d1bc1737a7dec0c27fe1'
COPIED_TRAINER_SHA='9d2e0eeb34e6bfa76ea90cbf6f9ffad996ab527e4746bb936e4f3cbc58724b8b'
sha256=helpers.sha
write_json=helpers.write


def semantic_configuration(args):
    return {'fixed':CONFIG,'paths_and_device':{k:v for k,v in vars(args).items()
            if k not in {'resume','stop_after_epoch'}}}


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
    if (preparation['training_sha256']!=TRAIN_SHA or preparation['excluded_input_sha256']!=EXCLUDED_SHA
        or preparation['training_document_ids']!=train_ids or preparation['excluded_document_ids']!=excluded_ids
        or not preparation['all_target_documents_excluded_from_own_generators']):
        raise ValueError('Meta32 provenance identity/isolation differs')
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


def encode_all(rows,tokenizer,output):
    encoded=[helpers.encode(row,tokenizer) for row in rows]
    lengths=[len(e['input_ids']) for e in encoded]
    if len(encoded)!=124 or max(lengths)!=5523 or sum(lengths)!=402594 or any(n>CONFIG['max_seq_length'] for n in lengths):
        raise ValueError('Frozen124 token accounting differs; no truncation permitted')
    if sum(e['supervised_tokens'] for e in encoded)!=7170:
        raise ValueError('Frozen supervised-token accounting differs')
    write_json(output/'token-mask-audit.json',{'rows':[{'row_id':e['row_id'],'full_tokens':len(e['input_ids']),
        'prefix_tokens':e['prefix_tokens'],'supervised_tokens':e['supervised_tokens']} for e in encoded],
        'rows_count':124,'total_full_tokens':sum(lengths),'max_full_tokens':max(lengths),
        'supervised_tokens':7170,'truncated_rows':0,'prefix_mask_eos_passed':True,
        'encoded_sha256':helpers.json_sha(encoded)})
    return encoded



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

def verify_weights(path,encoded,output):
    if sha256(path)!=WEIGHTS_SHA or sha256(inspect.getfile(weighted))!=LOSS_SHA:
        raise ValueError('Frozen value2 weights/loss differs')
    rows=json.loads(Path(path).read_text())
    if len(rows)!=124:raise ValueError('Weight row count differs')
    for e,w in zip(encoded,rows):
        expected_positions=[j-1 for j,label in enumerate(e['labels']) if label!=-100]
        expected_targets=[label for label in e['labels'] if label!=-100]
        if (e['row_id']!=w['row_id'] or len(e['input_ids'])!=w['full_tokens']
            or e['prefix_tokens']!=w['prefix_tokens'] or len(w['weights'])!=len(e['labels'])
            or hashlib.sha256(json.dumps(e['input_ids']).encode()).hexdigest()!=w['input_ids_sha256']
            or hashlib.sha256(json.dumps(e['labels']).encode()).hexdigest()!=w['labels_sha256']
            or w['positions']!=expected_positions or w['targets']!=expected_targets
            or any((weight!=0 if label==-100 else weight not in (1,2)) for label,weight in zip(e['labels'],w['weights']))
            or w['weights'][e['eos_supervised_position']]!=1):
            raise ValueError('Actual encoded row and reviewed weights disagree: '+e['row_id'])
    counts=Counter(value for w in rows for value in w['weights'])
    if counts[2]!=3220 or counts[1]!=3950 or sum(sum(w['weights']) for w in rows)!=10390:
        raise ValueError('Frozen value/EOS counts differ')
    groups=[batch_plan(124,1,8,CONFIG['seed'],epoch) for epoch in range(2)]
    if any([len(g) for g in gs]!=[8]*15+[4] or sorted(i[0] for g in gs for i in g)!=list(range(124)) for gs in groups):
        raise ValueError('Actual old124 batch schedule differs')
    write_json(output/'weighted-input-contract.json',{'weights_sha256':WEIGHTS_SHA,'loss_sha256':LOSS_SHA,
        'rows':124,'full_tokens':402594,'supervised':7170,'value_weight2_tokens':3220,'sum_token_coefficients':10390,
        'per_row_normalization':'sum(wCE)/sum(w), actual group mean8 or4',
        'original_ids_labels_and_EOS_exact':True,'batch_group_sizes':[[len(g) for g in gs] for gs in groups],
        'two_epochs_every_row_exactly_once_each':True,'no_quality':True})
    return rows


def run(args):
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.resume or args.stop_after_epoch is not None:raise ValueError('Initial weighted124 entry is fresh-only; partial recovery requires separate explicit authorization')
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
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM,get_linear_schedule_with_warmup
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
        weights=verify_weights(args.weights,encoded,output)
        protocol.causal_loss_probe(torch,output)
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
        fingerprint={'schema':'patientphex-qwen35-weighted124-json-relation-lora-v1','configuration':semantic_configuration(args),
            'source_sha256':sha256(__file__),'helper_sha256':HELPER_SHA,'protocol_sha256':PROTOCOL_SHA,
            'copied_epoch_loop_source_sha256':OLD_TRAINER_SHA,'copied204_trainer_sha256':COPIED_TRAINER_SHA,
            'data_preparation_sha256':PREPARATION_SHA,'weights_sha256':WEIGHTS_SHA,'loss_source_sha256':LOSS_SHA,'original_meta32_trainer_sha256':ORIGINAL_META32_TRAINER_SHA,'train_sha256':TRAIN_SHA,
            'excluded_input_sha256':EXCLUDED_SHA,'training_document_ids':training_ids,'excluded_document_ids':excluded_ids,
            'base_files':base_files,'environment':environment,'text_config':text_config.to_dict(),
            'encoded_sha256':helpers.json_sha(encoded)}
        # JSON round-trip normalizes config integer keys/tuples before comparing
        # a newly computed fingerprint with its persisted checkpoint form.
        fingerprint=json.loads(json.dumps(fingerprint,default=helpers.json_default,sort_keys=True))
        manifest_path=output/'run_manifest.json'
        if args.resume:verify_fingerprint(json.loads(manifest_path.read_text())['fingerprint'],fingerprint)
        manifest={'fingerprint':fingerprint,'examples':124,'max_tokens':5523,'total_tokens':402594,'supervised_tokens':7170,
            'steps_per_epoch':16,'epochs':2,'total_optimizer_steps':32,'warmup_steps':2,'smoke_only':False,
            'resume_unit':'saved optimizer-boundary state with next_epoch/group; this entry fresh-only; no automatic resume','new_gpu_resume_numerical_equivalence_tested':False,
            'fresh_base_on_initial_run':True,'validation_labels_used':False}
        write_json(manifest_path,manifest)
        for name,source in [('train_weighted.py',__file__),('loss.py',inspect.getfile(weighted)),('qwen35_smoke.py',inspect.getfile(helpers)),('qwen35_relation_cost.py',inspect.getfile(protocol))]:
            (output/name).write_bytes(Path(source).read_bytes())
        if args.mode=='cpu':
            status('meta_full_hybrid_coverage')
            with torch.device('meta'):
                base=Qwen3_5ForCausalLM(text_config).to(dtype=torch.bfloat16)
                model,named,coverage=protocol.attach_adapter(base,torch,peft)
            if coverage['all_parameter_devices']!=['meta'] or torch.cuda.is_initialized():raise RuntimeError('CPU audit initialized non-meta/GPU parameters')
            write_json(output/'meta-coverage.json',coverage)
            write_json(output/'cpu_complete.json',{'source_sha256':sha256(__file__),'manifest_sha256':sha256(manifest_path),
                'rows':124,'documents':32,'max_tokens':5523,'full_tokens':402594,'supervised_tokens':7170,
                'adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,
                'cuda_initialized':False,'elapsed_seconds':time.time()-started,'not_training':True})
            print(json.dumps({'complete':'cpu','cuda_initialized':False,'source_sha256':sha256(__file__)}),flush=True);return
        checkpoint=None;resumed_epochs=0
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
        base.config.use_cache=False
        model,named,coverage=protocol.attach_adapter(base,torch,peft)
        if not named or any('.lora_A.' not in n and '.lora_B.' not in n for n in named):raise RuntimeError('Only LoRA may train')
        if sum(p.numel() for p in named.values())!=protocol.ADAPTER_PARAMETERS or sum(p.numel() for p in model.parameters())!=protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS or not sum(p.numel() for p in model.parameters())<10_000_000_000:raise RuntimeError('Exact adapter/total parameter count differs')
        if {protocol.family(n) for n in named}!={'DeltaNet','full_attention','MLP'}:raise RuntimeError('Missing hybrid adapter family')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        if not model.is_gradient_checkpointing:raise RuntimeError('Gradient checkpointing not enabled')
        write_json(output/'model-coverage.json',coverage)
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        scheduler=get_linear_schedule_with_warmup(optimizer,2,32)
        completed_epochs=global_step=0
        def save_checkpoint(complete,next_epoch,next_group):
            name='epoch-002' if complete else f'partial-step-{global_step:03d}'
            destination=output/name;staging=output/('.'+name+'.incomplete')
            if destination.exists() or staging.exists():raise ValueError('No checkpoint overwrite')
            staging.mkdir();model.save_pretrained(staging/'adapter',safe_serialization=True)
            state={'completed_epochs':2 if complete else next_epoch,'global_step':global_step,
                'next_epoch_index':next_epoch,'next_group_index':next_group,
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,staging/'training_state.pt')
            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            sizes={str(p.relative_to(staging)):p.stat().st_size for p in sorted(staging.rglob('*')) if p.is_file()}
            write_json(staging/'checkpoint.json',{'fingerprint':fingerprint,'files':files,'file_sizes':sizes,
                'completed_epochs':2 if complete else next_epoch,'global_step':global_step,
                'next_epoch_index':next_epoch,'next_group_index':next_group,
                'final_fixed_epoch_only':complete,'partial_not_for_inference':not complete,
                'smoke_only':False,'new_resume_numerical_equivalence_tested':False})
            staging.replace(destination);return destination
        model.train();first_update_checked=False;status('formal_training')
        for epoch in range(completed_epochs,2):
            groups=batch_plan(len(encoded),1,8,CONFIG['seed'],epoch)
            if len(groups)!=16 or sum(len(g) for g in groups)!=124:raise RuntimeError('Frozen epoch batch/step accounting differs')
            loss_sum=0.;epoch_started=time.time()
            for group_index,group in enumerate(groups):
                elapsed=time.time()-started
                # Longest actual A100 weighted step:12.1805s/5523 tokens; slightly conservative per-token plus8s. No measured guarantee.
                next_estimate=sum(len(encoded[ix[0]]['input_ids']) for ix in group)*.0023+8
                if elapsed>=2040 or elapsed+next_estimate+60>=2100:
                    destination=save_checkpoint(False,epoch,group_index)
                    write_json(output/'training_terminal.json',{'status':'partial_budget_stop_not_for_inference',
                        'global_step':global_step,'next_epoch_index':epoch,'next_group_index':group_index,
                        'checkpoint':str(destination),'checkpoint_sha256':sha256(destination/'checkpoint.json'),
                        'elapsed_seconds':time.time()-started,'no_auto_resume':True,'no_quality':True})
                    control.update(status='partial_budget_stop_not_for_inference',global_step=global_step);write_json(control_path,control)
                    print('PARTIAL_SAVED_NO_INFERENCE',flush=True);return
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:probes[protocol.family(name)]=(name,p,p.detach().clone())
                for indices in group:
                    batch=collate([encoded[i] for i in indices],tokenizer.pad_token_id,torch)
                    token_weights=torch.tensor([weights[i]['weights'] for i in indices],dtype=torch.float32,device='cuda:0')
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=weighted.row_loss(model,batch,token_weights,torch)
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
                record={'epoch':epoch+1,'step':global_step,'group_rows':len(group),'loss':group_loss,'learning_rate':learning_rate,'elapsed_seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                with (output/'updates.jsonl').open('a') as trace:trace.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if not first_update_checked:raise RuntimeError('No nonzero-LR adapter update was verified')
            write_json(output/f'epoch-{epoch+1:03d}-summary.json',{'completed_epochs':epoch+1,'global_step':global_step,
                'mean_accumulation_group_loss':loss_sum/len(groups),'epoch_seconds':time.time()-epoch_started,
                'no_quality':True,'no_intermediate_checkpoint':True})
        if global_step!=32 or scheduler.last_epoch!=32 or optimizer.param_groups[0]['lr']!=0:raise RuntimeError('Fixed32-call schedule incomplete')
        destination=save_checkpoint(True,2,0)
        if time.time()-started>=2100:raise RuntimeError('Complete checkpoint exceeded hard budget')
        (output/'paused.json').unlink(missing_ok=True)
        write_json(output/'training_complete.json',{'smoke_only':False,'epochs':2,'global_step':global_step,'base_parameters':protocol.BASE_PARAMETERS,
            'trainable_adapter_parameters':protocol.ADAPTER_PARAMETERS,'total_parameters':protocol.BASE_PARAMETERS+protocol.ADAPTER_PARAMETERS,'elapsed_seconds':time.time()-started,
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'max_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(0),
            'source_sha256':sha256(__file__),'result_type':'training_only_no_evaluation_score','new_resume_numerical_equivalence_tested':False})
        write_json(output/'training_terminal.json',{'status':'complete_epoch002_32','checkpoint_sha256':sha256(destination/'checkpoint.json'),'complete_sha256':sha256(output/'training_complete.json')})
        control.update(status='complete_epoch002_32',global_step=global_step);write_json(control_path,control)
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
    for name in ['train','excluded-input','data-manifest','weights','model','model-source-manifest','output-dir']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid');parser.add_argument('--resume');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    validate_stop_epoch(args.stop_after_epoch,2)
    if args.mode=='cpu' and (args.resume or args.stop_after_epoch is not None):parser.error('CPU preparation does not pause/resume training')
    run(args)

if __name__=='__main__':main()
