#!/usr/bin/env python3
"""DRAFT: fresh grounded whole-report LoRA, 87 rows x 2; no execution budget here.

The parent must bind actual native-CPU and engineering receipts, freeze sources
and authorize a separate formal plan. This entry has no smoke, resume, inference,
quality, tokenization, answer rewrite, or alternate loss mode.
"""
import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

INDEXED_SHA='0d28201832721c800696321c6afa1ef7831d63b697620a02290d65317a4428bb'
MODEL_SOURCE_SHA='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
NATIVE_LOSS_SHA='83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe'
TRAIN_SHA='bf0091ef057ddb67f9906d035fdaf0ac7b41ae645f77432f0e5a007078291d08'
AUDIT_SHA='765316d8a22f554d7b590e9bc25de4efa10dee3cd0212bc3b530ce995ec3789f'
MODEL_MANIFEST_SHA='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
TARGET_MANIFEST_SHA='85979cdcafb37ddc3eef6b8adac864d0a50608c5ed2c917ac22db4be4643d095'
MODULE_SHAS={
    'indexed_head':INDEXED_SHA,
    'slot_model':'e8baf724a8aaf31d7686e53586033fdf0b07290d5f92f43e5617e436d1473647',
    'qwen35_runner':'33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669',
    'qwen35_relation_cost':'a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c',
    'qwen35_smoke':'9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3',
}
CONFIG={'rows':87,'epochs':2,'steps_per_epoch':11,'optimizer_calls':22,'warmup_steps':2,
    'batch_size':1,'gradient_accumulation':8,'tail_group_size':7,'seed':20261025,
    'learning_rate':1e-4,'weight_decay':.01,'max_grad_norm':1.,'rank':16,'lora_alpha':32,'dropout':.05,
    'dtype':'bfloat16','attention':'sdpa','gradient_checkpointing':{'use_reentrant':False},
    'packing':False,'quantization':False,'loss':'standard per-row mean causal CE; divide each row by actual group size',
    'full_training_tokens':451672,'supervised_tokens':11884,'maximum_full_tokens':9210}
ASSET_NAMES={'trainer','encoded','train_audit','target_manifest','model_manifest',
    'cpu_result','cpu_runtime','cpu_plan','native_cpu_source','probe_complete','probe_step','probe_manifest'}|set(MODULE_SHAS)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))


def identity(value):return {key:value[key] for key in ('bytes','sha256')}
def read(path):return json.loads(Path(path).read_text())
def json_sha(value):return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def json_default(value):
    if isinstance(value,(set,frozenset)):return sorted(value,key=repr)
    raise TypeError('Unsupported metadata container: '+type(value).__name__)


def sync_ancestors(directory):
    directory=Path(directory).resolve()
    while True:
        descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if directory.parent==directory:break
        directory=directory.parent


def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with temporary.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=json_default);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    temporary.replace(path);sync_ancestors(path.parent)


def check_deadline(deadline):
    if time.time()>=deadline:raise TimeoutError('Original formal absolute deadline reached; preserve complete epochs')


def load_module(name,item):
    assert ref(item['path'])==item
    if name in sys.modules:raise RuntimeError('Unexpected preloaded dependency: '+name)
    spec=importlib.util.spec_from_file_location(name,item['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def verify_inputs(plan):
    assert plan['schema']=='grounded-whole-report-formal87-v1' and plan['formal_training_authorized'] is True
    assert isinstance(plan['budget_seconds'],int) and plan['budget_seconds']>0
    assert plan['grace_seconds']==30 and plan['configuration']==CONFIG
    assets=plan['assets'];assert set(assets)==ASSET_NAMES
    for name,item in assets.items():assert ref(item['path'])==item,name
    assert assets['trainer']==ref(__file__)
    for name,sha in MODULE_SHAS.items():assert assets[name]['sha256']==sha,name
    assert assets['train_audit']['sha256']==AUDIT_SHA
    assert assets['target_manifest']['sha256']==TARGET_MANIFEST_SHA
    assert assets['model_manifest']['sha256']==MODEL_MANIFEST_SHA
    cpu=read(assets['cpu_result']['path']);cpu_plan=read(assets['cpu_plan']['path'])
    assert cpu['status']=='PASS_NATIVE_FULL_INPUT_AND_INDEXED_POSITION_CPU'
    assert cpu['train_rows']==87 and cpu['held_rows']==39 and len(cpu['numerical'])==6
    assert cpu['all_local_native_actual_IDs_labels_rendering_exact'] and cpu['complete_inputs_and_answers']
    assert identity(cpu['encoded'])==identity(assets['encoded'])
    assert identity(cpu['runtime'])==identity(assets['cpu_runtime'])
    assert identity(cpu['plan'])==identity(assets['cpu_plan'])
    assert identity(cpu['source'])==identity(assets['native_cpu_source'])
    assert cpu_plan['files'][cpu_plan['train']]['sha256']==TRAIN_SHA
    assert cpu_plan['files'][cpu_plan['train_audit']]['sha256']==AUDIT_SHA
    runtime=read(assets['cpu_runtime']['path'])
    assert runtime['torch']=='2.6.0+cu124' and runtime['transformers']=='5.20.2'
    assert runtime['peft']=='0.21.2' and runtime['tokenizers']=='0.23.1'
    assert runtime['modeling']['sha256']==MODEL_SOURCE_SHA and runtime['indexed_head']['sha256']==INDEXED_SHA
    probe=read(assets['probe_complete']['path']);step=read(assets['probe_step']['path'])
    probe_manifest=read(assets['probe_manifest']['path'])
    assert probe['step_sha256']==assets['probe_step']['sha256'] and probe['manifest_sha256']==assets['probe_manifest']['sha256']
    assert probe['actual_steps']==1 and probe['actual_backwards']==14 and probe['fresh_state_discarded'] and probe['no_checkpoint']
    assert step['status']=='PASS_indexed_head_local_contract_then_fresh_group8_update_three_accumulated_plus_three_Pareto_no_quality'
    assert probe['status']==step['status'] and step['actual_optimizer_steps']==1 and step['actual_backwards']==14
    assert step['all_three_Pareto_completed'] and step['no_checkpoint'] and step['no_input_truncation']
    assert probe_manifest['head_index_strategy']=='full backbone then supervised causal positions in BF16 head, external unchanged per-row CE'
    assert probe_manifest['no_saved_adapter_loaded'] and probe_manifest['base_frozen']
    # The native artifact contains held input IDs too; only its original train
    # list is used, and no held targets or document annotations are loaded here.
    payload=json.loads(gzip.decompress(Path(assets['encoded']['path']).read_bytes()))
    encoded=payload['train'];audit=read(assets['train_audit']['path'])
    assert len(encoded)==len(audit)==87 and len({r['row_id'] for r in encoded})==87
    assert len({str(r['pmc_id']) for r in audit})==32
    for index,(row,expected) in enumerate(zip(encoded,audit)):
        assert row['row_id']==expected['row_id'] and expected['index']==index
        ids=row['input_ids'];labels=row['labels'];prefix=row['prefix_tokens'];supervised=row['supervised_tokens']
        assert isinstance(ids,list) and all(type(v) is int and v>=0 for v in ids)
        assert len(ids)==len(labels)==expected['full_tokens'] and 0<prefix<len(ids)
        assert labels==[-100]*prefix+ids[prefix:] and len(ids)-prefix==supervised==expected['supervised_tokens']
        assert json_sha(ids)==expected['input_ids_sha256'] and json_sha(labels)==expected['labels_sha256']
        assert row['rendered_full_sha256']==expected['rendered_full_sha256']
        assert row['rendered_prefix_sha256']==expected['rendered_prefix_sha256']
        positions=[q-1 for q,value in enumerate(labels) if value!=-100]
        assert min(positions)>=0 and len(positions)==supervised and json_sha(positions)==expected['predecessor_positions_sha256']
        assert row['eos_token_id']==248046 and labels[prefix:].count(248046)==1
        assert labels[row['eos_supervised_position']]==248046
    assert sum(len(r['input_ids']) for r in encoded)==451672 and max(len(r['input_ids']) for r in encoded)==9210
    assert sum(r['supervised_tokens'] for r in encoded)==11884
    return encoded,audit


def batch_plan(epoch):
    indices=list(range(87));random.Random(20261025+epoch).shuffle(indices)
    groups=[indices[i:i+8] for i in range(0,87,8)]
    assert len(groups)==11 and [len(g) for g in groups]==[8]*10+[7]
    assert sorted(i for group in groups for i in group)==list(range(87))
    return groups


def durability_fsync_directory(path):
    descriptor=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def durability_inventory(root):
    members={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Checkpoint symlinks are forbidden')
        if path.is_file():members[path.relative_to(root).as_posix()]=path
        elif not path.is_dir():raise ValueError('Checkpoint has a nonregular member')
    return members


def durable_publish_checkpoint(output,staging,destination):
    # No state deserialization, optimizer/RNG calls or training policy changes.
    output=Path(output);staging=Path(staging);destination=Path(destination)
    if not output.is_dir() or output.is_symlink():raise ValueError('Require own existing checkpoint output directory')
    persistent_root=Path(os.environ['PATIENTPHEX_DURABILITY_ROOT']).resolve()
    if not persistent_root.is_dir() or persistent_root==output.resolve():raise ValueError('Require explicit existing persistent ancestor root')
    output.resolve().relative_to(persistent_root)

    if staging.is_symlink() or destination.is_symlink():raise ValueError('Checkpoint paths must not be symlinks')
    if staging.parent.resolve()!=output.resolve() or destination.parent.resolve()!=output.resolve():
        raise ValueError('Checkpoint stage and destination must share the own output parent')
    if destination.exists():raise ValueError('Checkpoint destination exists; no overwrite')
    if not staging.is_dir():raise ValueError('Checkpoint staging directory is missing')
    metadata_path=staging/'checkpoint.json'
    if metadata_path.is_symlink():raise ValueError('Checkpoint metadata must not be a symlink')
    metadata_raw=metadata_path.read_bytes();metadata=json.loads(metadata_raw)
    completed=metadata['completed_epochs']
    if type(completed) is not int or completed not in (1,2) or metadata['global_step']!=11*completed or metadata.get('smoke_only') is not False:
        raise ValueError('Require grounded87 complete-epoch checkpoint accounting')
    if destination.name!=f'epoch-{completed:03d}' or staging.name!=f'.epoch-{completed:03d}.incomplete':
        raise ValueError('Checkpoint epoch/staging namespace differs')
    receipt_path=output/f'durability-epoch-{completed:03d}.json'
    if receipt_path.exists() or receipt_path.is_symlink():raise ValueError('Durability receipt exists; no overwrite')
    source_sha=sha256(__file__)
    if metadata['fingerprint']['source_sha256']!=source_sha:
        raise ValueError('Checkpoint does not belong to this exact trainer source')
    expected=metadata['files']
    if not isinstance(expected,dict) or not expected:raise ValueError('Checkpoint member digest map is empty')
    expected=dict(expected)
    for name,digest in expected.items():
        relative=Path(name)
        if (not isinstance(name,str) or relative.is_absolute() or '..' in relative.parts
            or relative.as_posix()!=name or name=='checkpoint.json'
            or not isinstance(digest,str) or re.fullmatch(r'[0-9a-f]{64}',digest) is None):
            raise ValueError('Unsafe or invalid checkpoint member digest')
    if not {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(expected):
        raise ValueError('Checkpoint lacks full training state or adapter members')
    expected['checkpoint.json']=hashlib.sha256(metadata_raw).hexdigest()
    members=durability_inventory(staging)
    if set(members)!=set(expected):raise ValueError('Checkpoint member inventory differs before publish')
    # Existing writers have closed their files. Flush every member, then all
    # directories deepest-first (including the stage itself) before rename.
    for path in members.values():
        with path.open('rb') as handle:os.fsync(handle.fileno())
    directories=[path for path in staging.rglob('*') if path.is_dir()]
    for path in sorted(directories,key=lambda p:(-len(p.parts),str(p))):durability_fsync_directory(path)
    durability_fsync_directory(staging)
    if destination.exists() or receipt_path.exists():raise ValueError('Checkpoint destination/receipt appeared; refuse overwrite')
    staging.replace(destination)
    # Persist every newly-created ancestor directory entry through the explicit
    # project filesystem root before acknowledging this complete epoch.
    synced_ancestors=[];ancestor=output.resolve()
    while True:
        durability_fsync_directory(ancestor);synced_ancestors.append(str(ancestor))
        if ancestor==persistent_root:break
        ancestor=ancestor.parent
    actual_members=durability_inventory(destination)
    if set(actual_members)!=set(expected):raise ValueError('Checkpoint member inventory differs after publish')
    reopened={}
    for name,path in actual_members.items():
        actual_sha=sha256(path)
        if actual_sha!=expected[name]:raise ValueError('Reopened durable checkpoint SHA mismatch: '+name)
        reopened[name]={'bytes':path.stat().st_size,'sha256':actual_sha}
    # A failed post-rename hash leaves the checkpoint intact with NO success
    # receipt. Recovery must preserve this failure, never relabel it passed.
    receipt={'status':'PASS_file_and_directory_fsync_rename_and_reopened_SHA',
        'completed_epochs':completed,'global_step':metadata['global_step'],
        'checkpoint':str(destination),'source_sha256':source_sha,
        'checkpoint_metadata_sha256':expected['checkpoint.json'],'files':reopened,
        'file_directory_fsync':True,'reopened_SHA_all_exact':True,
        'persistent_root':str(persistent_root),'ancestor_directories_fsynced':synced_ancestors,
        'at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'all_members_reopened_and_SHA_verified':True,'file_fsync_count':len(members),
        'checkpoint_directory_fsync_count':len(directories)+1,
        'parent_directory_fsync_after_rename':True,'receipt_parent_directory_fsync_required':True,
        'no_state_deserialization':True,'no_rng_or_training_math_calls':True,
        'completed_unix_seconds':time.time()}
    with receipt_path.open('x',encoding='utf-8') as handle:
        json.dump(receipt,handle,ensure_ascii=False,indent=2);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    durability_fsync_directory(output)
    return receipt


def run(args):
    assert __debug__
    started=time.monotonic();plan=read(args.plan);encoded,audit=verify_inputs(plan)
    check_deadline(args.absolute_deadline_unix)
    assets=plan['assets'];output=Path(plan['output_dir']);persistent=Path(plan['storage_parent']).resolve()
    assert output.is_absolute() and not output.exists() and persistent.is_dir()
    output.resolve().relative_to(persistent);assert output.resolve()!=persistent
    assert os.environ['PATIENTPHEX_DURABILITY_ROOT']==str(persistent)
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
    assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
    output.mkdir(parents=True,exist_ok=False);sync_ancestors(output.parent)
    stage='runtime';control=dict(resume=None,started_completed_epochs=0,configured_epochs=2,
        global_step=0,status='running',pid=os.getpid(),started_unix_seconds=time.time())
    write_json(output/'run-control.json',control)
    try:
        import numpy as np
        import torch,transformers,peft,tokenizers
        import transformers.loss.loss_utils as native_loss
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
        from transformers import get_linear_schedule_with_warmup
        assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='5.20.2'
        assert peft.__version__=='0.21.2' and tokenizers.__version__=='0.23.1'
        assert sha256(native_loss.__file__)==NATIVE_LOSS_SHA and sha256(modeling.__file__)==MODEL_SOURCE_SHA
        assert importlib.util.find_spec('fla') is None
        bindings={name:dict(module=getattr(modeling,name).__module__,unwrapped=inspect.unwrap(getattr(modeling,name)).__module__)
            for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']}
        write_json(output/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            peft=peft.__version__,tokenizers=tokenizers.__version__,cuda=torch.version.cuda,native_bindings=bindings,
            native_loss_source=ref(native_loss.__file__),modeling_source=ref(modeling.__file__)))
        assert all(v['module']==v['unwrapped']==modeling.__name__ for v in bindings.values())
        modules={name:load_module(name,assets[name]) for name in
            ['qwen35_smoke','qwen35_relation_cost','qwen35_runner','slot_model','indexed_head']}
        loader=modules['slot_model'];protocol=modules['qwen35_relation_cost'];indexed=modules['indexed_head']
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        stage='fresh_public_base_and_fresh_LoRA';check_deadline(args.absolute_deadline_unix)
        model,named,loaded=loader.load_fresh(plan['model'],assets['model_manifest']['path'],20261025)
        base=model.get_base_model();assert base.loss_function is native_loss.ForCausalLMLoss
        assert not any(p.requires_grad for p in base.lm_head.parameters())
        assert loaded['fresh_public_base'] and loaded['no_saved_adapter_loaded']
        assert sum(p.numel() for p in named.values())==43278336
        assert sum(p.numel() for p in model.parameters())==8997081600<10_000_000_000
        assert {id(p) for p in named.values()}=={id(p) for p in model.parameters() if p.requires_grad}
        assert all('.lora_A.' in n or '.lora_B.' in n for n in named)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        assert model.is_gradient_checkpointing
        # Same helper and real forward/loss call as the new route's engineering
        # probe; no local head numeric contract is run again during formal training.
        fingerprint=dict(schema='grounded-whole-report87-standard-indexed-CE-v1',source_sha256=sha256(__file__),
            configuration=CONFIG,train_sha256=TRAIN_SHA,encoded_sha256=assets['encoded']['sha256'],
            input_ids_labels_native_verified=True,train_document_ids=sorted({str(r['pmc_id']) for r in audit}),
            assets=assets,indexed_head_sha256=INDEXED_SHA,native_loss_sha256=NATIVE_LOSS_SHA,
            runtime=loaded['runtime'],model_files=loaded['model_files'],fresh_public_base=True,no_saved_adapter_loaded=True,
            objective='standard per-row supervised mean CE; actual group average; no alpha05 patient/value weighting')
        fingerprint=json.loads(json.dumps(fingerprint,sort_keys=True,default=json_default))
        write_json(output/'run_manifest.json',dict(fingerprint=fingerprint,loaded=loaded,examples=87,epochs=2,
            steps_per_epoch=11,total_optimizer_steps=22,warmup_steps=2,full_tokens=451672,supervised_tokens=11884,
            max_tokens=9210,base_head_frozen=True,labels_None_in_model_forward=True,FP32_selected_CE=True,
            alpha05_loss_used=False,retokenized=False,validation_labels_used=False,no_automatic_resume=True))
        write_json(output/'epoch-groups.json',{str(epoch+1):batch_plan(epoch) for epoch in range(2)})
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,2,22)
        assert scheduler.last_epoch==0 and optimizer.param_groups[0]['lr']==0
        model.train();first_update_checked=False;global_step=0
        for epoch in range(2):
            groups=batch_plan(epoch);epoch_started=time.monotonic();loss_sum=0.
            for group_index,group in enumerate(groups):
                check_deadline(args.absolute_deadline_unix);stage='epoch_%d_group_%02d'%(epoch+1,group_index)
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={}
                if not first_update_checked and learning_rate>0:
                    for name,p in named.items():
                        if '.lora_B.' in name and protocol.family(name) not in probes:
                            probes[protocol.family(name)]=(name,p,p.detach().clone())
                for original_index in group:
                    row=encoded[original_index]
                    batch={'input_ids':torch.tensor([row['input_ids']],device='cuda',dtype=torch.long),
                           'labels':torch.tensor([row['labels']],device='cuda',dtype=torch.long)}
                    batch['attention_mask']=torch.ones_like(batch['input_ids'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=indexed.indexed_causal_loss(model,batch,native_loss.fixed_cross_entropy)
                    finite=bool(torch.isfinite(loss))
                    if not finite:
                        write_json(output/'nonfinite-loss.json',dict(epoch=epoch+1,group=group_index,row_id=row['row_id'],
                            original_index=original_index,loss_repr=str(float(loss.detach())),group_size=len(group)))
                        raise RuntimeError('Nonfinite formal loss; original observation saved')
                    (loss/len(group)).backward();group_loss+=float(loss.detach())/len(group)
                grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not bool(torch.isfinite(grad_norm)) or float(grad_norm)==0.:
                    write_json(output/'invalid-gradient.json',dict(epoch=epoch+1,group=group_index,gradient_norm_repr=str(float(grad_norm))))
                    raise RuntimeError('Nonfinite or zero LoRA gradient')
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                optimizer.step();scheduler.step();global_step+=1
                if probes:
                    changed={family:not torch.equal(before,p.detach()) for family,(_,p,before) in probes.items()}
                    write_json(output/'first-nonzero-update.json',dict(step=global_step,learning_rate=learning_rate,
                        gradient_norm_before_clip=float(grad_norm),changed_B_families=changed,fresh_run=True))
                    assert set(changed)=={'DeltaNet','full_attention','MLP'} and all(changed.values())
                    first_update_checked=True
                loss_sum+=group_loss
                write_json(output/'progress.json',dict(epoch=epoch+1,group_index=group_index,global_step=global_step,
                    actual_group_size=len(group),mean_group_loss=group_loss,learning_rate=learning_rate,training_only=True))
                print(json.dumps(dict(epoch=epoch+1,group=group_index,step=global_step,group_size=len(group),loss=group_loss)),flush=True)
            assert global_step==11*(epoch+1) and scheduler.last_epoch==global_step and first_update_checked
            check_deadline(args.absolute_deadline_unix);stage='save_complete_epoch_%d'%(epoch+1)
            destination=output/f'epoch-{epoch+1:03d}';staging=output/f'.epoch-{epoch+1:03d}.incomplete'
            assert not destination.exists() and not staging.exists()
            staging.mkdir();model.save_pretrained(staging/'adapter',safe_serialization=True)
            assert all((staging/'adapter'/name).is_file() for name in ['adapter_model.safetensors','adapter_config.json'])
            state={'completed_epochs':epoch+1,'global_step':global_step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'optimizer_parameter_names':list(named)}
            torch.save(state,staging/'training_state.pt')
            files={str(p.relative_to(staging)):sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
            write_json(staging/'checkpoint.json',dict(fingerprint=fingerprint,files=files,completed_epochs=epoch+1,
                global_step=global_step,smoke_only=False,new_resume_numerical_equivalence_tested=False))
            durable_publish_checkpoint(output,staging,destination)
            write_json(output/('summary-epoch-%03d.json'%(epoch+1)),dict(completed_epochs=epoch+1,
                global_step=global_step,mean_accumulation_group_loss=loss_sum/11,epoch_seconds=time.monotonic()-epoch_started,
                checkpoint=str(destination),durability=ref(output/f'durability-epoch-{epoch+1:03d}.json')))
            control.update(completed_epochs=epoch+1,global_step=global_step,checkpoint=str(destination),updated_unix_seconds=time.time())
            write_json(output/'run-control.json',control)
        assert global_step==22
        complete=dict(smoke_only=False,epochs=2,global_step=22,rows=87,source_sha256=sha256(__file__),
            indexed_head_sha256=INDEXED_SHA,elapsed_seconds=time.monotonic()-started,base_parameters=8953803264,
            trainable_adapter_parameters=43278336,total_parameters=8997081600,
            max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            durability={str(e):ref(output/f'durability-epoch-{e:03d}.json') for e in (1,2)},
            result_type='training_only_no_inference_or_quality',fresh_base_and_adapter=True,probe_state_used=False,
            both_epochs_file_directory_fsync_and_reopened_SHA_verified=True,automatic_resume=False)
        write_json(output/'training_complete.json',complete)
        control.update(status='complete');write_json(output/'run-control.json',control)
        print(json.dumps(dict(complete=ref(output/'training_complete.json'))),flush=True)
    except BaseException as error:
        control.update(status='failed',stage=stage,error_type=type(error).__name__)
        write_json(output/'run-control.json',control)
        write_json(output/'failure.json',dict(stage=stage,error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,source_sha256=sha256(__file__),
            preserve_original_complete_epochs=True,no_automatic_retry_or_resume=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--absolute-deadline-unix',required=True,type=float)
    run(parser.parse_args())
