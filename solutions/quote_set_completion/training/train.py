#!/usr/bin/env python3
"""Fixed warm-d52 one-epoch denoising loop; full state saved once for durable ACK.

The parent must bind actual native-CPU and engineering receipts, freeze sources
and authorize a separate formal plan. This entry has no smoke, inference,
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
MODULE_SHAS={
    'indexed_head':INDEXED_SHA,
    'qwen35_runner':'33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669',
    'qwen35_relation_cost':'a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c',
    'qwen35_smoke':'9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3',
}
CONFIG=None  # Bound once from the immutable authorized route plan.
import configuration
import runtime_contract as contract


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
    global CONFIG
    assert plan['schema']=='quote-set-completion-warm-oneepoch-v2' and plan['formal_training_authorized'] is True
    CONFIG=configuration.validate(plan['configuration'],plan['route'])
    configuration.validate_storage(plan)
    assert plan['budget_seconds']>0 and plan['grace_seconds']==30
    assets=plan['assets']
    for name,item in assets.items():assert ref(item['path'])==item,name
    for name,sha in MODULE_SHAS.items():assert assets[name]['sha256']==sha,name
    assert assets['trainer']==ref(__file__) and assets['configuration']==ref(configuration.__file__)
    assert assets['runtime_contract']==ref(contract.__file__)
    assert assets['warm_model']['sha256']==plan['warm_model_sha256']
    assert assets['model_manifest']['sha256']=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
    cpu=read(assets['cpu_result']['path']); runtime=read(assets['cpu_runtime']['path'])
    assert cpu['status']=='PASS_NATIVE_FULL_INPUT_AND_INDEXED_POSITION_CPU'
    assert cpu['train_rows']==CONFIG['rows'] and cpu['held_rows']==CONFIG['held_rows']
    assert cpu['all_local_native_actual_IDs_labels_rendering_exact'] and cpu['complete_inputs_and_answers']
    assert identity(cpu['encoded'])==identity(assets['encoded'])
    assert identity(cpu['runtime'])==identity(assets['cpu_runtime'])
    assert identity(cpu['plan'])==identity(assets['cpu_plan'])
    assert identity(cpu['source'])==identity(assets['native_cpu_source'])
    assert identity(cpu['state_interface'])==identity(assets['state_cpu_result'])
    state_cpu=read(assets['state_cpu_result']['path']);assert state_cpu['status']=='PASS_PARAMETERIZED_COMPLETE_STATE_CPU_INTERFACE'
    assert state_cpu['actual_CUDA_RNG_roundtrip_exercised'] is False
    assert runtime['torch']=='2.7.1+cu126' and runtime['transformers']=='5.20.2'
    assert runtime['peft']=='0.21.2' and runtime['tokenizers']=='0.23.1'
    assert runtime['modeling']['sha256']==MODEL_SOURCE_SHA and runtime['indexed_head']['sha256']==INDEXED_SHA
    probe=read(assets['probe_complete']['path']);step=read(assets['probe_step']['path'])
    probe_manifest=read(assets['probe_manifest']['path']);expected=plan['probe_contract']
    assert probe['step_sha256']==assets['probe_step']['sha256'] and probe['manifest_sha256']==assets['probe_manifest']['sha256']
    assert probe['actual_steps']==1 and probe['actual_backwards']==expected['backwards']
    assert probe['fresh_state_discarded'] and probe['no_checkpoint']
    assert probe['status']==step['status']==expected['status'] and step['actual_optimizer_steps']==1
    assert step['actual_backwards']==expected['backwards'] and step['no_checkpoint'] and step['no_input_truncation']
    assert probe_manifest['head_index_strategy']=='full backbone then supervised causal positions in BF16 head, external unchanged per-row CE'
    assert not probe_manifest['no_saved_adapter_loaded'] and probe_manifest['base_frozen']
    assert probe_manifest['warm_adapter']==plan['warm_adapter'] and probe_manifest['warm_adapter_weights_only']
    assert step['state_save_roundtrip']['logical_state_exact'] and step['state_save_roundtrip']['discarded_after_check']
    assert identity(probe_manifest['encoded'])==identity(assets['encoded'])
    assert step['covered_frontier']==expected['frontier']
    payload=json.loads(gzip.decompress(Path(assets['encoded']['path']).read_bytes()))
    encoded=payload['train'];audit=read(assets['train_audit']['path'])
    assert len(encoded)==len(audit)==CONFIG['rows'] and len({r['row_id'] for r in encoded})==CONFIG['rows']
    assert len({str(r['pmc_id']) for r in audit})==CONFIG['documents']
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
    assert sum(len(r['input_ids']) for r in encoded)==CONFIG['full_training_tokens']
    assert max(len(r['input_ids']) for r in encoded)==CONFIG['maximum_full_tokens']
    assert sum(r['supervised_tokens'] for r in encoded)==CONFIG['supervised_tokens']
    return encoded,audit


def batch_plan(epoch):return configuration.batch_plan(CONFIG,epoch)


def run(args):
    assert __debug__
    started=time.monotonic();plan=read(args.plan);encoded,audit=verify_inputs(plan)
    check_deadline(args.absolute_deadline_unix)
    assets=plan['assets'];output=Path(plan['output_dir']);persistent=Path(plan['storage_parent']).resolve()
    assert output.is_absolute() and persistent.is_dir()
    assert not output.exists() and args.stop_after_epoch==1
    if (output/'training_complete.json').exists():raise ValueError('Refuse already complete run')
    output.resolve().relative_to(persistent);assert output.resolve()!=persistent
    assert os.environ['PATIENTPHEX_DURABILITY_ROOT']==str(persistent)
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
    assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
    output.mkdir(parents=True,exist_ok=False);sync_ancestors(output.parent)
    stage='runtime';control=dict(warm_adapter_weights_only=True,stop_after_epoch=args.stop_after_epoch,started_completed_epochs=0,configured_epochs=1,
        global_step=0,status='running',pid=os.getpid(),started_unix_seconds=time.time())
    write_json(output/'run-control.json',control)
    try:
        import numpy as np
        import torch,transformers,peft,tokenizers
        import transformers.loss.loss_utils as native_loss
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
        from transformers import get_linear_schedule_with_warmup
        assert torch.__version__=='2.7.1+cu126' and transformers.__version__=='5.20.2'
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
            ['qwen35_smoke','qwen35_relation_cost','qwen35_runner','warm_model','indexed_head']}
        loader=modules['warm_model'];protocol=modules['qwen35_relation_cost'];indexed=modules['indexed_head']
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        stage='fresh_public_base_warm_d52_fresh_Adam_RNG';check_deadline(args.absolute_deadline_unix)
        model,named,loaded=loader.load_warm(plan['model'],assets['model_manifest']['path'],CONFIG['seed'],plan['warm_adapter'])
        base=model.get_base_model();assert base.loss_function is native_loss.ForCausalLMLoss
        assert not any(p.requires_grad for p in base.lm_head.parameters())
        assert loaded['fresh_public_base'] and not loaded['no_saved_adapter_loaded'] and not loaded['old_optimizer_scheduler_RNG_loaded']
        assert sum(p.numel() for p in named.values())==43278336
        assert sum(p.numel() for p in model.parameters())==8997081600<10_000_000_000
        assert {id(p) for p in named.values()}=={id(p) for p in model.parameters() if p.requires_grad}
        assert all('.lora_A.' in n or '.lora_B.' in n for n in named)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        assert model.is_gradient_checkpointing
        # Same helper and real forward/loss call as the new route's engineering
        # probe; no local head numeric contract is run again during formal training.
        fingerprint=dict(schema='quote-set-completion-warm-oneepoch-indexed-CE-v2',source_sha256=sha256(__file__),
            configuration=CONFIG,route=plan['route'],train_sha256=assets['train']['sha256'],encoded_sha256=assets['encoded']['sha256'],
            input_ids_labels_native_verified=True,train_document_ids=sorted({str(r['pmc_id']) for r in audit}),
            assets=assets,indexed_head_sha256=INDEXED_SHA,native_loss_sha256=NATIVE_LOSS_SHA,
            runtime=loaded['runtime'],model_files=loaded['model_files'],initial_public_base_fresh_adapter_d52=True,warm_adapter=plan['warm_adapter'],
            objective='standard per-row supervised mean CE; actual group average; no alpha05 patient/value weighting')
        fingerprint=json.loads(json.dumps(fingerprint,sort_keys=True,default=json_default))
        write_json(output/'run_manifest.json',dict(fingerprint=fingerprint,loaded=loaded,examples=261,epochs=1,
            steps_per_epoch=33,total_optimizer_steps=33,warmup_steps=2,full_tokens=1380333,supervised_tokens=12406,max_tokens=9307,
            base_head_frozen=True,labels_None_in_model_forward=True,FP32_selected_CE=True,alpha05_loss_used=False,
            retokenized=False,validation_labels_used=False,no_automatic_resume=True,warm_d52_weights_only=True,freshAdam=True))
        for name in ['trainer','configuration','runtime_contract','warm_model','indexed_head','qwen35_smoke','qwen35_relation_cost']:
            source=Path(assets[name]['path']);destination=output/('train.py' if name=='trainer' else name+'.py')
            with destination.open('xb') as handle:handle.write(source.read_bytes());handle.flush();os.fsync(handle.fileno())
        sync_ancestors(output)
        write_json(output/'epoch-groups.json',{'1':batch_plan(0)})
        parameters=list(named.values());optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=.01)
        assert not optimizer.state
        scheduler=get_linear_schedule_with_warmup(optimizer,2,33)
        assert scheduler.last_epoch==0 and optimizer.param_groups[0]['lr']==0
        completed_epochs=global_step=0
        model.train();first_update_checked=False
        for epoch in range(1):
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
                        gradient_norm_before_clip=float(grad_norm),changed_B_families=changed,fresh_optimizer=True,warm_d52=True,resumed_epoch=False))
                    assert set(changed)=={'DeltaNet','full_attention','MLP'} and all(changed.values())
                    first_update_checked=True
                loss_sum+=group_loss
                write_json(output/'progress.json',dict(epoch=epoch+1,group_index=group_index,global_step=global_step,
                    actual_group_size=len(group),mean_group_loss=group_loss,learning_rate=learning_rate,training_only=True))
                print(json.dumps(dict(epoch=epoch+1,group=group_index,step=global_step,group_size=len(group),loss=group_loss)),flush=True)
            assert global_step==CONFIG['steps_per_epoch']*(epoch+1) and scheduler.last_epoch==global_step and first_update_checked
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
                global_step=global_step,smoke_only=False,new_resume_numerical_equivalence_tested=False,
                serialized_state_digest=contract.tree_digest(state,torch),trainable_parameter_digest=contract.tree_digest({n:p for n,p in named.items()},torch)))
            local_receipt=contract.local_publish_checkpoint(staging,destination,output)
            write_json(output/f'local-checkpoint-epoch-{epoch+1:03d}.json',local_receipt)
            write_json(output/('summary-epoch-%03d.json'%(epoch+1)),dict(completed_epochs=epoch+1,
                global_step=global_step,mean_accumulation_group_loss=loss_sum/CONFIG['steps_per_epoch'],epoch_seconds=time.monotonic()-epoch_started,
                checkpoint=str(destination),local_checkpoint=ref(output/f'local-checkpoint-epoch-{epoch+1:03d}.json'),persistent_ACK_pending=True))
            control.update(completed_epochs=epoch+1,global_step=global_step,checkpoint=str(destination),updated_unix_seconds=time.time())
            write_json(output/'run-control.json',control)
        assert global_step==CONFIG['optimizer_calls']
        complete=dict(smoke_only=False,epochs=1,global_step=CONFIG['optimizer_calls'],rows=CONFIG['rows'],source_sha256=sha256(__file__),
            indexed_head_sha256=INDEXED_SHA,elapsed_seconds=time.monotonic()-started,base_parameters=8953803264,
            trainable_adapter_parameters=43278336,total_parameters=8997081600,
            max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            local_checkpoints={str(e):ref(output/f'local-checkpoint-epoch-{e:03d}.json') for e in (1,)},
            result_type='training_only_no_inference_or_quality',initial_fresh_base_warm_d52=True,fresh_optimizer_RNG=True,probe_state_used=False,
            one_epoch_SHM_only_fsync_reopen_verified=True,epoch1_persistent_ACK_still_required=True,automatic_resume=False)
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
    parser.add_argument('--stop-after-epoch',type=int,choices=[1],required=True)
    run(parser.parse_args())
