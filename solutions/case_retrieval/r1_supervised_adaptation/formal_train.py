"""Fresh R1 supervised adaptation: original124 rows, two complete epochs, no quality.

Each epoch writes complete adapter/Adam/scheduler/RNG, fsyncs, reopens and verifies.
No inference, evaluation, precision fallback, resume, or automatic retry.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
import numpy as np
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    def default(value):
        if isinstance(value,(set,frozenset)):return sorted(value,key=repr)
        raise TypeError(type(value).__name__)
    def safe(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        return value
    with Path(path).open('x') as f:
        json.dump(safe(value),f,ensure_ascii=False,indent=2,default=default,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def observe_then_require(path,value,condition):
    save(path,value)
    assert condition, 'Observed finite/frozen contract failed; evidence written first'

def checked(ref):
    p=Path(ref['path']);assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'];return p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args()
    start=time.monotonic();plan=json.loads(Path(args.plan).read_text());out=Path(plan['output_dir']);out.mkdir(exist_ok=False)
    stage='fresh_source_data_asset_GPU_gate';last_row=None;updates=0;torch=None
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Authorized formal1800-second deadline reached')))
    def stamp():return datetime.now(timezone.utc).isoformat()
    save(out/'started.json',dict(at_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),plan_sha256=sha(args.plan)))
    try:
        assert sha(__file__)==plan['source_sha256'] and plan['budget_seconds']==1800 and plan['grace_seconds']==30
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        proc=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        target=[x.split(', ') for x in gpu.splitlines() if plan['gpu_uuid'] in x];assert len(target)==1
        assert target[0][0]=='1' and target[0][2]=='NVIDIA A30' and int(target[0][-1])>=23500
        assert not any(x.split(',')[0].strip()==plan['gpu_uuid'] for x in proc.splitlines())
        disk=shutil.disk_usage(out);assert disk.free>2*1024**3
        save(out/'fresh-gate.json',dict(at_utc=stamp(),gpu=gpu,compute_processes=proc,free_disk_bytes=disk.free))
        for ref in plan['refs'].values():checked(ref)
        checked(plan['frozen_quality_contract'])
        cpu=json.loads(checked(plan['refs']['cpu_result']).read_text());assert cpu['status']=='PASS_R1_SFT_CPU_DATA_TOKEN_BOUNDARY_META_PARAMETER_AND_TOY_LOSS_CONTRACT'
        assert cpu['token_contract']['total_train_tokens']==401291 and cpu['token_contract']['prefix_think_open_rows']==0
        modeldir=Path(plan['model_dir']);assets={}
        for name,item in plan['model_files'].items():
            path=modeldir/name;assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
            st=path.stat();assets[name]=dict(item,mtime_ns=st.st_mtime_ns)
        rows=[json.loads(x) for x in gzip.decompress(checked(plan['refs']['encoded']).read_bytes()).decode().splitlines()]
        assert len(rows)==124 and len({x['pmc_id'] for x in rows})==32 and sum(len(x['input_ids']) for x in rows)==401291
        assert {x['pmc_id'] for x in rows}==set(plan['train32_ids']) and not set(plan['train32_ids'])&set(plan['excluded_ids'])
        for row in rows:
            n=row['prompt_length'];assert row['input_ids'][-1]==151645 and row['input_ids'][n-1]==151670
            assert row['labels']==[-100]*n+row['input_ids'][n:] and len(row['input_ids'])<=8192
        indices=list(range(124));random.Random(20261023).shuffle(indices)
        longest=max(range(124),key=lambda i:len(rows[i]['input_ids']));assert longest==6 and len(rows[longest]['input_ids'])==5519
        diagnostic=[i for i in indices if i!=longest][:11]+[longest];assert diagnostic==plan['diagnostic_indices']
        import torch as torch_module
        import transformers
        import tokenizers
        import peft
        from transformers import AutoConfig,AutoModelForCausalLM
        from transformers.loss.loss_utils import ForCausalLMLoss
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from peft import LoraConfig,get_peft_model
        sys.path.insert(0,str(checked(plan['refs']['selected_loss']).parent));import selected_loss
        torch=torch_module;torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.7.1+cu126' and transformers.__version__=='5.20.2' and peft.__version__=='0.21.2' and tokenizers.__version__=='0.23.1'
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
        derived=json.loads(checked(plan['refs']['derived_config']).read_text());expected=json.loads((modeldir/'config.json').read_text());expected['rope_scaling']['attention_factor']=1.0;assert derived==expected
        config=AutoConfig.from_pretrained(str(checked(plan['refs']['derived_config'])),local_files_only=True,trust_remote_code=False)
        config.use_cache=False;config._attn_implementation='sdpa'
        stage='fresh_BF16_base_load';load_start=time.monotonic()
        base,loading=AutoModelForCausalLM.from_pretrained(modeldir,config=config,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        assert type(base).__name__=='Qwen3ForCausalLM' and sum(p.numel() for p in base.parameters())==8190735360
        assert hashlib.sha256(inspect.getsource(type(base).forward).encode()).hexdigest()==plan['actual_forward_sha256']
        assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' and bool(torch.isfinite(p).all()) for p in base.parameters())
        rope=base.model.rotary_emb;native=ROPE_INIT_FUNCTIONS['yarn'];assert sha(inspect.getsourcefile(native))==plan['rotary']['native_function_source_sha256']
        refgpu,scale=native(base.config,device=rope.inv_freq.device);refcpu,scalecpu=native(base.config,device=torch.device('cpu'))
        assert rope.inv_freq.dtype==torch.float32 and torch.equal(rope.inv_freq,rope.original_inv_freq)
        assert hashlib.sha256(refcpu.numpy().tobytes()).hexdigest()==plan['rotary']['CPU_reference_bytes_sha256']
        assert torch.allclose(rope.inv_freq,refgpu,rtol=8*torch.finfo(torch.float32).eps,atol=0) and torch.allclose(refgpu.cpu(),refcpu,rtol=8*torch.finfo(torch.float32).eps,atol=0)
        assert float(rope.attention_scaling)==float(scale)==float(scalecpu)==1.0
        assert base.loss_function is ForCausalLMLoss
        model=get_peft_model(base,LoraConfig(**plan['lora']))
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        for p in named.values():p.data=p.data.float()
        assert sum(p.numel() for p in named.values())==43646976 and sum(p.numel() for p in model.parameters())==8234382336
        assert len(named)==504 and all('.lora_A.' in n or '.lora_B.' in n for n in named)
        assert all(p.dtype==torch.float32 and p.device.type=='cuda' for p in named.values())
        assert not any(p.requires_grad for p in base.lm_head.parameters()) and not any(p.requires_grad for p in base.get_input_embeddings().parameters())
        model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();assert model.is_gradient_checkpointing
        model.train();params=list(named.values())
        def memory():return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0])
        def batch(row):
            x=torch.tensor([row['input_ids']],device='cuda',dtype=torch.long)
            return dict(input_ids=x,attention_mask=torch.ones_like(x),labels=torch.tensor([row['labels']],device='cuda',dtype=torch.long))
        def frozen():return all(p.grad is None for p in model.parameters() if not p.requires_grad)
        save(out/'model-manifest.json',dict(source_sha256=sha(__file__),plan_sha256=sha(args.plan),asset_files=assets,loading=loading,load_seconds=time.monotonic()-load_start,
            base_parameters=8190735360,lora_parameters=43646976,total_parameters=8234382336,target_linear_modules=252,trainable_tensors=len(named),only_lora_trainable=True,
            full_head_frozen=True,embedding_frozen=True,checkpointing_nonreentrant=True,use_cache=False,TF32=False,precision='BF16 base + FP32 LoRA, unquantized full head',
            derived_config=plan['refs']['derived_config'],actual_rotary_CPU_sha256=plan['rotary']['CPU_reference_bytes_sha256'],actual_rotary_scaling=1.,runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
            formal_recipe=plan['formal_recipe'],original124_tokens=401291,original124_supervised_tokens=7046,memory=memory()))
        # Prior exact same-logits CE proof replaces unstable cross-backward hard gate.
        diagnostic=json.loads(checked(plan['refs']['numerical_result']).read_text())
        assert diagnostic['status']=='NUMERICAL_DIAGNOSTIC_COMPLETED_NO_TOLERANCE_CHANGE_OR_TRAINING'
        assert diagnostic['logits_gradient_comparison']['exact'] and diagnostic['logits_gradient_comparison']['abs_max']==0.
        assert all(v['ignored_gradient_exact_zero'] for v in diagnostic['same_actual_logits_CE'].values())
        assert diagnostic['model_comparisons']['nativeB']['allclose_original'] is False
        assert diagnostic['optimizer_updates']==0 and diagnostic['original_gradient_tolerance']==dict(rtol=1e-4,atol=1e-6)
        cpu=json.loads(checked(plan['refs']['cpu_result']).read_text())
        assert all(v['max_abs_gradient_difference']==0 for v in cpu['parameter_contract']['CPU_toy_loss_gradient_checks'])
        # Keep original recipe's shuffled diagnostic indices, no added model passes.
        diagnostic=plan['diagnostic_indices'];base.loss_function=selected_loss.selected_causal_loss
        assert base.loss_function is selected_loss.selected_causal_loss
        stress=json.loads(checked(plan['refs']['stress_result']).read_text())
        assert stress['status']=='PASS_R1_STRESS_V2_FIXED_CE_PROOF_ADAM_GROUP8_THEN_3_GRADIENTS_PLUS_MAX5519'
        assert stress['optimizer_updates']==1 and stress['diagnostic_forward_backward']==12 and stress['final_inventory']['all_grad_finite']
        recipe=plan['formal_recipe']
        assert recipe==dict(epochs=2,batch_size=1,gradient_accumulation=8,optimizer_steps=32,warmup_steps=2,learning_rate=1e-4,weight_decay=.01,max_grad_norm=1.,seed=20261023,max_length_reject_cap=8192,packing=False,quantization=False)
        from transformers import get_linear_schedule_with_warmup
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        scheduler=get_linear_schedule_with_warmup(optimizer,2,32)
        assert {id(p) for group in optimizer.param_groups for p in group['params']}=={id(p) for p in params}
        assert all(torch.count_nonzero(p).item()==0 for n,p in named.items() if '.lora_B.' in n)
        first_update_verified=False;global_step=0;completed_epochs=0;row_counter=0;token_counter=0;supervised_counter=0
        def inventory():
            grads=[p.grad for p in params if p.grad is not None];states=[v for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
            return dict(gradient_tensors=len(grads),gradient_bytes=sum(g.numel()*g.element_size() for g in grads),all_grad_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                optimizer_parameters_with_state=len(optimizer.state),optimizer_tensor_bytes=sum(v.numel()*v.element_size() for v in states),frozen_base_no_gradient=frozen())
        def update_progress(value):
            temporary=out/'progress.json.next';assert not temporary.exists();save(temporary,value);os.replace(temporary,out/'progress.json')
        def sync_directory(path):
            fd=os.open(path,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        def file_ref(path):return dict(bytes=path.stat().st_size,sha256=sha(path))
        fingerprint=dict(source_sha256=sha(__file__),plan_sha256=sha(args.plan),encoded_sha256=plan['refs']['encoded']['sha256'],
            train_data_sha256='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8',model_id='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
            model_revision='5ee5fa4832e658ab84298806de06ed2a167834bd',base_parameters=8190735360,lora_parameters=43646976,total_parameters=8234382336)
        def save_checkpoint(epoch):
            tick=time.monotonic();staging=out/('.epoch-%03d.incomplete'%epoch);target=out/('epoch-%03d'%epoch)
            assert not staging.exists() and not target.exists();staging.mkdir();adapter=staging/'adapter'
            model.save_pretrained(adapter,safe_serialization=True)
            state=dict(completed_epochs=epoch,global_step=global_step,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),master_seed=plan['seed'])
            with (staging/'training_state.pt').open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            for path in staging.rglob('*'):
                if path.is_file():
                    with path.open('rb') as f:os.fsync(f.fileno())
            files={str(path.relative_to(staging)):file_ref(path) for path in sorted(staging.rglob('*')) if path.is_file()}
            assert {'adapter/adapter_model.safetensors','adapter/adapter_config.json','training_state.pt'}<=set(files)
            metadata=dict(status='complete_epoch',completed_epochs=epoch,global_step=global_step,files=files,fingerprint=fingerprint,
                recipe=recipe,train_rows_per_epoch=124,all_train_rows_used=True,total_rows_seen=row_counter,total_tokens_seen=token_counter,
                total_supervised_seen=supervised_counter,first_nonzero_update_verified=first_update_verified,base_frozen=True,no_validation_or_quality=True,
                saved_at_utc=stamp(),training_state_keys=sorted(state))
            save(staging/'checkpoint.json',metadata)
            for path in [adapter,staging]:sync_directory(path)
            staging.rename(target);sync_directory(out)
            # Reopen this exact durable checkpoint and compare all adapter/Adam tensors.
            reopened=json.loads((target/'checkpoint.json').read_text());assert reopened==metadata
            for name,expected in files.items():assert file_ref(target/name)==expected
            saved_adapter=load_file(str(target/'adapter/adapter_model.safetensors'),device='cpu')
            actual_adapter=get_peft_model_state_dict(model)
            assert set(saved_adapter)==set(actual_adapter) and len(saved_adapter)==504
            assert sum(t.numel() for t in saved_adapter.values())==43646976
            for name,tensor in saved_adapter.items():
                assert tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()) and torch.equal(tensor,actual_adapter[name].detach().cpu())
            saved_state=torch.load(target/'training_state.pt',map_location='cpu',weights_only=False)
            assert saved_state['completed_epochs']==epoch and saved_state['global_step']==global_step==16*epoch
            assert saved_state['scheduler']==scheduler.state_dict() and saved_state['optimizer']['param_groups']==optimizer.state_dict()['param_groups']
            assert saved_state['python_rng']==state['python_rng'] and saved_state['numpy_rng'][0]==state['numpy_rng'][0]
            assert np.array_equal(saved_state['numpy_rng'][1],state['numpy_rng'][1]) and saved_state['numpy_rng'][2:]==state['numpy_rng'][2:]
            assert torch.equal(saved_state['torch_rng'],state['torch_rng']) and all(torch.equal(a,b) for a,b in zip(saved_state['cuda_rng'],state['cuda_rng']))
            current=optimizer.state_dict()['state'];assert set(saved_state['optimizer']['state'])==set(current) and len(current)==504
            moments=0
            for key,items in current.items():
                assert set(items)==set(saved_state['optimizer']['state'][key])
                for name,value in items.items():
                    other=saved_state['optimizer']['state'][key][name]
                    if isinstance(value,torch.Tensor):assert torch.equal(other,value.detach().cpu()) and bool(torch.isfinite(other).all());moments+=1
                    else:assert other==value
            ack=dict(status='COMPLETE_EPOCH_DURABLE_FSYNC_REOPEN_ALL_SHA_ADAPTER_AND_ADAM_RNG_EXACT',completed_epochs=epoch,global_step=global_step,
                checkpoint=dict(path=str(target/'checkpoint.json'),**file_ref(target/'checkpoint.json')),adapter=dict(path=str(target/'adapter/adapter_model.safetensors'),**files['adapter/adapter_model.safetensors']),
                training_state=dict(path=str(target/'training_state.pt'),**files['training_state.pt']),adapter_config=dict(path=str(target/'adapter/adapter_config.json'),**files['adapter/adapter_config.json']),
                adapter_tensors_reopened=504,optimizer_states_reopened=504,optimizer_tensors_reopened=moments,all_rng_reopened_exact=True,seconds=time.monotonic()-tick,recorded_utc=stamp())
            save(out/('epoch-%03d-persistent-ack.json'%epoch),ack)
            return ack
        acknowledgements=[];training_started=time.monotonic()
        for epoch in range(2):
            order=list(range(124));random.Random(plan['seed']+epoch).shuffle(order);groups=[order[i:i+8] for i in range(0,124,8)]
            assert len(groups)==16 and sum(map(len,groups))==124 and len(groups[-1])==4
            epoch_start=time.monotonic();epoch_loss=0.
            save(out/('epoch-%03d-order.json'%(epoch+1)),dict(indices=order,groups=groups,seed=plan['seed']+epoch,rows=124))
            for group_number,indices in enumerate(groups,1):
                optimizer.zero_grad(set_to_none=True);group_loss=0.;learning_rate=optimizer.param_groups[0]['lr'];probes={};group_started=time.monotonic()
                if not first_update_verified and learning_rate>0:
                    for n,p in named.items():
                        family=n.split('.')[-4]
                        if '.lora_B.' in n and family not in probes:probes[family]=(n,p,p.detach().clone())
                    assert set(probes)==set(plan['lora']['target_modules'])
                for within,index in enumerate(indices,1):
                    row=rows[index];last_row=dict(epoch=epoch+1,group=group_number,within_group=within,index=index,row_id=row['row_id'],tokens=len(row['input_ids']),supervised_tokens=len(row['input_ids'])-row['prompt_length'])
                    stage='formal_forward';update_progress(dict(stage=stage,at_utc=stamp(),row=last_row,global_step=global_step,loss_pending=True,memory=memory()))
                    b=batch(row)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**b,use_cache=False).loss
                    finite=bool(torch.isfinite(loss));update_progress(dict(stage='formal_forward_observed',at_utc=stamp(),row=last_row,global_step=global_step,loss=float(loss.detach()),loss_finite=finite,memory=memory()))
                    assert finite,'Nonfinite forward saved before assertion'
                    stage='formal_backward';(loss/len(indices)).backward();group_loss+=float(loss.detach())/len(indices)
                    row_counter+=1;token_counter+=len(row['input_ids']);supervised_counter+=last_row['supervised_tokens']
                inv=inventory();norm=torch.nn.utils.clip_grad_norm_(params,1.);norm_finite=bool(torch.isfinite(norm));stage='formal_Adam_update'
                update_progress(dict(stage='formal_gradient_observed',at_utc=stamp(),row=last_row,global_step=global_step,inventory=inv,gradient_norm=float(norm),gradient_norm_finite=norm_finite,memory=memory()))
                assert inv['all_grad_finite'] and inv['frozen_base_no_gradient'] and inv['gradient_tensors']==504 and norm_finite and float(norm)>0
                optimizer.step();scheduler.step();global_step+=1;updates=global_step
                adapter_finite=all(bool(torch.isfinite(p).all()) for p in params)
                changed={family:not torch.equal(before,p.detach()) for family,(_,p,before) in probes.items()}
                step=dict(epoch=epoch+1,group=group_number,global_step=global_step,group_size=len(indices),row_indices=indices,loss=group_loss,learning_rate_used=learning_rate,
                    learning_rate_next=optimizer.param_groups[0]['lr'],gradient_norm_before_clip=float(norm),adapter_finite=adapter_finite,changed_first_nonzero_B_probes=changed,
                    inventory=inventory(),memory=memory(),seconds=time.monotonic()-group_started)
                save(out/('step-%03d.json'%global_step),step)
                assert adapter_finite and step['inventory']['frozen_base_no_gradient']
                if probes:assert all(changed.values());first_update_verified=True
                epoch_loss+=group_loss
                if global_step%4==0:print(json.dumps(dict(epoch=epoch+1,step=global_step,mean_group_loss=group_loss,peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)
            assert global_step==16*(epoch+1) and first_update_verified
            stage='complete_epoch_%d_persistence'%(epoch+1);ack=save_checkpoint(epoch+1);acknowledgements.append(ack);completed_epochs=epoch+1
            save(out/('epoch-%03d-result.json'%completed_epochs),dict(completed_epochs=completed_epochs,global_step=global_step,training_seconds=time.monotonic()-epoch_start-ack['seconds'],
                mean_group_loss=epoch_loss/16,checkpoint=ack['checkpoint'],persistent_ack_sha256=sha(out/('epoch-%03d-persistent-ack.json'%completed_epochs))))
        assert global_step==32 and row_counter==248 and token_counter==802582 and supervised_counter==14092
        torch.cuda.synchronize();save(out/'result.json',dict(status='complete_fresh_R1_124x2_32steps',program_seconds=time.monotonic()-start,training_and_persistence_seconds=time.monotonic()-training_started,
            completed_epochs=2,global_step=32,rows_seen=248,input_tokens_seen=802582,supervised_tokens_seen=14092,acknowledgements=acknowledgements,fingerprint=fingerprint,
            first_nonzero_update_verified=first_update_verified,final_inventory=inventory(),memory=memory(),fresh_public_base=True,probe_state_not_loaded=True,
            no_truncation_or_quantization=True,base_frozen=True,TF32=False,inference_or_quality=False,full80_not_automatically_authorized=True,finished_utc=stamp()))
        print(json.dumps(dict(status='complete_fresh_R1_124x2_32steps',program_seconds=time.monotonic()-start,global_step=32)),flush=True)
    except BaseException as exc:
        mem=None
        try:
            if torch is not None and torch.cuda.is_initialized():mem=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0])
        except Exception:pass
        if not (out/'failure.json').exists():save(out/'failure.json',dict(stage=stage,last_row=last_row,actual_updates=updates,error_type=type(exc).__name__,message=str(exc)[:4000],program_seconds=time.monotonic()-start,memory=mem,no_retry=True,no_quality=True,completed_epochs=locals().get('completed_epochs',0)))
        raise

if __name__=='__main__':main()
