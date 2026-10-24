"""Independent R1 stress v2: frozen-logits CE proof, Adam, accumulated max row.

No adapter checkpoint, inference, evaluation, precision fallback, or automatic retry.
The diagnostic uses a nonzero lr without formal warmup; all state is discarded.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
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
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Authorized 420-second deadline reached')))
    def stamp():return datetime.now(timezone.utc).isoformat()
    save(out/'started.json',dict(at_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),plan_sha256=sha(args.plan)))
    try:
        assert sha(__file__)==plan['source_sha256'] and plan['budget_seconds']==420 and plan['grace_seconds']==30
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
        random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
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
            diagnostic_order=diagnostic,diagnostic_nonzero_lr_no_warmup=True,formal_order_not_replayed=True,memory=memory()))
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
        # No empty_cache or allocator change; stress loop matches normal loss lifetimes.
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        assert {id(p) for group in optimizer.param_groups for p in group['params']}=={id(p) for p in params}
        assert all(torch.count_nonzero(p).item()==0 for n,p in named.items() if '.lora_B.' in n)
        probes={}
        for n,p in named.items():
            family=n.split('.')[-4]
            if '.lora_B.' in n and family not in probes:probes[family]=(n,p,p.detach().clone())
        assert set(probes)==set(plan['lora']['target_modules'])
        def inventory():
            grads=[p.grad for p in params if p.grad is not None];states=[v for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
            return dict(gradient_tensors=len(grads),gradient_bytes=sum(g.numel()*g.element_size() for g in grads),all_grad_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                optimizer_parameters_with_state=len(optimizer.state),optimizer_tensor_bytes=sum(v.numel()*v.element_size() for v in states),frozen_base_no_gradient=frozen())
        torch.cuda.synchronize();tick=time.monotonic();torch.cuda.reset_peak_memory_stats();group_loss=0.
        for ordinal,index in enumerate(diagnostic,1):
            row=rows[index];last_row=dict(ordinal=ordinal,index=index,row_id=row['row_id'],tokens=len(row['input_ids']),supervised_tokens=len(row['input_ids'])-row['prompt_length'])
            if ordinal==9:optimizer.zero_grad(set_to_none=True)
            before=inventory()
            if ordinal==12:
                assert before['optimizer_parameters_with_state']==504 and before['gradient_tensors']==504 and before['all_grad_finite']
                assert any(bool(torch.count_nonzero(p.grad)) for p in params)
            stage='forward_%02d'%ordinal;save(out/('row-%02d-before.json'%ordinal),dict(row=last_row,stage=stage,updates=updates,inventory=before,memory=memory()))
            rowstart=time.monotonic();b=batch(row)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**b,use_cache=False).loss
            loss_finite=bool(torch.isfinite(loss));stage='backward_%02d'%ordinal
            observe_then_require(out/('row-%02d-forward.json'%ordinal),dict(row=last_row,loss=float(loss.detach()),loss_finite=loss_finite,memory=memory()),loss_finite)
            (loss/8).backward();group_loss+=float(loss.detach())/8;torch.cuda.synchronize()
            after=inventory()
            observe_then_require(out/('row-%02d-after.json'%ordinal),dict(row=last_row,seconds=time.monotonic()-rowstart,loss=float(loss.detach()),loss_finite=True,inventory=after,memory=memory()),after['all_grad_finite'] and after['frozen_base_no_gradient'])
            if ordinal==8:
                stage='complete_group8_Adam_update';assert inventory()['all_grad_finite'] and frozen()
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                observe_then_require(out/'first-update-before.json',dict(gradient_norm_before_clip=float(norm),finite=bool(torch.isfinite(norm)),inventory=inventory(),memory=memory()),bool(torch.isfinite(norm)) and float(norm)>0)
                optimizer.step();updates+=1
                changed={family:not torch.equal(before,p.detach()) for family,(_,p,before) in probes.items()}
                parameter_finite=all(bool(torch.isfinite(p).all()) for p in params)
                observe_then_require(out/'first-update.json',dict(updates=updates,effective_group_size=8,lr=1e-4,gradient_norm_before_clip=float(norm),mean_group_loss=group_loss,
                    all7_projection_B_probes_changed=changed,all_adapter_parameters_finite=parameter_finite,inventory=inventory(),memory=memory()),all(changed.values()) and parameter_finite)
        final=inventory();assert final['all_grad_finite'] and final['frozen_base_no_gradient'] and final['optimizer_parameters_with_state']==504 and final['gradient_tensors']==504
        torch.cuda.synchronize()
        save(out/'result.json',dict(status='PASS_R1_STRESS_V2_FIXED_CE_PROOF_ADAM_GROUP8_THEN_3_GRADIENTS_PLUS_MAX5519',started_utc=json.loads((out/'started.json').read_text())['at_utc'],finished_utc=stamp(),program_seconds=time.monotonic()-start,
            diagnostic_seconds=time.monotonic()-tick,diagnostic_forward_backward=12,equivalence_forward_backward=0,optimizer_updates=updates,longest=last_row,final_inventory=final,memory=memory(),
            full_training_authorized=False,checkpoint_saved=False,inference_or_quality=False,probe_state_discarded_on_exit=True,no_truncation_or_quantization=True,base_frozen=True,TF32=False,cross_model_backward_allclose_not_a_hard_gate=True,prior_numerical_result=plan['refs']['numerical_result'],prior_original_probe_remains_failed=True))
        print(json.dumps(dict(status='PASS',program_seconds=time.monotonic()-start,memory=memory())),flush=True)
    except BaseException as exc:
        mem=None
        try:
            if torch is not None and torch.cuda.is_initialized():mem=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0])
        except Exception:pass
        if not (out/'failure.json').exists():save(out/'failure.json',dict(stage=stage,last_row=last_row,actual_updates=updates,error_type=type(exc).__name__,message=str(exc)[:4000],program_seconds=time.monotonic()-start,memory=mem,no_retry=True,no_quality=True,no_checkpoint=True))
        raise

if __name__=='__main__':main()
