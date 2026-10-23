#!/usr/bin/env python3
"""One longest-train-example Qwen3.5 cost probe, not a training entrypoint.

CPU audits the frozen265 full64 rows, longest full answer mask, causal loss and meta
LoRA coverage. GPU performs exactly one fresh-base optimizer step. No inference,
heldout labels, model selection, optimizer resume or full training is supported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
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
from qwen35_smoke import sha, json_sha, write, encode, model_fingerprint, tensor_fingerprints

import train_qwen35_full64 as trainer
TRAINER_SHA = 'f83236587d8f307bb6e51ca44c9c69177efce27661b0f889bd00745808030f03'
COPIED_COST_SHA = 'a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'

HELPER_SHA = '9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
TRAIN_SHA = trainer.TRAIN_SHA
EXCLUDED_SHA = trainer.EXCLUDED_SHA
TARGETS = ['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a',
           'out_proj','gate_proj','up_proj','down_proj']
BASE_PARAMETERS = 8_953_803_264
ADAPTER_PARAMETERS = 43_278_336
SEED = 20261023


def family(name):
    for marker,label in [('.linear_attn.','DeltaNet'),('.self_attn.','full_attention'),('.mlp.','MLP')]:
        if marker in name:
            return label
    raise ValueError('Unexpected adapter parameter family: '+name)


def select_longest(data, tokenizer, output):
    encoded_rows = [encode(row,tokenizer) for row in data]
    audit = [dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),full_tokens=len(e['input_ids']),
                  prefix_tokens=e['prefix_tokens'],supervised_tokens=e['supervised_tokens'])
             for row,e in zip(data,encoded_rows)]
    write(output/'token-mask-audit.json',{'rows':audit,'prefix_mask_eos_all_passed':True,
          'max_seq_length':6400,'truncated_rows':0,'all_encoded_sha256':json_sha(encoded_rows)})
    if (len(data)!=265 or max(x['full_tokens'] for x in audit)!=6153
        or sum(x['full_tokens'] for x in audit)!=864937 or sum(x['supervised_tokens'] for x in audit)!=15410):
        raise ValueError('Frozen actual265-row / full864937 / supervised15410 / max6153 accounting differs')
    if any(x['full_tokens']>6400 for x in audit):
        raise ValueError('No truncation: a real full training example exceeds6400')
    index = min(range(len(data)),key=lambda i:(-audit[i]['full_tokens'],data[i]['row_id']))
    result = {'rule':'Maximum full training tokens; lexical row_id tie-break; no heldout labels',
              'row':data[index],'encoded':encoded_rows[index],
              'total_full_tokens':sum(x['full_tokens'] for x in audit),
              'total_supervised_tokens':sum(x['supervised_tokens'] for x in audit),
              'encoded_sha256':json_sha(encoded_rows[index]),'all_encoded_sha256':json_sha(encoded_rows)}
    write(output/'selection.json',result)
    return result


def causal_loss_probe(torch, output):
    from transformers.loss.loss_utils import ForCausalLMLoss
    # Labels are next-token targets: token0 and token1 are prompt; token4 is padding.
    logits = torch.tensor([[[9.,-2.,3.],[0.,3.,-1.],[-1.,0.,3.],[2.,-4.,8.],[1.,2.,3.]]],requires_grad=True)
    labels = torch.tensor([[-100,-100,1,2,-100]])
    actual = ForCausalLMLoss(logits=logits,labels=labels,vocab_size=3)
    expected = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,3),labels[:,1:].reshape(-1),ignore_index=-100)
    if not torch.allclose(actual,expected,atol=1e-7,rtol=0):
        raise RuntimeError('Actual causal loss disagrees with independent shifted masked CE')
    actual.backward()
    gradient = logits.grad
    if not (gradient[0,1,1]<0 and gradient[0,2,2]<0
            and bool((gradient[0,[0,3,4]]==0).all())):
        raise RuntimeError('Causal supervised-gradient direction or ignored-position mask differs')
    changed = logits.detach().clone(); changed[0,[0,3,4]] = torch.tensor([100.,-100.,50.])
    ignored_changed_loss = ForCausalLMLoss(logits=changed,labels=labels,vocab_size=3)
    if not torch.equal(actual.detach(),ignored_changed_loss):
        raise RuntimeError('Ignored causal/padding positions affected loss')
    result = {'actual_loss':float(actual.detach()),'manual_shifted_ce':float(expected.detach()),
              'supervised_gradients':[float(gradient[0,1,1]),float(gradient[0,2,2])],
              'ignored_gradients_all_zero':True,'masked_logits_change_loss_unchanged':True,
              'loss_source_sha256':sha(inspect.getsourcefile(ForCausalLMLoss)),
              'cpu_only':True,'not_a_model_quality_check':True}
    write(output/'causal-loss-cpu-check.json',result)
    return result


def attach_adapter(base, torch, peft):
    if sum(p.numel() for p in base.parameters()) != BASE_PARAMETERS:
        raise RuntimeError('Exact text base parameter count differs')
    modules = {name:module for name,module in base.named_modules()
               if name.rsplit('.',1)[-1] in TARGETS}
    expected = {name:(24 if name.startswith('in_proj_') or name=='out_proj' else
                     32 if name in {'gate_proj','up_proj','down_proj'} else 8) for name in TARGETS}
    counts = Counter(name.rsplit('.',1)[-1] for name in modules)
    if dict(counts)!=expected or not all(isinstance(m,torch.nn.Linear) for m in modules.values()):
        raise RuntimeError('Missing or unexpected hybrid Linear projection modules')
    config = peft.LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,
                             target_modules=TARGETS,bias='none')
    model = peft.get_peft_model(base,config)
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if any('.lora_A.' not in n and '.lora_B.' not in n for n in trainable):
        raise RuntimeError('Only LoRA parameters may train')
    if sum(p.numel() for p in trainable.values())!=ADAPTER_PARAMETERS:
        raise RuntimeError('Exact full-hybrid LoRA parameter count differs')
    total = sum(p.numel() for p in model.parameters())
    if total!=BASE_PARAMETERS+ADAPTER_PARAMETERS or not total<10_000_000_000:
        raise RuntimeError('Total parameter accounting/cap differs')
    families = Counter()
    for name,p in trainable.items():
        families[family(name)] += p.numel()
    coverage = {'base_parameters':BASE_PARAMETERS,'adapter_parameters':ADAPTER_PARAMETERS,
                'total_parameters':total,'target_counts':dict(counts),'target_module_names':sorted(modules),
                'trainable_parameter_families':dict(families),'all_trainable_names':sorted(trainable),
                'all_parameter_devices':sorted({p.device.type for p in model.parameters()})}
    return model,trainable,coverage


def gpu_probe(args, selection, output, status, torch, peft, text_config):
    from transformers import Qwen3_5ForCausalLM
    if not args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('One allocated UUID must be the sole CUDA_VISIBLE_DEVICES')
    check = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                            '--format=csv,noheader'],capture_output=True,text=True,check=True)
    write(output/'gpu-preflight.json',{'uuid':args.gpu_uuid,'processes':check.stdout})
    if any(line.split(',')[0].strip()==args.gpu_uuid for line in check.stdout.splitlines()):
        raise RuntimeError('Allocated GPU is occupied; do not share')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Need exactly one BF16 CUDA device')
    stages = {}; peak_allocated = 0; peak_reserved = 0

    def measure(name, function):
        nonlocal peak_allocated,peak_reserved
        status(name); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter(); succeeded = False
        try:
            result = function(); torch.cuda.synchronize(); succeeded = True
            return result
        finally:
            value = {'seconds':time.perf_counter()-begin,'success':succeeded,
                     'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                     'ending_allocated_bytes':torch.cuda.memory_allocated()}
            peak_allocated = max(peak_allocated,value['peak_allocated_bytes'])
            peak_reserved = max(peak_reserved,value['peak_reserved_bytes'])
            stages[name] = value
            write(output/'timings.json',{'stages':stages,'max_observed_allocated_bytes':peak_allocated,
                  'max_observed_reserved_bytes':peak_reserved,
                  'timing':'Wall perf_counter bounded by CUDA synchronize before/after successful stages'})

    def load_base():
        return Qwen3_5ForCausalLM.from_pretrained(args.model,config=text_config,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,
            trust_remote_code=False,output_loading_info=True)
    base,loading = measure('load_fresh_text_base',load_base)
    write(output/'base-loading-info.json',loading)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('Incomplete or mismatched text checkpoint load')
    if any(not key.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for key in loading.get('unexpected_keys',[])):
        raise RuntimeError('Unexpected non-vision/non-MTP checkpoint key')
    if base.__class__.__name__!='Qwen3_5ForCausalLM' or any('visual' in n for n,_ in base.named_modules()):
        raise RuntimeError('Wrong text-only model class')
    base.config.use_cache = False
    model,trainable,coverage = measure('attach_new_full_lora',lambda:attach_adapter(base,torch,peft))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    if not model.is_gradient_checkpointing:
        raise RuntimeError('Nonreentrant gradient checkpointing was not enabled')
    coverage.update(gpu_uuid=args.gpu_uuid,gpu_name=torch.cuda.get_device_name(0),
                    gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
                    gradient_checkpointing=True,use_reentrant=False,base_adapter_loaded=False)
    write(output/'model-coverage.json',coverage)
    before = tensor_fingerprints(peft.get_peft_model_state_dict(model),torch)
    encoded = selection['encoded']
    def make_batch():
        ids = torch.tensor([encoded['input_ids']],dtype=torch.long,device='cuda:0')
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
                'labels':torch.tensor([encoded['labels']],dtype=torch.long,device='cuda:0')}
    batch = measure('batch_to_device',make_batch)
    optimizer = torch.optim.AdamW(list(trainable.values()),lr=1e-4,weight_decay=.01)
    model.train(); optimizer.zero_grad(set_to_none=True)
    def forward():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(**batch,use_cache=False).loss
    loss = measure('forward',forward)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite longest-example loss')
    measure('backward',lambda:loss.backward())
    gradients = {}; group_norms = defaultdict(list)
    def gradients_and_clip():
        for name,p in trainable.items():
            norm = None if p.grad is None else float(p.grad.detach().float().norm().item())
            gradients[name] = norm
            if norm is not None:
                if not math.isfinite(norm):
                    raise RuntimeError('Nonfinite LoRA gradient')
                group_norms[family(name)].append(norm)
        if set(group_norms)!={'DeltaNet','full_attention','MLP'} or any(not any(v>0 for v in values) for values in group_norms.values()):
            raise RuntimeError('Each DeltaNet/full-attention/MLP family needs nonzero gradients')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen base parameter received gradient')
        total = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0).item())
        if not math.isfinite(total) or total<=0:
            raise RuntimeError('Nonfinite or zero aggregate gradient')
        return total
    gradient_norm = measure('gradient_checks_and_clip',gradients_and_clip)
    measure('optimizer_first_step',lambda:optimizer.step())
    optimizer.zero_grad(set_to_none=True)
    if not all(bool(torch.isfinite(p).all()) for p in trainable.values()):
        raise RuntimeError('Nonfinite adapter values after optimizer')
    after = tensor_fingerprints(peft.get_peft_model_state_dict(model),torch)
    changed_b = [name for name in before if 'lora_B' in name and before[name]!=after[name]]
    changed_families = sorted({family(name) for name in changed_b})
    if changed_families!=['DeltaNet','MLP','full_attention']:
        raise RuntimeError('LoRA B did not change in each hybrid family')
    write(output/'single-step.json',{'loss':float(loss.detach().item()),'optimizer_steps':1,
        'gradient_norm_before_clip':gradient_norm,'gradient_norms':gradients,
        'family_nonzero_gradient_tensor_counts':{k:sum(x>0 for x in v) for k,v in group_norms.items()},
        'changed_B_names':changed_b,'changed_B_families':changed_families,
        'before_adapter_sha256':json_sha(before),'after_adapter_sha256':json_sha(after),
        'result_type':'longest_sample_cost_probe_not_training_or_score'})
    status('save_probe_adapter')
    adapter = output/'adapter'; model.save_pretrained(adapter,safe_serialization=True)
    adapter_files = {p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in adapter.iterdir() if p.is_file()}
    if not {'adapter_model.safetensors','adapter_config.json'}.issubset(adapter_files):
        raise RuntimeError('Probe adapter safe weights/config were not both saved')
    write(output/'adapter-checkpoint.json',{'adapter_files':adapter_files,'adapter_state_sha256':json_sha(after),
        'only_cost_probe':True,'optimizer_state_saved':False,'resume_tested':False,'fresh_base_used':True})
    fb = stages['forward']['seconds']+stages['backward']['seconds']
    update = stages['gradient_checks_and_clip']['seconds']+stages['optimizer_first_step']['seconds']
    longest_step = fb+update
    estimates = {'measured_longest_forward_backward_seconds':fb,
        'measured_longest_update_including_gradient_diagnostics_seconds':update,
        'measured_one_sample_step_seconds':longest_step,
        'assumed_two_pass_microbatches':530,'assumed_accum8_optimizer_updates':68,
        'conservative_530_times_longest_seconds':530*longest_step,
        'token_scaled_two_pass_seconds':(2*selection['total_full_tokens']/len(encoded['input_ids']))*fb+68*update,
        'these_are_measurements':False,
        'interpretation':'Planning envelope and token-linear estimate from one cold longest sample; neither guarantees total training wall time.',
        'assumptions':'530 examples vs68 accumulated updates; token estimate assumes proportional forward/backward cost. Longest envelope deliberately overcounts optimizer updates.',
        'excluded':'Base loading/hash checks, data pipeline, checkpoints, evaluation, scheduling contention, allocator/kernel variability. No full training was run.'}
    write(output/'training-time-estimates.json',estimates)
    result = {'optimizer_steps':1,'loss':float(loss.detach().item()),'three_families_nonzero_gradient':True,
              'three_families_B_changed':True,'measured_step_seconds':longest_step,
              'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,
              'adapter_parameters':ADAPTER_PARAMETERS,'full_parameters':BASE_PARAMETERS+ADAPTER_PARAMETERS}
    del optimizer,trainable,model,base,batch,loss
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','gpu'],required=True)
    for name in ['train','excluded-input','data-manifest','model','model-source-manifest','output-dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('Use new empty cost-probe output; no overwrite/resume')
    output.mkdir(parents=True,exist_ok=True)
    started=time.time();stage='initialize'
    def status(name):
        nonlocal stage
        stage=name;write(output/'status.json',{'stage':stage,'mode':args.mode,'elapsed_seconds':time.time()-started})
        print(json.dumps({'stage':stage,'mode':args.mode}),flush=True)
    try:
        status('source_data_runtime')
        helper_path=Path(inspect.getfile(helpers))
        if sha(helper_path)!=HELPER_SHA:raise ValueError('Frozen smoke helper source changed')
        if sha(inspect.getfile(trainer))!=TRAINER_SHA:raise ValueError('Frozen full64 data/trainer protocol differs')
        data,train_ids,excluded_ids=trainer.verify_sources_and_data(args)
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM
        if transformers.__version__!='5.20.2' or peft.__version__!='0.21.2':
            raise RuntimeError('Require isolated TF5.20.2/PEFT0.21.2')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        random.seed(SEED);torch.manual_seed(SEED)
        packages={}
        for name in ['torch','transformers','peft','accelerate','tokenizers','huggingface-hub','numpy','safetensors','fla-core','causal-conv1d']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        manifest={'source_sha256':sha(__file__),'helper_source_sha256':HELPER_SHA,
          'full64_trainer_source_sha256':TRAINER_SHA,'copied_cost_source_sha256':COPIED_COST_SHA,
          'full64_preparation_manifest_sha256':trainer.PREPARATION_SHA,'configuration':vars(args),
          'training_sha256':TRAIN_SHA,'excluded_input_sha256':EXCLUDED_SHA,'training_documents':train_ids,
          'excluded_documents':excluded_ids,'seed':SEED,'not_formal_training':True,
          'fixed':{'r':16,'alpha':32,'dropout':.05,'targets':TARGETS,'dtype':'bfloat16','quantization':False,
                   'packing':False,'max_seq_length':6400,'attention':'sdpa','batch_size':1,
                   'optimizer_steps':1,'optimizer':'AdamW','lr':1e-4,'weight_decay':.01,'clip_norm':1.0,
                   'scheduler':None,'gradient_checkpointing':{'use_reentrant':False}},
          'runtime':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                     'torch':torch.__version__,'torch_cuda':torch.version.cuda,'packages':packages,
                     'modeling_source_sha256':sha(inspect.getsourcefile(Qwen3_5ForCausalLM)),
                     'peft_model_source_sha256':sha(inspect.getsourcefile(peft.PeftModel))}}
        write(output/'run-manifest.json',manifest)
        status('actual_model_provenance')
        manifest['base']=model_fingerprint(args.model,args.model_source_manifest,args.mode=='gpu')
        write(output/'run-manifest.json',manifest)
        status('full_token_mask_longest_selection')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        selection=select_longest(data,tokenizer,output)
        manifest['selection_sha256']=sha(output/'selection.json')
        manifest['tokenizer']={'class':type(tokenizer).__name__,'eos_token_id':tokenizer.eos_token_id,
                               'pad_token_id':tokenizer.pad_token_id,'all_encoded_sha256':selection['all_encoded_sha256']}
        status('cpu_causal_loss_check')
        causal_loss_probe(torch,output)
        text_config=Qwen3_5TextConfig.from_pretrained(args.model,local_files_only=True)
        text_config._attn_implementation='sdpa';text_config.use_cache=False
        manifest['text_config']=text_config.to_dict();write(output/'run-manifest.json',manifest)
        result={'selected_row_id':selection['row']['row_id'],'full_tokens':len(selection['encoded']['input_ids']),
                'prefix_tokens':selection['encoded']['prefix_tokens'],'supervised_tokens':selection['encoded']['supervised_tokens']}
        if args.mode=='cpu':
            status('meta_full_target_coverage')
            with torch.device('meta'):
                base=Qwen3_5ForCausalLM(text_config).to(dtype=torch.bfloat16)
                model,trainable,coverage=attach_adapter(base,torch,peft)
            if coverage['all_parameter_devices']!=['meta'] or torch.cuda.is_initialized():
                raise RuntimeError('CPU meta audit materialized non-meta/GPU parameters')
            write(output/'meta-coverage.json',coverage)
            result.update(adapter_parameters=ADAPTER_PARAMETERS,total_parameters=BASE_PARAMETERS+ADAPTER_PARAMETERS,
                          cuda_initialized=False)
            del trainable,model,base
        else:
            result.update(gpu_probe(args,selection,output,status,torch,peft,text_config))
        status('complete')
        result.update(mode=args.mode,elapsed_seconds=time.time()-started,source_sha256=sha(__file__),
                      run_manifest_sha256=sha(output/'run-manifest.json'),cost_probe_only=True,
                      result_type='cpu_preparation' if args.mode=='cpu' else 'one_longest_train_sample_gpu_cost')
        write(output/'complete.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write(output/'failure.json',{'stage':stage,'exception':type(error).__name__,'message':str(error),
              'elapsed_seconds':time.time()-started,'source_sha256':sha(__file__),'complete':False,
              'not_formal_training':True})
        print(json.dumps({'failed_stage':stage,'exception':type(error).__name__}),flush=True)
        raise


if __name__=='__main__':main()
