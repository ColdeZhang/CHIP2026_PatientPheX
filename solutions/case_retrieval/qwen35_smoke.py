#!/usr/bin/env python3
"""One text-only Qwen3.5 interface smoke, never a training or scoring entrypoint.

CPU mode audits every real training row and selects the shortest nonempty target
with <=512 full tokens. GPU mode repeats the audit, performs one optimizer step,
greedy-generates at most 16 tokens, then unloads/reloads the saved LoRA adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
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

REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
MAX_TOKENS = 512
SEED = 20261023
TARGETS = ('in_proj_qkv', 'q_proj', 'v_proj')
CPU_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
             'vocab.json', 'merges.txt', 'chat_template.jinja')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def nonempty(value):
    if isinstance(value, dict):
        return any(nonempty(v) for v in value.values())
    if isinstance(value, list):
        return any(nonempty(v) for v in value)
    return bool(value)


def training_rows(train, excluded):
    data, heldout = rows(train), rows(excluded)
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
    if len(document_ids) != 48 or set(document_ids) & set(heldout_ids):
        raise ValueError('Require the actual 48-document training fold, disjoint from excluded input')
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


def encode(row, tokenizer):
    kwargs = dict(enable_thinking=False)
    prefix_text = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                               add_generation_prompt=True, **kwargs)
    full_messages = row['messages'] + [{'role':'assistant','content':row['answer']}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False,
                                             add_generation_prompt=False, **kwargs)
    if not full_text.startswith(prefix_text):
        raise ValueError('Rendered assistant prefix mismatch: '+row['row_id'])
    eos = tokenizer.eos_token
    if not eos or full_text[len(prefix_text):] != row['answer'].strip()+eos+'\n':
        raise ValueError('Completion is not exact answer plus chat EOS/newline: '+row['row_id'])
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                          add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(full_messages, tokenize=True, return_dict=False,
                                        add_generation_prompt=False, **kwargs)
    if not isinstance(full, list) or not all(isinstance(x,int) for x in full):
        raise ValueError('Unexpected tokenizer output type')
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Token prefix mismatch or empty completion: '+row['row_id'])
    suffix = full[len(prefix):]
    if suffix.count(tokenizer.eos_token_id) != 1:
        raise ValueError('Supervised answer must contain exactly one chat EOS: '+row['row_id'])
    labels = [-100]*len(prefix) + suffix
    if labels[:len(prefix)] != [-100]*len(prefix) or labels[len(prefix):] != full[len(prefix):]:
        raise ValueError('Assistant-only loss mask is incorrect')
    if not any(x != -100 for x in labels[1:]):
        raise ValueError('No causally supervised tokens')
    return {'row_id':row['row_id'], 'input_ids':full, 'labels':labels,
            'prefix_tokens':len(prefix), 'supervised_tokens':len(suffix),
            'eos_token_id':tokenizer.eos_token_id,
            'eos_supervised_position':len(prefix)+suffix.index(tokenizer.eos_token_id),
            'rendered_prefix_sha256':hashlib.sha256(prefix_text.encode()).hexdigest(),
            'rendered_full_sha256':hashlib.sha256(full_text.encode()).hexdigest()}


def audit_selection(data, tokenizer, output):
    audit = []; candidates = []; selected = None
    for row in data:
        encoded = encode(row, tokenizer)
        positive = nonempty(json.loads(row['answer']))
        audit.append({'row_id':row['row_id'],'pmc_id':str(row['pmc_id']),
                      'full_tokens':len(encoded['input_ids']),
                      'prefix_tokens':encoded['prefix_tokens'],
                      'supervised_tokens':encoded['supervised_tokens'],
                      'nonempty_target':positive,'prefix_mask_eos_passed':True})
        if positive and len(encoded['input_ids']) <= MAX_TOKENS:
            candidates.append((len(encoded['input_ids']),row['row_id'],row,encoded))
    write(output/'token-mask-audit.json', {'rows':audit,'all_rows_no_truncation':True,
          'eligible_nonempty_le512':len(candidates),'audit_sha256':json_sha(audit)})
    if not candidates:
        raise ValueError('No nonempty real train48 sample fits 512 tokens; no truncation or synthetic fallback')
    _, _, row, selected = min(candidates, key=lambda x:(x[0],x[1]))
    selection = {'rule':'Shortest full tokenized nonempty answer <=512; row_id tie-break; no label evaluation',
                 'row':row,'encoded':selected,'encoded_sha256':json_sha(selected),
                 'eligible_candidates':len(candidates),'total_training_rows':len(data)}
    write(output/'selection.json', selection)
    return selection


def model_fingerprint(directory, provenance, require_weights):
    source = json.loads(Path(provenance).read_text())
    if source.get('model') != 'Qwen/Qwen3.5-9B' or source.get('hf_revision') != REVISION:
        raise ValueError('Unexpected official model/revision provenance')
    expected = {x['name']:x for x in source['files']}
    required = set(CPU_FILES)
    if require_weights:
        required |= {n for n in expected if n.endswith('.safetensors') or n.endswith('.index.json')}
        if not any(n.endswith('.safetensors') for n in required):
            raise ValueError('No weight files in official manifest')
    actual = {}
    for name in sorted(required):
        path = Path(directory)/name
        if not path.is_file():
            raise FileNotFoundError('Missing required model file: '+name)
        actual[name] = {'bytes':path.stat().st_size,'sha256':sha(path)}
        if actual[name] != {'bytes':expected[name]['size'],'sha256':expected[name]['sha256']}:
            raise ValueError('Actual model bytes disagree with source: '+name)
    return {'model':'Qwen/Qwen3.5-9B','hf_revision':REVISION,
            'source_manifest_sha256':sha(provenance),'actual_files':actual,
            'weight_bytes_verified':require_weights}


def tensor_fingerprints(state, torch):
    result = {}
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        result[name] = {'shape':list(value.shape),'dtype':str(value.dtype),
                       'sha256':hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def gpu_smoke(args, selection, tokenizer, output, status, torch, transformers, peft):
    if not args.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('GPU mode requires the explicitly allocated UUID as sole CUDA_VISIBLE_DEVICES')
    check = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                            '--format=csv,noheader'],capture_output=True,text=True,check=True)
    write(output/'gpu-preflight.json', {'processes':check.stdout,'uuid':args.gpu_uuid})
    if any(line.split(',')[0].strip() == args.gpu_uuid for line in check.stdout.splitlines()):
        raise RuntimeError('Allocated GPU already has a process; refuse to share')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Need exactly one visible BF16 CUDA device')
    status('load_text_base')
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    text_config = Qwen3_5TextConfig.from_pretrained(args.model, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=text_config, dtype=torch.bfloat16, device_map={'':'cuda:0'},
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False,
        output_loading_info=True)
    write(output/'base-loading-info.json', loading)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('Text-only base has missing/mismatched/error loading keys')
    unexpected = loading.get('unexpected_keys', [])
    if any(not (key.startswith(('mtp.','model.mtp.','model.visual.','visual.'))) for key in unexpected):
        raise RuntimeError('Unexpected non-vision/non-MTP checkpoint weights')
    if model.__class__.__name__ != 'Qwen3_5ForCausalLM' or any('visual' in name for name,_ in model.named_modules()):
        raise RuntimeError('Must instantiate the text-only model class')
    base_count = sum(p.numel() for p in model.parameters())
    if base_count != 8_953_803_264:
        raise RuntimeError('Text base parameter count differs from official checkpoint accounting')
    targets = {name:module for name,module in model.named_modules() if name.rsplit('.',1)[-1] in TARGETS}
    target_counts = Counter(name.rsplit('.',1)[-1] for name in targets)
    layers = Counter(text_config.layer_types)
    expected_counts = {'in_proj_qkv':layers['linear_attention'],'q_proj':layers['full_attention'],
                       'v_proj':layers['full_attention']}
    if dict(target_counts) != expected_counts or not all(isinstance(x,torch.nn.Linear) for x in targets.values()):
        raise RuntimeError('LoRA projection names/counts differ from the reviewed hybrid architecture')
    model.config.use_cache = False
    model = get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=list(TARGETS),bias='none',task_type='CAUSAL_LM'))
    trainable = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if not trainable or any('.lora_A.' not in n and '.lora_B.' not in n for n in trainable):
        raise RuntimeError('Unexpected trainable base parameters')
    total_count = sum(p.numel() for p in model.parameters())
    if not total_count < 10_000_000_000:
        raise RuntimeError('Actual base plus adapter violates <10B parameter cap')
    before = tensor_fingerprints(get_peft_model_state_dict(model),torch)
    write(output/'model-parameters.json', {'base_parameters':base_count,'total_with_adapter':total_count,
          'trainable_parameters':sum(p.numel() for p in trainable.values()),
          'target_counts':dict(target_counts),'target_module_names':sorted(targets),
          'trainable_names':sorted(trainable),'gpu':torch.cuda.get_device_name(0),
          'gpu_uuid':args.gpu_uuid,'gpu_compute_capability':list(torch.cuda.get_device_capability(0)),
          'before_adapter_tensors':before,'text_config':text_config.to_dict()})
    encoded = selection['encoded']; ids = torch.tensor([encoded['input_ids']],device='cuda:0')
    batch = {'input_ids':ids,'attention_mask':torch.ones_like(ids),
             'labels':torch.tensor([encoded['labels']],device='cuda:0')}
    optimizer = torch.optim.AdamW(list(trainable.values()),lr=1e-4,weight_decay=0.0)
    status('single_optimizer_step')
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = model(**batch,use_cache=False).loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    gradients = {name:float(p.grad.detach().float().norm().item()) for name,p in trainable.items() if p.grad is not None}
    if not gradients or not all(math.isfinite(x) for x in gradients.values()) or not any(x>0 for x in gradients.values()):
        raise RuntimeError('Nonfinite or all-zero adapter gradient')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen base received a gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0).item())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError('Nonfinite or zero aggregate gradient norm')
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if not all(bool(torch.isfinite(p).all()) for p in trainable.values()):
        raise RuntimeError('Nonfinite adapter tensor after optimizer step')
    after = tensor_fingerprints(get_peft_model_state_dict(model),torch)
    changed_b = [name for name in before if 'lora_B' in name and before[name] != after[name]]
    if not changed_b:
        raise RuntimeError('No LoRA B tensor changed after the optimizer step')
    write(output/'single-step.json',{'loss':float(loss.detach().item()),'gradient_norm_before_clip':norm,
          'nonzero_gradient_tensors':sum(x>0 for x in gradients.values()),'gradient_norms':gradients,
          'changed_B_tensors':changed_b,'before_adapter_sha256':json_sha(before),
          'after_adapter_sha256':json_sha(after),'optimizer_steps':1,
          'result_type':'interface_only_not_generalization_or_competition_score'})
    status('save_adapter')
    adapter = output/'adapter'; model.save_pretrained(adapter,safe_serialization=True)
    adapter_files = {p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in adapter.iterdir() if p.is_file()}
    if 'adapter_model.safetensors' not in adapter_files or 'adapter_config.json' not in adapter_files:
        raise RuntimeError('Incomplete saved adapter')
    status('unload_reload_adapter')
    base = model.unload()
    del optimizer, trainable, targets, model, loss, batch, ids
    gc.collect(); torch.cuda.empty_cache()
    if any('lora_' in name for name,_ in base.named_modules()):
        raise RuntimeError('Adapter modules remain after unload')
    model = PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=True)
    reloaded = tensor_fingerprints(get_peft_model_state_dict(model),torch)
    if after != reloaded:
        raise RuntimeError('Saved/reloaded adapter tensor bytes do not match')
    write(output/'adapter-reload.json', {'verification':'all named adapter tensors exact shape/dtype/SHA',
          'base_reloaded':False,'adapter_unloaded_then_loaded_from_disk':True,
          'trained_adapter_state_sha256':json_sha(after),'reloaded_adapter_state_sha256':json_sha(reloaded),
          'adapter_files':adapter_files,'tensor_fingerprints':reloaded})
    status('greedy_generation')
    model.eval()
    prompt = torch.tensor([encoded['input_ids'][:encoded['prefix_tokens']]],device='cuda:0')
    config_eos = text_config.eos_token_id
    generation_eos = sorted({tokenizer.eos_token_id} | set(
        config_eos if isinstance(config_eos,list) else [config_eos]))
    with torch.no_grad():
        generated = model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
            max_new_tokens=16,do_sample=False,use_cache=True,
            eos_token_id=generation_eos,pad_token_id=tokenizer.pad_token_id)
    new_tokens = generated[0,prompt.shape[1]:].detach().cpu().tolist()
    if not 1 <= len(new_tokens) <= 16:
        raise RuntimeError('Unexpected greedy generation length')
    write(output/'generation.json', {'row_id':encoded['row_id'],'input_is_training_sample':True,
          'max_new_tokens':16,'do_sample':False,'use_cache':True,'token_ids':new_tokens,
          'text':tokenizer.decode(new_tokens,skip_special_tokens=False),
          'eos_token_id':generation_eos,'training_chat_eos_token_id':tokenizer.eos_token_id,
          'pad_token_id':tokenizer.pad_token_id,
          'quality_not_evaluated':True})
    status('forced_chat_eos_engine_probe')
    with torch.no_grad():
        stopped = model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
            max_new_tokens=16,do_sample=False,use_cache=True,eos_token_id=generation_eos,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=lambda batch_id,input_ids:[tokenizer.eos_token_id])
    stop_tokens = stopped[0,prompt.shape[1]:].detach().cpu().tolist()
    write(output/'generation-stop-check.json', {
        'purpose':'Forced im_end engine stopping probe, not natural model EOS behavior or quality',
        'row_id':encoded['row_id'],'input_is_same_training_prompt':True,'max_new_tokens':16,
        'do_sample':False,'eos_token_id':generation_eos,
        'prefix_allowed_tokens_fn':'Only tokenizer.eos_token_id is allowed at each generation step',
        'allowed_token_id':tokenizer.eos_token_id,'actual_new_token_ids':stop_tokens,
        'stopped_after_one_token':stop_tokens==[tokenizer.eos_token_id],
        'normal_generation_unchanged':True})
    if stop_tokens != [tokenizer.eos_token_id]:
        raise RuntimeError('Forced im_end did not terminate generation after exactly one token')
    result = {'optimizer_steps':1,'loss':float(json.loads((output/'single-step.json').read_text())['loss']),
              'nonzero_gradient':True,'B_changed':True,'adapter_reload_exact':True,
              'forced_chat_eos_engine_stop_verified':True,
              'greedy_new_tokens':len(new_tokens),'parameters':total_count,
              'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
    del model,base,prompt,generated,stopped
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','gpu'],required=True)
    for name in ['train','excluded-input','model','model-source-manifest','output-dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-uuid',help='Required in GPU mode; must be the sole allocated visible UUID')
    args = parser.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.mode == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; smoke has no resume or overwrite')
    output.mkdir(parents=True,exist_ok=True)
    started = time.time(); current_stage = 'initialize'; manifest = {'schema':'qwen35-interface-smoke-v1',
        'configuration':vars(args),'source_sha256':sha(__file__),'seed':SEED,'mode':args.mode,
        'fixed':{'max_sample_tokens':MAX_TOKENS,'r':8,'alpha':16,'dropout':0.0,'targets':list(TARGETS),
                 'dtype':'bfloat16','attention':'sdpa','packing':False,'gradient_checkpointing':False,
                 'optimizer':'AdamW','lr':1e-4,'weight_decay':0.0,'clip_grad_norm':1.0,
                 'optimizer_steps':1,'greedy_max_new_tokens':16,
                 'generation_eos_policy':'sorted unique tokenizer chat EOS plus text config EOS'},'smoke_only':True}

    def status(stage):
        nonlocal current_stage
        current_stage = stage
        write(output/'status.json',{'stage':stage,'elapsed_seconds':time.time()-started,'mode':args.mode})
        print(json.dumps({'stage':stage,'mode':args.mode}),flush=True)

    try:
        status('data_and_runtime')
        data, document_ids, excluded_ids = training_rows(args.train,args.excluded_input)
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        if transformers.__version__ != '5.20.2' or peft.__version__ != '0.21.2':
            raise RuntimeError('Require isolated transformers==5.20.2 and peft==0.21.2')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        random.seed(SEED); torch.manual_seed(SEED)
        if args.mode == 'cpu' and torch.cuda.is_initialized():
            raise RuntimeError('CPU audit unexpectedly initialized CUDA')
        packages = {}
        for name in ['torch','transformers','peft','accelerate','tokenizers','huggingface-hub','numpy','safetensors','fla-core','flash-linear-attention','causal-conv1d']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        source_paths = {'qwen35_modeling':inspect.getsourcefile(Qwen3_5ForCausalLM),
                        'peft_model':inspect.getsourcefile(peft.PeftModel),
                        'peft_lora_layer':inspect.getsourcefile(peft.tuners.lora.layer.Linear)}
        manifest.update(runtime={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                                 'packages':packages,'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
                                 'source_files':{k:{'path':p,'sha256':sha(p)} for k,p in source_paths.items()}},
                        data={'train_sha256':sha(args.train),'excluded_input_sha256':sha(args.excluded_input),
                              'training_document_ids':document_ids,'excluded_document_ids':excluded_ids,
                              'training_rows':len(data),'documents_disjoint':True})
        write(output/'run-manifest.json',manifest)
        status('model_provenance')
        manifest['base'] = model_fingerprint(args.model,args.model_source_manifest,args.mode=='gpu')
        write(output/'run-manifest.json',manifest)
        status('tokenizer_mask_selection')
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            raise ValueError('Expected official tokenizer pad token')
        selection = audit_selection(data,tokenizer,output)
        manifest['selection_sha256'] = sha(output/'selection.json')
        manifest['tokenizer'] = {'class':type(tokenizer).__name__,'eos_token':tokenizer.eos_token,
                                 'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,
                                 'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
        write(output/'run-manifest.json',manifest)
        result = {'selected_row_id':selection['row']['row_id'],'full_tokens':len(selection['encoded']['input_ids']),
                  'prefix_tokens':selection['encoded']['prefix_tokens'],
                  'supervised_tokens':selection['encoded']['supervised_tokens']}
        if args.mode == 'gpu':
            result.update(gpu_smoke(args,selection,tokenizer,output,status,torch,transformers,peft))
        elif torch.cuda.is_initialized():
            raise RuntimeError('CPU audit must not initialize CUDA')
        status('complete')
        result.update(mode=args.mode,elapsed_seconds=time.time()-started,smoke_only=True,
                      cuda_initialized=torch.cuda.is_initialized(),source_sha256=sha(__file__),
                      run_manifest_sha256=sha(output/'run-manifest.json'),
                      result_type='cpu_tokenizer_audit' if args.mode=='cpu' else 'gpu_interface_smoke_not_score')
        write(output/'complete.json',result)
        print(json.dumps(result),flush=True)
    except Exception as error:
        write(output/'failure.json',{'stage':current_stage,'type':type(error).__name__,
              'message':str(error),'elapsed_seconds':time.time()-started,'source_sha256':sha(__file__),
              'smoke_only':True,'mode':args.mode,'complete':False})
        print(json.dumps({'failed_stage':current_stage,'type':type(error).__name__}),flush=True)
        raise


if __name__ == '__main__':
    main()
