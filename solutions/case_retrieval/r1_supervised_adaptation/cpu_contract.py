"""R1 SFT readiness: cached data/tokenizers, meta-only module inventory, toy CPU CE.

No model weight values, neural forward, GPU, quality evaluation, or training.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def checked(reference):
    path = Path(reference['path'])
    assert path.stat().st_size == reference['bytes'] and sha(path) == reference['sha256']
    return path


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan['output_dir']);out.mkdir(exist_ok=False)
    result = dict(status='RUNNING_CPU_ONLY_CONTRACT',source=ref(__file__),plan=ref(args.plan),
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert plan['source_sha256'] == sha(__file__)
        for reference in plan['refs'].values():
            checked(reference)
        data_bytes = gzip.decompress(Path(plan['refs']['train_gzip']['path']).read_bytes())
        assert hashlib.sha256(data_bytes).hexdigest() == plan['train_uncompressed_sha256']
        training = [json.loads(line) for line in data_bytes.decode().splitlines()]
        prompts_bytes = gzip.decompress(Path(plan['refs']['F1_prompts_gzip']['path']).read_bytes())
        assert hashlib.sha256(prompts_bytes).hexdigest() == plan['F1_prompts_uncompressed_sha256']
        prompts = [json.loads(line) for line in prompts_bytes.decode().splitlines()]
        assert len(training) == 124 and len(prompts) == 80
        assert {r['pmc_id'] for r in training} == set(plan['train32_ids'])
        assert {r['pmc_id'] for r in prompts} == set(plan['F1_ids'])
        assert not set(plan['train32_ids']) & (set(plan['F1_ids']) | set(plan['F0_ids']))
        assert len(set(plan['train32_ids'])) == 32 and len(set(plan['F1_ids'])) == 16

        import torch
        import transformers
        import tokenizers
        import peft
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
        from peft import LoraConfig, get_peft_model
        assert not torch.cuda.is_initialized()
        assert transformers.__version__ == '5.20.2' and tokenizers.__version__ == '0.23.1'
        tc = json.loads(Path(plan['refs']['tokenizer_config']['path']).read_text())
        original = json.loads(Path(plan['refs']['config']['path']).read_text())
        def special(name):
            value=tc.get(name)
            return value.get('content') if isinstance(value,dict) else value
        raw = tokenizers.Tokenizer.from_file(plan['refs']['tokenizer']['path'])
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=plan['refs']['tokenizer']['path'],
            bos_token=special('bos_token'),eos_token=special('eos_token'),
            pad_token=special('pad_token'),unk_token=special('unk_token'),
            chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,
            model_max_length=tc['model_max_length'])
        a,b=json.loads(raw.to_str()),json.loads(tokenizer.backend_tokenizer.to_str())
        assert all(a.get(k)==b.get(k) for k in ['model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'])
        assistant_id = raw.token_to_id('<｜Assistant｜>')
        think_ids = [raw.token_to_id('<think>'),raw.token_to_id('</think>')]
        eos = tokenizer.eos_token_id
        assert assistant_id == 151670 and think_ids == [151667,151668] and eos == 151645
        strict = module('strict_final',plan['refs']['strict_parser']['path'])
        normalizer = module('normalized_final',plan['refs']['normalizer']['path'])
        previous_F1 = [json.loads(line) for line in gzip.decompress(
            Path(plan['refs']['existing_F1_encoded']['path']).read_bytes()).decode().splitlines()]
        assert len(previous_F1) == 80
        encoded, audits, F1_audits = [], [], []
        for index,row in enumerate(training + prompts):
            is_train = index < 124
            expected_keys = {'row_id','pmc_id','code_map','patient_ids','messages'} | ({'answer'} if is_train else set())
            assert set(row) == expected_keys and [m['role'] for m in row['messages']] == ['system','user']
            assert row['patient_ids'] and len(row['patient_ids']) == len(set(row['patient_ids']))
            prefix = tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
            assert prefix == special('bos_token') + row['messages'][0]['content'] + '<｜User｜>' + row['messages'][1]['content'] + '<｜Assistant｜>'
            raw_ids = raw.encode(prefix,add_special_tokens=False).ids
            generic_ids = tokenizer.encode(prefix,add_special_tokens=False)
            native_ids = tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
            assert raw_ids == generic_ids == native_ids and raw_ids[-1] == assistant_id
            assert all(t not in raw_ids for t in think_ids)
            assert raw.decode(raw_ids,skip_special_tokens=False) == prefix
            assert tokenizer.decode(generic_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False) == prefix
            if not is_train:
                prior = previous_F1[index-124]
                assert prior['row_id'] == row['row_id'] and prior['input_ids'] == raw_ids
                F1_audits.append(dict(row_id=row['row_id'],tokens=len(raw_ids),input_ids_sha256=ids_sha(raw_ids),
                                     equal_existing_correct_published_F1=True,think_prefix_open=False))
                continue
            answer = row['answer'];answer_object=json.loads(answer,object_pairs_hook=strict.unique)
            assert set(answer_object) == set(row['code_map']) and '<think>' not in answer and '</think>' not in answer
            parsed = normalizer.parse_normalized(answer,eos_observed=True,stopped_at_cap=False,
                        code_map=row['code_map'],patient_ids=row['patient_ids'])
            assert parsed['provided_mapping'] == answer_object and parsed['normalized_every_CID']
            assert not parsed['semantic_answer_repair_performed'] and not parsed['integer_PID_aliases_applied']
            full = tokenizer.apply_chat_template(row['messages']+[dict(role='assistant',content=answer)],
                                                  tokenize=False,add_generation_prompt=False)
            assert full == prefix + answer + special('eos_token')
            full_ids = raw.encode(full,add_special_tokens=False).ids
            other_ids = tokenizer.encode(full,add_special_tokens=False)
            chat_ids = tokenizer.apply_chat_template(row['messages']+[dict(role='assistant',content=answer)],
                tokenize=True,add_generation_prompt=False,return_dict=False)
            assert full_ids == other_ids == chat_ids and full_ids[:len(raw_ids)] == raw_ids
            assert full_ids[-1] == eos and eos not in full_ids[len(raw_ids):-1]
            assert tokenizer.decode(full_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False) == full
            assert raw.decode(full_ids,skip_special_tokens=False) == full
            suffix = tokenizer.decode(full_ids[len(raw_ids):],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert suffix == answer + special('eos_token')
            labels = [-100]*len(raw_ids) + full_ids[len(raw_ids):]
            assert len(labels) == len(full_ids) and labels[-1] == eos
            assert labels[:len(raw_ids)] == [-100]*len(raw_ids)
            item = dict(row_id=row['row_id'],pmc_id=row['pmc_id'],input_ids=full_ids,
                        labels=labels,prompt_length=len(raw_ids))
            encoded.append(item)
            audits.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],tokens=len(full_ids),
                prompt_tokens=len(raw_ids),supervised_tokens=len(full_ids)-len(raw_ids),
                input_ids_sha256=ids_sha(full_ids),labels_sha256=ids_sha(labels),
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                prefix_ends_assistant_without_think=True,full_encode_prefix_exact=True,
                supervised_json_and_native_EOS_only=True,original_parser_accepts_complete_target=True))
        payload=''.join(json.dumps(item,separators=(',',':'))+'\n' for item in encoded).encode()
        (out/'train124.encoded.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
        save(out/'train-token-audit.json',audits);save(out/'F1-token-audit.json',F1_audits)
        lengths=[item['tokens'] for item in audits]
        longest=max(range(124),key=lambda i:lengths[i]);shortest=min(range(124),key=lambda i:lengths[i])
        result['token_contract']=dict(train_rows=124,train_documents=32,F1_rows=80,F1_documents=16,
            total_train_tokens=sum(lengths),total_supervised_tokens=sum(x['supervised_tokens'] for x in audits),
            min_train_tokens=min(lengths),max_train_tokens=max(lengths),longest_index=longest,
            longest_row_id=audits[longest]['row_id'],shortest_index=shortest,shortest_row_id=audits[shortest]['row_id'],
            proposed_reject_cap=8192,all_fit_proposed_cap=max(lengths)<=8192,truncate=False,
            all124_full_encode_prefix_and_target_EOS_exact=True,all80_match_existing_correct_R1_ids=True,
            prefix_think_open_rows=0,supervision_policy='Original closed-set JSON answer + nativeEOS; no fabricated or empty reasoning. Original template does not pre-open think.',
            outputs={name:ref(out/name) for name in ['train124.encoded.jsonl.gz','train-token-audit.json','F1-token-audit.json']})

        header_shapes={}
        for item in plan['weight_headers']:
            path=Path(item['path']);assert path.stat().st_size == item['bytes']
            with path.open('rb') as stream:
                header_bytes=struct.unpack('<Q',stream.read(8))[0]
                header=json.loads(stream.read(header_bytes))
            for name,value in header.items():
                if name=='__metadata__':continue
                assert name not in header_shapes and value['dtype']=='BF16'
                header_shapes[name]=value['shape']
        assert len(header_shapes)==399 and sum(math.prod(v) for v in header_shapes.values())==8190735360
        targets=plan['lora']['target_modules'];rank=plan['lora']['r']
        target_shapes={k:v for k,v in header_shapes.items() if k.endswith('.weight') and k.split('.')[-2] in targets}
        assert len(target_shapes)==36*7 and all(len(v)==2 for v in target_shapes.values())
        lora_count=sum(rank*sum(shape) for shape in target_shapes.values())
        config=AutoConfig.from_pretrained(plan['refs']['derived_config']['path'],local_files_only=True,trust_remote_code=False)
        config.use_cache=False;config._attn_implementation='sdpa'
        with torch.device('meta'):
            base=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,trust_remote_code=False)
            assert type(base).__name__=='Qwen3ForCausalLM'
            assert sum(p.numel() for p in base.parameters())==8190735360
            wrapped=get_peft_model(base,LoraConfig(**plan['lora']))
        trainable={name:p for name,p in wrapped.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in trainable.values())==lora_count
        assert all('.lora_A.' in n or '.lora_B.' in n for n in trainable)
        assert all(p.device.type=='meta' for p in wrapped.parameters())
        for p in trainable.values():p.data=p.data.float()
        assert all(p.dtype==torch.float32 for p in trainable.values())
        wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assert base.is_gradient_checkpointing and not base.config.use_cache
        forward_source=inspect.getsource(type(base).forward)
        assert 'self.lm_head(' in forward_source and 'self.loss_function(' in forward_source
        (out/'actual-qwen3-forward.txt').write_text(forward_source)
        native_loss=base.loss_function
        selected=module('selected_loss',plan['refs']['selected_loss']['path']).selected_causal_loss
        base.loss_function=selected
        assert base.loss_function is selected
        torch.manual_seed(20261024)
        loss_checks=[]
        for dtype in [torch.float32,torch.bfloat16]:
            values=torch.randn(2,7,19,dtype=dtype)
            labels=torch.tensor([[-100,-100,4,2,5,-100,1],[-100,3,-100,8,9,1,-100]],dtype=torch.long)
            for explicit in [False,True]:
                first=values.clone().requires_grad_();second=values.clone().requires_grad_()
                kwargs=dict(logits=first,labels=labels,vocab_size=19)
                if explicit:kwargs['num_items_in_batch']=torch.tensor(8)
                a=native_loss(**kwargs);kwargs['logits']=second;b=base.loss_function(**kwargs)
                a.backward();b.backward()
                assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
                assert torch.allclose(first.grad,second.grad,atol=1e-6,rtol=1e-6)
                loss_checks.append(dict(dtype=str(dtype),explicit_denominator=explicit,
                    native_loss=float(a),selected_loss=float(b),max_abs_gradient_difference=float((first.grad-second.grad).abs().max())))
        assert not torch.cuda.is_initialized()
        result['parameter_contract']=dict(actual_weight_header_tensors=399,base_parameters=8190735360,
            base_bytes_BF16=2*8190735360,target_linear_modules=len(target_shapes),lora_parameters=lora_count,
            total_base_plus_lora=8190735360+lora_count,less_than_10B=8190735360+lora_count<10_000_000_000,
            nominal_lora_FP32_weight_gradient_two_Adam_moments_bytes=16*lora_count,
            nominal_fixed_training_resident_bytes_excluding_activations=2*8190735360+16*lora_count,
            meta_instantiation_only=True,actual_weight_values_read=False,
            trainable_parameters_only_lora=True,base_head_and_embedding_frozen=True,
            gradient_checkpointing_enabled_on_meta_model=True,use_reentrant=False,use_cache=False,
            actual_model_class=type(base).__name__,actual_lm_head_class=type(base.lm_head).__name__,
            actual_head_shape=list(base.lm_head.weight.shape),
            loss_hook_reachable_in_actual_forward_source=True,actual_forward=ref(out/'actual-qwen3-forward.txt'),
            CPU_toy_loss_gradient_checks=loss_checks,real_model_forward_or_backward_performed=False,
            remaining_required='Real BF16 model/LoRA gradients, actual Adam accumulation and maximum-length A30 memory require separate GPU probe; meta/toy checks do not establish feasibility.')
        result.update(status='PASS_R1_SFT_CPU_DATA_TOKEN_BOUNDARY_META_PARAMETER_AND_TOY_LOSS_CONTRACT',
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,peft=peft.__version__),
            CUDA_initialized=False,NN_model_weight_values_loaded=False,train_optimizer_steps=0,
            new_quality_calls=0,F0_or_dev_gold_read=False)
    except BaseException as error:
        result.update(status='FAILED_CPU_CONTRACT_PRESERVED',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),program_seconds=time.monotonic()-started)
        save(out/'result.json',result)
        print(json.dumps({k:result.get(k) for k in ['status','program_seconds','error','token_contract','parameter_contract']},ensure_ascii=False))


if __name__=='__main__':main()
