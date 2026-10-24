"""One fixed on-policy SCST supply/gradient probe. No held data or quality.

All eight trajectories are generated at the unchanged parent. The controller
normalizes them with the original public mapper while this model waits. Only
the prelocked supply gate permits a single disposable Adam update.
"""
import argparse,contextlib,gzip,importlib.util,inspect,json,math,os,random,shutil,subprocess,time,traceback,types
from pathlib import Path
from common import admit,atomic,checked,ids_sha,now,ref,sha
from scst_ops import disable_dropout,token_logps,replay_observation,family

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',required=True);ap.add_argument('--admission',required=True)
    ap.add_argument('--TERM-unix',type=float,required=True);ap.add_argument('--KILL-unix',type=float,required=True);a=ap.parse_args()
    ep=Path(a.execution).resolve();plan=json.loads(ep.read_text());ad=json.loads(Path(a.admission).read_text());admit(ad,ep)
    out=ep.parent/'terminal';out.mkdir(exist_ok=False);start=time.monotonic();torch=None;model=None
    r=dict(status='RUNNING_FIXED4_TRAIN_ONLY_SCST_PROBE',started_at=now(),source=ref(__file__),execution=ref(ep),rows=[],optimizer_steps=0,
           real_backward_calls=0,no_held_inputs_or_gold=True,no_quality=True,no_formal_training=True,TERM_unix=a.TERM_unix,KILL_unix=a.KILL_unix)
    def budget():
        if time.time()>=a.TERM_unix-10:raise TimeoutError('Original shared1800 deadline exhausted; no partial PASS')
    try:
        assert a.KILL_unix==a.TERM_unix+30 and plan['schema']=='patient-set-risk-fixed4-probe-v1'
        for name,h in plan['sources'].items():assert sha(ep.parent/name)==h,name
        for v in plan['references'].values():checked(v)
        assert plan['fixed']==dict(indices=[76,44,83,19],max_new_tokens=1600,seed=20261024,shared_seconds=1800,CE_weight=1.,RL_weight=1.,C=11884/87)
        rows=json.loads(gzip.decompress(checked(plan['references']['rows']).read_bytes()))
        assert [x['index'] for x in rows]==[76,44,83,19]
        for x in rows:
            n=x['prefix_tokens'];assert len(x['input_ids'])==len(x['labels']) and all(y==-100 for y in x['labels'][:n])
            assert x['labels'][n:]==x['input_ids'][n:] and len(x['labels'])-n==x['supervised_tokens']
        assert max(x['prefix_tokens'] for x in rows)==8823 and sum(x['supervised_tokens'] for x in rows)==1461
        admit(ad,ep);budget();assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        q=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
        gpu={l.split(',')[0].strip():dict(used=int(l.split(',')[1]),free=int(l.split(',')[2])) for l in q.splitlines()}
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
        resource=dict(at_utc=now(),uuid=plan['gpu_uuid'],**gpu[plan['gpu_uuid']],apps=apps,disk_free=shutil.disk_usage(ep.parent).free)
        atomic(out/'resource-before-import.json',resource)
        assert resource['used']<=64 and not any(l.split(',')[0].strip()==plan['gpu_uuid'] for l in apps.splitlines()),'assigned card has another process'
        assert resource['free']>=38000 and resource['disk_free']>=1024**3,'Requires actual >=38000MiB free plus1GiB durable state space'
        import torch as tm,transformers,peft,tokenizers
        torch=tm
        from transformers import AutoTokenizer,Qwen3_5TextConfig,Qwen3_5ForCausalLM,GenerationConfig,LogitsProcessor,LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
        from peft import PeftModel
        import transformers.models.qwen3_5.modeling_qwen3_5 as native
        assert sha(native.__file__)=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
        assert importlib.util.find_spec('fla') is None
        assert all(inspect.unwrap(getattr(native,n)).__module__==native.__name__ for n in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'])
        r['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__)
        assert r['runtime']==plan['runtime'],r['runtime']
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        manifest=json.loads(checked(plan['references']['model_manifest']).read_text());assert manifest['hf_revision']=='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        modeldir=Path(plan['model']);verified={}
        for x in manifest['files']:
            p=modeldir/x['name'];assert p.stat().st_size==x['size'] and sha(p)==x['sha256']
            s=p.stat();verified[x['name']]=dict(bytes=s.st_size,sha256=x['sha256'],mtime_ns=s.st_mtime_ns,inode=s.st_ino)
        checked(plan['adapter_model']);checked(plan['adapter_config']);r['model_identity']=verified
        tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True);ntok=tokenizers.Tokenizer.from_file(str(modeldir/'tokenizer.json'))
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        for x in rows:
            text=tok.apply_chat_template(x['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert tok.encode(text,add_special_tokens=False)==ntok.encode(text,add_special_tokens=False).ids==x['input_ids'][:x['prefix_tokens']]
        admit(ad,ep);budget();tick=time.monotonic()
        config=Qwen3_5TextConfig.from_pretrained(modeldir,local_files_only=True)
        model,loading=Qwen3_5ForCausalLM.from_pretrained(modeldir,config=config,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        atomic(out/'loading.json',loading);assert all(not loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        assert sum(p.numel() for p in model.parameters())==8953803264
        model=PeftModel.from_pretrained(model,plan['adapter'],is_trainable=True)
        assert sum(p.numel() for p in model.parameters())==8997081600
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in params.values())==43278336 and all('lora_' in n for n in params)
        r['disabled_dropout']=disable_dropout(model);model.eval();r['load_seconds']=time.monotonic()-tick
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None);assert eos
        r['eos_ids']=sorted(eos);r['effective_policy']=dict(autocast='BF16',cache_enabled=False,dropout=0,generation_mode='eval/use_cache=True',replay_mode='train/use_cache=False/checkpoint_nonreentrant',generation_config_inherited=False,normalizer='original_b59_FP16',sampling_distribution='untruncated_T1_fullcategorical')
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False)
        # One previous logits vector (not1600 full vocabulary tensors) captures
        # the actual probability of each selected token, including terminal EOS.
        class Observe(LogitsProcessor):
            def __init__(self,prefix,stem):self.prefix=prefix;self.stem=stem;self.previous=None;self.ids=[];self.logps=[]
            def finish_previous(self,ids):
                if self.previous is not None:
                    target=int(ids[0,-1]);assert ids.shape[1]-self.prefix==len(self.ids)+1
                    self.ids.append(target);self.logps.append(float(torch.log_softmax(self.previous.float(),dim=-1)[0,target]));self.previous=None
            def __call__(self,input_ids,scores):
                self.finish_previous(input_ids)
                if not bool(torch.isfinite(scores).all()):
                    atomic(out/(self.stem+'.nonfinite-logits.json'),dict(shape=list(scores.shape),nonfinite_count=int((~torch.isfinite(scores)).sum()),
                        generated_ids=input_ids[0,self.prefix:].tolist(),cached_selected_logps=self.logps,first_nonfinite_retained=True))
                    raise FloatingPointError('nonfinite original logits')
                self.previous=scores.detach().clone();return scores
        underlying=model.get_base_model();original_processor=underlying._get_logits_processor;processor_audits=[]
        def audited_processors(self,*args,**kwargs):
            processors=original_processor(*args,**kwargs)
            names=[type(p).__name__ for p in processors];processor_audits.append(names)
            assert len(processors)==1 and isinstance(processors[0],Observe),('hidden generation processors',names)
            return processors
        underlying._get_logits_processor=types.MethodType(audited_processors,underlying)
        records=[]
        for row in rows:
            for arm in ('greedy','sample'):
                budget();n=row['prefix_tokens'];prefix=row['input_ids'][:n];seed=20261024+row['index'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                ids=torch.tensor([prefix],device='cuda',dtype=torch.long);tick=time.monotonic();stem=f"row{row['index']}-{arm}";observer=Observe(n,stem)
                cfg=GenerationConfig(do_sample=arm=='sample',temperature=1.,top_p=1.,top_k=0,min_p=None,typical_p=1.,epsilon_cutoff=0.,eta_cutoff=0.,
                    repetition_penalty=1.,encoder_repetition_penalty=1.,no_repeat_ngram_size=0,encoder_no_repeat_ngram_size=0,bad_words_ids=None,
                    force_words_ids=None,suppress_tokens=None,begin_suppress_tokens=None,forced_bos_token_id=None,forced_eos_token_id=None,
                    sequence_bias=None,guidance_scale=None,renormalize_logits=False,remove_invalid_values=False,
                    max_new_tokens=1600,num_beams=1,num_beam_groups=1,num_return_sequences=1,use_cache=True,
                    eos_token_id=sorted(eos),pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=False)
                atomic(out/(stem+'.generation-config.json'),cfg.to_dict())
                class Stop(StoppingCriteria):
                    def __init__(self):self.wall=False;self.last=0
                    def __call__(self,input_ids,scores,**kwargs):
                        count=input_ids.shape[1]-n;self.wall=time.time()>=a.TERM_unix-10
                        if count-self.last>=32 or self.wall:
                            atomic(out/(stem+'.partial.json'),dict(generated_ids=input_ids[0,n:].tolist(),selected_logps=observer.logps,wall_stop=self.wall));self.last=count
                        return self.wall
                stopper=Stop()
                with torch.no_grad(),amp():
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=cfg,
                        logits_processor=LogitsProcessorList([observer]),stopping_criteria=StoppingCriteriaList([stopper]))
                    observer.finish_previous(generated.sequences)
                torch.cuda.synchronize();full=generated.sequences[0].tolist();assert full[:n]==prefix;suffix=full[n:]
                assert suffix==observer.ids and len(suffix)==len(observer.logps)<=1600 and suffix
                ended=suffix[-1] in eos;cap=len(suffix)>=1600 and not ended
                item=dict(row_id=row['row_id'],index=row['index'],arm=arm,seed=seed,input_tokens=n,input_ids_sha256=ids_sha(prefix),generated_ids=suffix,
                    cached_selected_logps=observer.logps,EOS=ended,cap=cap,wall_stop=stopper.wall,seconds=time.monotonic()-tick,
                    processor_classes=processor_audits[-1],raw_before_parse=True)
                atomic(out/(stem+'.ids.json'),item)
                textids=suffix[:-1] if ended else suffix
                response=ntok.decode(textids,skip_special_tokens=False);hf=tok.decode(textids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                item['response']=response;item['native_HF_decode_equal']=response==hf
                atomic(out/(stem+'.raw.json'),item);records.append(item);r['rows'].append({k:item[k] for k in ['row_id','arm','EOS','cap','seconds']})
                atomic(out/'progress.json',dict(r,phase='generation',completed=len(records)));assert response==hf
                del generated,ids,observer
                if stopper.wall:raise TimeoutError('wall-stopped raw retained, no reroll')
        assert len(records)==8
        rawref=atomic(out/'responses.json',records)
        atomic(out/'reward-request.json',dict(status='ALL8_RAW_FROZEN_WAITING_SAME_PARENT',responses=rawref,execution=ref(ep),TERM_unix=a.TERM_unix,KILL_unix=a.KILL_unix,optimizer_steps=0))
        rewardpath=ep.parent/'rewards.json'
        while not rewardpath.exists():budget();time.sleep(1)
        rewards=json.loads(rewardpath.read_text());r['rewards']=ref(rewardpath)
        # Controller binds its fully frozen reward product to this exact raw file.
        delivery=json.loads((ep.parent/'reward-delivery.json').read_text())
        assert delivery['responses_sha256']==rawref['sha256'] and delivery['rewards_sha256']==sha(rewardpath)
        assert delivery['normalizer_manifest_sha256']==plan['normalizer_manifest_sha256']
        assert rewards['original_raw_responses']['sha256']==rawref['sha256']
        pairs={x['row_id']:x for x in rewards['pairs']};assert set(pairs)=={x['row_id'] for x in rows}
        assert all(math.isfinite(x['advantage']) and x['advantage']==x['sample_reward']-x['greedy_reward'] for x in pairs.values())
        nonzero=sum(x['advantage']!=0 for x in pairs.values());semantic=sum(x['semantic_difference'] for x in pairs.values())
        gate=nonzero>=2 and semantic>=1;assert gate==rewards['supply_gate_pass'];r['supply_gate']=dict(pass_gate=gate,nonzero_groups=nonzero,nonempty_gold_valid_semantic_groups=semantic)
        if not gate:
            r['status']='NEGATIVE_FIXED_SUPPLY_GATE_NO_GRADIENT_OR_UPDATE';return
        budget();model.train();model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        assert model.training and model.is_gradient_checkpointing
        assert all(m.p==0 for m in model.modules() if isinstance(m,torch.nn.Dropout))
        checkpoint_calls={'grad_enabled':0,'no_grad':0};checkpoint_modules=[]
        for name,module in model.named_modules():
            if getattr(module,'gradient_checkpointing',False) and hasattr(module,'_gradient_checkpointing_func'):
                fn=module._gradient_checkpointing_func;checkpoint_modules.append(name)
                def counted(*args,_fn=fn,**kwargs):
                    checkpoint_calls['grad_enabled' if torch.is_grad_enabled() else 'no_grad']+=1;return _fn(*args,**kwargs)
                module._gradient_checkpointing_func=counted
        assert checkpoint_modules
        optimizer=torch.optim.AdamW(list(params.values()),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        assert not optimizer.state;optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().cpu().clone() for n,p in params.items() if 'lora_B' in n}
        samples={x['row_id']:x for x in records if x['arm']=='sample'};observations=[]
        for row in rows:
            budget();sample=samples[row['row_id']];n=row['prefix_tokens'];suffix=sample['generated_ids'];full=row['input_ids'][:n]+suffix
            inp=torch.tensor([full],device='cuda',dtype=torch.long);labels=inp.clone();labels[:,:n]=-100
            with torch.no_grad(),amp():ng=token_logps(model,inp,labels).detach()
            cached=torch.tensor(sample['cached_selected_logps'],device='cuda',dtype=torch.float32)
            if not bool(torch.isfinite(ng).all() and torch.isfinite(cached).all()):
                atomic(out/f"row{row['index']}-nonfinite-replay.json",dict(row_id=row['row_id'],tokens=len(suffix),
                    cached_nonfinite=int((~torch.isfinite(cached)).sum()),full_nonfinite=int((~torch.isfinite(ng)).sum()),
                    cached_logps=[float(v) if math.isfinite(float(v)) else str(float(v)) for v in cached],
                    full_logps=[float(v) if math.isfinite(float(v)) else str(float(v)) for v in ng]))
                raise FloatingPointError('nonfinite replay retained; no mask or update')
            cached_diff=(cached-ng).abs();obs=dict(row_id=row['row_id'],advantage=pairs[row['row_id']]['advantage'],
                cached_token_logps=cached.tolist(),full_no_grad_token_logps=ng.tolist(),cached_full_max_abs=float(cached_diff.max()),
                cached_full_sum_abs=float(cached_diff.sum()),cached_sum=float(cached.sum()),full_sum=float(ng.sum()),
                cached_full_gate='Observed, not assigned arbitrary1e-5 equality; actual result informs formal usability')
            atomic(out/f"row{row['index']}-replay-before-grad.json",obs)
            with amp():lp=token_logps(model,inp,labels)
            same=replay_observation(ng,lp);obs['same_full_no_grad_grad']=same
            atomic(out/f"row{row['index']}-replay-numeric.json",obs)
            assert same['allclose_rtol_atol_1e5'],'Same complete input no-grad/grad replay mismatch; preserve and stop'
            coefficient=-float(pairs[row['row_id']]['advantage'])/(11884/87)/4
            policy_loss=coefficient*lp.sum();assert bool(torch.isfinite(policy_loss));policy_loss.backward();r['real_backward_calls']+=1
            obs['policy_loss']=float(policy_loss.detach());del lp,policy_loss,inp,labels,ng,cached
            budget();gold=torch.tensor([row['input_ids']],device='cuda',dtype=torch.long);goldlabels=torch.tensor([row['labels']],device='cuda',dtype=torch.long)
            with amp():goldlp=token_logps(model,gold,goldlabels);ce=-goldlp.mean()/4
            assert bool(torch.isfinite(ce));ce.backward();r['real_backward_calls']+=1;obs['CE_quarter']=float(ce.detach());del ce,goldlp,gold,goldlabels
            observations.append(obs);atomic(out/'loss-progress.json',dict(observations=observations,real_backward_calls=r['real_backward_calls'],checkpoint_calls=checkpoint_calls))
        assert r['real_backward_calls']==8 and checkpoint_calls['grad_enabled']>0
        gradient={}
        for n,p in params.items():
            finite=p.grad is not None and bool(torch.isfinite(p.grad).all());norm=p.grad.float().norm() if p.grad is not None else None
            gradient[n]=dict(finite=finite,norm=float(norm) if norm is not None and bool(torch.isfinite(norm)) else None,
                             nonfinite_count=int((~torch.isfinite(p.grad)).sum()) if p.grad is not None else None)
        atomic(out/'gradient.json',dict(parameters=gradient,checkpoint_modules=checkpoint_modules,checkpoint_calls=checkpoint_calls))
        assert all(x['finite'] for x in gradient.values()) and all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        atomic(out/'clip-observation.json',dict(finite=bool(torch.isfinite(norm)),norm=float(norm) if bool(torch.isfinite(norm)) else None))
        assert bool(torch.isfinite(norm));budget();optimizer.step();r['optimizer_steps']=1
        delta={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in params.items() if n in before}
        assert all(bool(torch.isfinite(p).all()) for p in params.values())
        for state in optimizer.state.values():
            assert all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in state.values())
        families={family(n) for n,d in delta.items() if d>0};assert families=={'DeltaNet','full_attention','MLP'}
        atomic(out/'optimizer.json',dict(steps=1,gradient_norm=float(norm),changed_B=delta,changed_families=sorted(families),state_parameters=len(optimizer.state),all_finite=True))
        budget();checkpoint=out/'probe-state';checkpoint.mkdir();model.save_pretrained(checkpoint/'adapter',safe_serialization=True)
        torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),steps=1,discarded_probe=True),checkpoint/'training_state.pt')
        files={}
        for p in checkpoint.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
                files[str(p.relative_to(checkpoint))]=ref(p)
        for d in sorted([p for p in checkpoint.rglob('*') if p.is_dir()]+[checkpoint,out],key=lambda p:len(p.parts),reverse=True):
            fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
        assert all(ref(x['path'])==x for x in files.values())
        atomic(out/'probe-state-ACK.json',dict(files=files,all_reopened_SHA=True,file_and_directory_fsync=True,discarded_probe=True,formal_authorized=False))
        r.update(status='PASS_FIXED_SUPPLY_AND_ONE_DISCARDED_SCST_ENGINEERING_UPDATE',checkpoint_ACK=ref(out/'probe-state-ACK.json'),cached_full_difference_observations=observations,
                 same_full_no_grad_grad_replay_all_pass=True,checkpoint_calls=checkpoint_calls,formal_usability_requires_observed_cached_full_review=True)
    except BaseException as e:
        r.update(status='FAILED_ENGINEERING_OR_RESOURCE_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        r.update(finished_at=now(),program_seconds=time.monotonic()-start,process_pid=os.getpid())
        if torch is not None and torch.cuda.is_initialized():r['peak_memory']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        atomic(out/'result.json',r);print(json.dumps({k:r.get(k) for k in ['status','program_seconds','error','supply_gate','optimizer_steps']}),flush=True)
if __name__=='__main__':main()
