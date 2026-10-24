"""One fresh native10epoch Task1 run, with no-skip guards and complete state saves."""
import argparse
from collections import Counter
import dataclasses
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from probe import observe_batch, sha, write
from metadata import native_summary


def main():
    ap=argparse.ArgumentParser()
    for name in ('model','train','audit','plan','output'):ap.add_argument('--'+name,required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();state={'phase':'imports','steps':0,'epochs':[]};write(out/'progress.json',state)
    try:
        import numpy as np
        import torch,transformers,gliner2
        from tokenizers import Tokenizer
        from safetensors import safe_open
        from gliner2.joint_ie import JointIE
        from gliner2.training.trainer import ExtractorTrainer,TrainingConfig
        import gliner2.models.boundary.model as boundary
        torch.set_num_threads(4);assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        plan=json.loads(Path(args.plan).read_text());assert sha(args.train)==plan['data']['train_native_sha256'];assert sha(args.audit)==plan['data']['target_audit_sha256']
        for a in plan['assets']:
            p=Path(args.model)/a['name'];assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
        rows=[json.loads(x) for x in Path(args.train).read_text().splitlines()];audits=[json.loads(x) for x in Path(args.audit).read_text().splitlines()];assert len(rows)==len(audits)==320
        lookup={}
        for i,r in enumerate(rows):
            for text in [r['input'],r['input']+'.']:
                assert text not in lookup or lookup[text]==i;lookup[text]=i
        state['phase']='fresh_public_load';write(out/'progress.json',state)
        engine=JointIE.from_pretrained(args.model,device='cpu',dtype='float32',local_files_only=True);model=engine.model
        assert sum(p.numel() for p in model.parameters())==193581591 and len(model.state_dict())==334
        assert model.processor.sampling_config.synthetic_entity_label_prob==.2
        assert model.processor.sampling_config.remove_entity_prob==model.processor.sampling_config.remove_entities_prob==0
        write(out/'inventory.json',{'parameters':193581591,'tensors':334,'all_bundle_parameters_counted':True,'strict_fresh_public_load':True,'weight_sha256':sha(Path(args.model)/'model.safetensors'),'torch':torch.__version__,'transformers':transformers.__version__,'transformers_file':transformers.__file__,'gliner2_file':gliner2.__file__,'sampling_config':dataclasses.asdict(model.processor.sampling_config),'source_sha256':sha(__file__),'shared_observer_sha256':sha(Path(__file__).with_name('probe.py')),'plan_sha256':sha(args.plan)})
        native=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'));cache={};ctx={};variants={};seen={i:[] for i in range(10)};target_counts=Counter();epoch_losses=Counter();raw_loss_counts=Counter();state_saves=[]
        (out/'variants').mkdir();native_finite=boundary._finite_loss_term
        @functools.wraps(native_finite)
        def finite_guard(value):
            finite=bool(torch.isfinite(value.detach()).all().item());raw_loss_counts[ctx.get('epoch',-1)]+=1
            if not finite:
                write(out/'nonfinite-raw-loss.json',{'epoch':ctx.get('epoch'),'step':ctx.get('step'),'finite':False,'source':'before native nan_to_num'});raise FloatingPointError('Nonfinite raw native loss')
            return native_finite(value)
        boundary._finite_loss_term=finite_guard
        class FullTrainer(ExtractorTrainer):
            def _prepare_data(self,data,is_train=True):
                expected=Counter(json.dumps(r,sort_keys=True) for r in rows)
                ds=super()._prepare_data(data,is_train=is_train)
                actual=Counter(json.dumps(r,sort_keys=True) for r in ds.data)
                write(out/'loader.json',{'records':len(ds),'canonical_multiset_unchanged':actual==expected});assert len(ds)==320 and actual==expected
                return ds
            def _create_dataloader(self,dataset,batch_size,shuffle=True,is_training=True):
                loader=super()._create_dataloader(dataset,batch_size,shuffle,is_training);assert len(loader)==160 and loader.batch_size==2 and loader.drop_last
                collate=loader.collate_fn
                def tracked(raw):
                    before=random.getstate();b=collate(raw);after=random.getstate();ctx['rng_hashes']=[hashlib.sha256(repr(x).encode()).hexdigest() for x in [before,after]];return b
                loader.collate_fn=tracked;return loader
            def _record_failed_batch(self,batch,*,data_loader_step,error):
                super()._record_failed_batch(batch,data_loader_step=data_loader_step,error=error);raise error
            def _evaluate(self,*a,**kw):raise AssertionError('No eval/holdout permitted')
            def _backward_one(self,batch,step,use_amp,amp_dtype,*,is_last_micro=True):
                t=time.monotonic();ctx.update(epoch=self.epoch,step=self.global_step)
                raw=observe_batch(batch,lookup,audits,self.processor,native,cache);refs=[]
                for item in raw:
                    payload=json.dumps(item,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();h=hashlib.sha256(payload).hexdigest();name=f'{item["row_id"]:03d}-{h[:16]}.json'
                    if h not in variants:write(out/'variants'/name,item);variants[h]=name
                    refs.append(name)
                # Persist current identities before any assertion or neural operation.
                write(out/'active-batch.json',{'epoch':self.epoch+1,'step':self.global_step,'batch':step,'variants':refs,'rng_before_after_sha256':ctx['rng_hashes']})
                for x in raw:
                    assert x['actual_input_ids']==x['independent_tokenizer_json_ids'];assert sorted(x['actual_positive_pairs'])==sorted(x['expected_positive_pairs']);assert x['query_mask']==[True] and x['input_ids']<=4096
                    assert sum(sum(mask) for mask in x['mention_mask'])==len(x['actual_positive_pairs'])
                loss=super()._backward_one(batch,step,use_amp,amp_dtype,is_last_micro=is_last_micro)
                grads=[p.grad.detach() for p in self.model.parameters() if p.grad is not None];finite_grad=all(bool(torch.isfinite(g).all().item()) for g in grads);norm=float(torch.stack([g.float().norm() for g in grads]).norm().item());scalar=float(loss.item())
                obs={'epoch':self.epoch+1,'step':self.global_step,'batch':step,'row_ids':[x['row_id'] for x in raw],'variants':refs,'input_lengths':[x['input_ids'] for x in raw],'gold':[len(x['actual_positive_pairs']) for x in raw],'loss':scalar if math.isfinite(scalar) else None,'finite_loss':math.isfinite(scalar),'finite_gradients':finite_grad,'gradient_norm':norm if math.isfinite(norm) else None,'fb_seconds':time.monotonic()-t,'raw_loss_terms_seen':raw_loss_counts[self.epoch],'rng_before_after_sha256':ctx['rng_hashes']}
                write(out/'active-step.json',obs)
                assert obs['finite_loss'] and finite_grad and math.isfinite(norm)
                seen[self.epoch].extend(obs['row_ids']);target_counts[self.epoch]+=sum(obs['gold']);epoch_losses[self.epoch]+=scalar;ctx['obs']=obs
                return loss
            def _optimizer_step(self):
                applied=super()._optimizer_step();obs=ctx['obs'];first_param=next(p for p in self.model.parameters() if p in self.optimizer.state)
                observed_step=int(self.optimizer.state[first_param]['step'].item());assert applied and observed_step==self.global_step+1
                obs.update(applied=True,completed_step=observed_step,optimizer_parameter_states=len(self.optimizer.state),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(obs,separators=(',',':'),allow_nan=False)+'\n');f.flush()
                state.update(phase='training',steps=observed_step,epoch=self.epoch+1,elapsed_seconds=time.monotonic()-started)
                if observed_step%160==0:
                    assert sorted(seen[self.epoch])==list(range(320)) and target_counts[self.epoch]==2642
                    e={'epoch':self.epoch+1,'steps':160,'samples':320,'targets':target_counts[self.epoch],'average_loss':epoch_losses[self.epoch]/160,'row_order_sha256':hashlib.sha256(json.dumps(seen[self.epoch]).encode()).hexdigest(),'all_rows_exactly_once':True};state['epochs'].append(e)
                    write(out/f'epoch-{self.epoch+1:02d}.json',e)
                    if observed_step==800:self.save_full_state('checkpoint-epoch5',800,5)
                if observed_step%10==0:write(out/'progress.json',state)
                return applied
            def _save_checkpoint(self,name):
                assert name=='final' and self.global_step==1600 and self.epoch==9
                self.save_full_state('final',1600,10)
            def save_full_state(self,name,completed_steps,completed_epochs):
                dst=self.output_dir/name;dst.mkdir(parents=True,exist_ok=False);self.model.save_pretrained(str(dst))
                # Native saver persists model/tokenizer only; explicitly save optimizer/schedule/RNG.
                payload={'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all(),'completed_steps':completed_steps,'completed_epochs':completed_epochs,'next_sampler_epoch':completed_epochs,'sampler_seed':self.config.seed,'planned_steps':1600,'source_sha256':sha(__file__)}
                p=dst/'training-state.pt';torch.save(payload,p)
                for saved in dst.rglob('*'):
                    if saved.is_file():
                        with saved.open('rb') as f:os.fsync(f.fileno())
                restored=torch.load(p,map_location='cpu',weights_only=False);assert restored['completed_steps']==completed_steps and restored['scheduler']==payload['scheduler'] and restored['scaler']==payload['scaler'] and restored['python_rng']==payload['python_rng']
                assert restored['numpy_rng'][0]==payload['numpy_rng'][0] and np.array_equal(restored['numpy_rng'][1],payload['numpy_rng'][1]) and restored['numpy_rng'][2:]==payload['numpy_rng'][2:]
                assert torch.equal(restored['torch_cpu_rng'],payload['torch_cpu_rng']) and len(restored['torch_cuda_rng'])==len(payload['torch_cuda_rng']) and all(torch.equal(a,b) for a,b in zip(restored['torch_cuda_rng'],payload['torch_cuda_rng']))
                assert restored['optimizer']['param_groups']==payload['optimizer']['param_groups']
                for key,values in payload['optimizer']['state'].items():
                    for field,value in values.items():
                        actual=restored['optimizer']['state'][key][field]
                        assert torch.equal(actual,value.detach().cpu()) if isinstance(value,torch.Tensor) else actual==value
                with safe_open(str(dst/'model.safetensors'),framework='pt',device='cpu') as f:
                    expected=self.model.state_dict();assert set(f.keys())==set(expected)
                    for n in f.keys():assert torch.equal(f.get_tensor(n),expected[n].detach().cpu())
                report={'name':name,'completed_epochs':completed_epochs,'completed_steps':completed_steps,'model_tensors':334,'optimizer_parameter_states':len(restored['optimizer']['state']),'readback_model_optimizer_scheduler_scaler_RNG_exact':True,'automatic_resume_implemented_or_tested':False,'resume_scope':'Complete end-of-epoch state retained. Native load_checkpoint alone only restores weights; full resume requires explicit restored states and sampler epoch.','files':{str(x.relative_to(dst)):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in dst.rglob('*') if x.is_file()}}
                write(dst/'state-manifest.json',report);state_saves.append(report);del restored,payload
        config=TrainingConfig(output_dir=str(out/'native'),num_epochs=10,batch_size=2,gradient_accumulation_steps=1,encoder_lr=1e-5,task_lr=5e-4,weight_decay=.01,max_grad_norm=1.,fp16=False,bf16=True,seed=20261024,num_workers=0,eval_strategy='no',early_stopping=False,save_best=False,max_train_samples=-1,max_steps=-1,strict_training=True,allow_invalid_samples=False,skip_step_errors=False,ignore_nonfinite_losses=False,dedupe_contained_entity_spans=False,on_capacity_exceeded='raise',logging_steps=50,report_to_wandb=False,log_proposal_metrics=True,diagnostics_every_n_steps=50,gold_injection_start=1.,gold_injection_end=.25,gold_injection_hold_frac=.15,warmup_ratio=.1,scheduler_type='linear')
        trainer=FullTrainer(model,config,train_data=rows,eval_data=None);torch.cuda.reset_peak_memory_stats();summary=native_summary(trainer.train(train_data=rows,eval_data=None))
        assert summary['total_steps']==1600 and summary['total_epochs']==10 and not summary['eval_metrics_history'];assert len(state['epochs'])==10
        report={'status':'FRESH_NATIVE_TRAIN32_10EPOCH_1600UPDATES_COMPLETE','summary':summary,'epochs':state['epochs'],'source_sha256':sha(__file__),'input_sha256':sha(args.train),'audit_sha256':sha(args.audit),'plan_sha256':sha(args.plan),'raw_loss_finite_checks':sum(raw_loss_counts.values()),'native_query_augmentation':.2,'parameters':193581591,'save_checkpoints':[x['name'] for x in state_saves],'final_state':state_saves[-1],'actual_schema_variants':len(variants),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started,'quality_evaluations':0}
        write(out/'complete.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['summary','epochs','final_state']}),flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'phase':state.get('phase'),'completed_steps':state.get('steps'),'elapsed_seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
