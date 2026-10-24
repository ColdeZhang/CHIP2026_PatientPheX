"""Prepared synthetic diagnostic. Refuses GPU use without an explicit launch flag/UUID."""
import argparse, gc, inspect, json, os, pathlib, random, statistics, subprocess, time
import qwen35_smoke as H
import qwen35_relation_cost as C
from cpu_contract import graph


def tensor_metrics(a,b,torch,thresholds,limit,logits=False):
    a=a.detach().float().cpu();b=b.detach().float().cpu()
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    d=b-a;norm=float(torch.linalg.vector_norm(a.double()));err=float(torch.linalg.vector_norm(d.double()))
    out={'reference_l2':norm,'error_l2':err,'relative_l2':err/norm if norm else None,'max_abs':float(d.abs().max()),'shape':list(a.shape)}
    passed=out['max_abs']<=thresholds['near_zero_max_abs'] if norm<=thresholds['near_zero_reference_l2'] else err/norm<=limit
    if logits:
        log_a=a.log_softmax(-1);log_b=b.log_softmax(-1)
        kl=float((log_a.exp()*(log_a-log_b)).sum(-1).mean());out['mean_kl']=kl
        passed=passed and out['max_abs']<=thresholds['logits_max_abs'] and kl<=thresholds['logits_mean_kl']
    out['passed']=bool(passed);return out


def compare_states(a,b,torch,th):
    assert set(a)==set(b)
    assert all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape for k in a)
    out={k:tensor_metrics(a[k],b[k],torch,th,th['cache_relative_l2']) for k in a}
    assert all(x['passed'] for x in out.values()),'cache numerical threshold'
    return out


def states(cache,torch):
    out={}
    for i,layer in enumerate(cache.layers):
        for kind in ['conv_states','recurrent_states']:
            values=getattr(layer,kind,None)
            if values is None:continue
            if torch.is_tensor(values):values=[values]
            for j,t in enumerate(values):
                if t is not None:out[f'{i}/{kind}/{j}']=t.detach().cpu().clone()
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default='probe-plan.json');ap.add_argument('--prompts',default='cpu-native-v1/synthetic-prompts.json');ap.add_argument('--output',required=True);ap.add_argument('--gpu-uuid');ap.add_argument('--run-authorized',action='store_true');a=ap.parse_args()
    if not a.run_authorized or not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:
        raise SystemExit('No GPU launch: require explicit root grant, --run-authorized and exact allocated UUID')
    manifest=json.loads(pathlib.Path('probe-source-manifest.json').read_text())
    assert all(H.sha(f)==v['sha256'] for f,v in manifest['files'].items()),'frozen source/plan changed'
    plan=json.loads(pathlib.Path(a.plan).read_text());th=plan['thresholds'];out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert H.sha('qwen35_smoke.py')=='9a15b1bbed46926ae8e6530c9f7704774088bcd5c6fee25cade36a9e9ada93c3'
    assert H.sha('qwen35_relation_cost.py')=='a2df013eff13c81c86ead1bab6e6473c7ce139eb9aa97d5a5ae797c95692106c'
    gate=json.loads(pathlib.Path('gpu-import-gate.json').read_text());assert gate['passed']
    cpu=json.loads(pathlib.Path('cpu-native-v1/cpu-contract.json').read_text());assert cpu['passed'];assert H.sha(a.prompts)==cpu['synthetic_prompts_sha256']
    query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    H.write(out/'gpu-preflight.json',{'uuid':a.gpu_uuid,'processes':query})
    assert not any(x.split(',')[0].strip()==a.gpu_uuid for x in query.splitlines()),'allocated GPU occupied'
    import torch,peft
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from transformers.cache_utils import DynamicCache
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    torch.manual_seed(plan['seed']);random.seed(plan['seed']);torch.backends.cuda.matmul.allow_tf32=False
    names=['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'];fused={n:getattr(q,n) for n in names};native={n:inspect.unwrap(fused[n]) for n in names}
    bindings={n:graph(fused[n]) for n in names}
    assert all(any(t['module'].startswith('fla.ops.gated_delta_rule') for t in bindings[n]) for n in names)
    assert all(native[n].__module__==q.__name__ for n in names)
    H.write(out/'actual-bindings.json',{'fused':bindings,'native':{n:graph(native[n]) for n in names},'conv':{n:graph(getattr(q,n)) for n in ['causal_conv1d_fn','causal_conv1d_update']}})
    def select(arm):
        for n,v in (native if arm=='native' else fused).items():setattr(q,n,v)
    started=time.monotonic();stage_start=started;stage_limit=0;timings={};results={}
    def stage(name):
        nonlocal stage_start,stage_limit
        stage_start=time.monotonic();stage_limit=next(x['budget_seconds'] for x in plan['stages'] if x['name']==name)
        H.write(out/'status.json',{'stage':name,'elapsed_seconds':stage_start-started,'completed':False})
    def check_time():
        assert time.monotonic()-started<plan['gpu_total_hard_seconds'],'total deadline'
        assert time.monotonic()-stage_start<stage_limit,'stage deadline'
    def timed(label,fn):
        check_time();torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();seconds=time.perf_counter()-t
        timings.setdefault(label,[]).append(seconds);H.write(out/'timings.json',timings);check_time();return v
    def assert_metric(label,x,y,limit,logits=False):
        m=tensor_metrics(x,y,torch,th,limit,logits);results[label]=m;H.write(out/'metrics.json',results);assert m['passed'],label;return m
    cfg=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True)
    stage('single_gdn');block=q.Qwen3_5GatedDeltaNet(cfg,layer_idx=0).to(device='cuda',dtype=torch.bfloat16)
    hidden=torch.randn(1,256,cfg.hidden_size,device='cuda',dtype=torch.bfloat16);continuation=torch.randn(1,8,cfg.hidden_size,device='cuda',dtype=torch.bfloat16)
    block_results={}
    for arm in ['native','fused']:
        select(arm);block.zero_grad(set_to_none=True)
        def fb():
            y=block(hidden);loss=y.float().square().mean();loss.backward();return y.detach().cpu(),float(loss.detach())
        y,loss=timed('block_'+arm+'_cold_fb',fb);grads={n:p.grad.detach().cpu().clone() for n,p in block.named_parameters() if p.grad is not None}
        with torch.no_grad():
            cache=DynamicCache(config=cfg);prefill=block(hidden,cache_params=cache);snapshots=[states(cache,torch)];tokens=[]
            for i in range(8):tokens.append(block(continuation[:,i:i+1],cache_params=cache).cpu());snapshots.append(states(cache,torch))
            for rep in range(plan['warm_repeats']):timed('block_'+arm+'_warm_forward',lambda:block(hidden))
        block_results[arm]={'y':y,'loss':loss,'grads':grads,'decode':torch.cat(tokens,dim=1),'states':snapshots}
    b0,b1=block_results['native'],block_results['fused'];assert_metric('block_output',b0['y'],b1['y'],th['logits_relative_l2']);assert abs(b0['loss']-b1['loss'])<=th['loss_abs'];assert_metric('block_decode',b0['decode'],b1['decode'],th['logits_relative_l2'])
    for n in b0['grads']:assert_metric('block_grad/'+n,b0['grads'][n],b1['grads'][n],th['gradient_relative_l2'])
    for i in range(9):results[f'block_cache/{i}']=compare_states(b0['states'][i],b1['states'][i],torch,th)
    H.write(out/'metrics.json',results);del block,hidden,continuation,block_results,b0,b1;gc.collect();torch.cuda.empty_cache()
    stage('load_text_only_9b');select('native')
    model,loading=timed('load_model',lambda:Qwen3_5ForCausalLM.from_pretrained(plan['model'],config=cfg,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,trust_remote_code=False,output_loading_info=True))
    H.write(out/'loading-info.json',loading)
    assert not any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs'])
    assert all(n.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for n in loading.get('unexpected_keys',[]))
    assert sum(p.numel() for p in model.parameters())==C.BASE_PARAMETERS
    assert model.__class__.__name__=='Qwen3_5ForCausalLM' and not any('visual' in n for n,_ in model.named_modules())
    model.eval();prompts=json.loads(pathlib.Path(a.prompts).read_text());assert [len(x['input_ids']) for x in prompts]==plan['prompt_lengths']
    def infer(row,arm,record):
        select(arm);ids=torch.tensor([row['input_ids']],device='cuda');extra=row['teacher_forced_ids'];snapshots=[]
        with torch.no_grad():
            r=timed(f"prefill/{len(row['input_ids'])}/{arm}",lambda:model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True))
            logits=r.logits.detach().cpu().float() if record else None;cache=r.past_key_values
            if record:snapshots.append(states(cache,torch))
            forced=[]
            def decode():
                nonlocal cache
                for t in extra:
                    o=model(input_ids=torch.tensor([[t]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1);cache=o.past_key_values
                    if record:forced.append(o.logits.detach().cpu().float());snapshots.append(states(cache,torch))
            timed(f"decode/{len(row['input_ids'])}/{arm}",decode)
            del cache,r
            greedy=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),min_new_tokens=8,max_new_tokens=8,do_sample=False,use_cache=True)[:,ids.shape[1]:].cpu().tolist() if record else None
        return {'logits':logits,'forced':torch.cat(forced,dim=1) if record else None,'states':snapshots,'greedy':greedy}
    stage('four_synthetic_inference')
    for row in prompts:
        nr=infer(row,'native',True);fr=infer(row,'fused',True);key=row['id']
        assert_metric(key+'/logits',nr['logits'],fr['logits'],th['logits_relative_l2'],True)
        assert_metric(key+'/forced',nr['forced'],fr['forced'],th['logits_relative_l2'],True)
        target=torch.tensor(row['input_ids'][1:]);losses=[float(torch.nn.functional.cross_entropy(x['logits'][0,:-1],target)) for x in [nr,fr]]
        results[key+'/loss']={'native':losses[0],'fused':losses[1],'abs_difference':abs(losses[0]-losses[1])};assert abs(losses[0]-losses[1])<=th['loss_abs']
        results[key+'/greedy']={'native':nr['greedy'],'fused':fr['greedy'],'equal':nr['greedy']==fr['greedy']};assert nr['greedy']==fr['greedy']
        for i in range(9):results[f'{key}/cache/{i}']=compare_states(nr['states'][i],fr['states'][i],torch,th)
        H.write(out/'metrics.json',results);del nr,fr
        for rep in range(plan['warm_repeats']):
            for arm in (['native','fused'] if rep%2==0 else ['fused','native']):infer(row,arm,False)
    stage('short_lora_step');model.config.use_cache=False;model,trainable,coverage=C.attach_adapter(model,torch,peft)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();assert model.is_gradient_checkpointing
    H.write(out/'lora-coverage.json',dict(coverage,dropout_override=0.0,gradient_checkpointing=True,use_reentrant=False))
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()};ids=torch.tensor([prompts[0]['input_ids']],device='cuda');labels=ids.clone();labels[:,:64]=-100
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels,'use_cache':False};train_results={}
    for arm in ['native','fused']:
        select(arm)
        with torch.no_grad():
            for n,p in trainable.items():p.copy_(initial[n])
        optimizer=torch.optim.AdamW(list(trainable.values()),lr=1e-4,weight_decay=.01);model.train();optimizer.zero_grad(set_to_none=True)
        def step():
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            loss.backward();grads={n:p.grad.detach().float().cpu().clone() for n,p in trainable.items() if p.grad is not None}
            assert len(grads)==len(trainable) and all(torch.isfinite(g).all() for g in grads.values())
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0);optimizer.step();delta={n:p.detach().float().cpu()-initial[n].float() for n,p in trainable.items()}
            for family in ['DeltaNet','full_attention','MLP']:assert any(float(d.norm())>0 for n,d in delta.items() if '.lora_B.' in n and C.family(n)==family)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):post=float(model(**batch).loss)
            return {'loss':float(loss.detach()),'post_loss':post,'grad':grads,'delta':delta}
        train_results[arm]=timed('lora_step/'+arm,step);del optimizer;model.zero_grad(set_to_none=True)
    nr,fr=train_results['native'],train_results['fused'];assert abs(nr['loss']-fr['loss'])<=th['loss_abs'];assert abs(nr['post_loss']-fr['post_loss'])<=th['loss_abs']
    for kind,limit in [('grad',th['gradient_relative_l2']),('delta',th['parameter_delta_relative_l2'])]:
        for n in nr[kind]:assert_metric('lora/'+kind+'/'+n,nr[kind][n],fr[kind][n],limit)
        assert_metric('lora/'+kind+'/aggregate',torch.cat([t.reshape(-1) for t in nr[kind].values()]),torch.cat([t.reshape(-1) for t in fr[kind].values()]),limit)
    summary={'numerical_passed':True,'elapsed_seconds':time.monotonic()-started,'plan_sha256':H.sha(a.plan),'source_sha256':H.sha(__file__),'speed':{},'training_loss':{arm:{k:r[k] for k in ['loss','post_loss']} for arm,r in train_results.items()},'quality_evaluation':False}
    for n in plan['prompt_lengths']:
        for phase in ['prefill','decode']:
            t0=statistics.median(timings[f'{phase}/{n}/native'][-3:]);t1=statistics.median(timings[f'{phase}/{n}/fused'][-3:]);summary['speed'][f'{phase}/{n}']={'native_median_seconds':t0,'fused_median_seconds':t1,'speedup':t0/t1}
    summary['speed_expansion_passed']=all(v['speedup']>=th['prefill_speedup_for_expansion'] if k.startswith('prefill') else v['speedup']>=1/th['maximum_decode_or_step_slowdown'] for k,v in summary['speed'].items()) and timings['lora_step/fused'][0]<=th['maximum_decode_or_step_slowdown']*timings['lora_step/native'][0]
    summary['lora_timing_note']='one fixed optimizer step per arm only; timing is diagnostic, not three-repeat training throughput'
    H.write(out/'summary.json',summary);H.write(out/'status.json',{'completed':True,'summary':'summary.json'});select('native')
if __name__=='__main__':main()
