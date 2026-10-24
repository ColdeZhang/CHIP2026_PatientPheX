"""A only: new raw numerical evidence, no old-gate retry, training or quality."""
import argparse
from collections.abc import Mapping
from datetime import datetime,timezone
import gc
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import zipfile
from evidence import checked,fsync_chain,jsha,persist_then_observe,positions,ref,save,sha
from distribution_stats import compare


def binding_graph(fn,seen=None):
    seen=set() if seen is None else seen
    if id(fn) in seen or not inspect.isfunction(fn):return []
    seen.add(id(fn));out=[dict(name=fn.__qualname__,module=fn.__module__,file=inspect.getsourcefile(fn))]
    for cell in fn.__closure__ or ():
        try:value=cell.cell_contents
        except ValueError:continue
        if inspect.isfunction(value):out+=binding_graph(value,seen)
    if hasattr(fn,'__wrapped__'):out+=binding_graph(fn.__wrapped__,seen)
    return out


def cache_layout(cache,torch):
    """H43 structure policy; deliberately no claim of cache tensor equality."""
    layout=[];tensors=0
    def visit(value,path):
        nonlocal tensors
        if value is None:layout.append([path,'none']);return
        if torch.is_tensor(value):
            tensors+=1;layout.append([path,'tensor',list(value.shape),str(value.dtype)]);return
        if isinstance(value,Mapping):
            assert all(type(k) in (int,str) for k in value)
            keys=sorted(value,key=lambda k:(type(k).__name__,str(k)))
            layout.append([path,'mapping',[[type(k).__name__,k] for k in keys]])
            for key in keys:visit(value[key],path+'/'+type(key).__name__+':'+str(key))
            return
        if isinstance(value,(list,tuple)):
            layout.append([path,type(value).__name__,len(value)])
            for index,item in enumerate(value):visit(item,path+'/'+str(index))
            return
        raise TypeError('Unsupported cache value '+type(value).__name__+' at '+path)
    assert isinstance(cache.layers,(list,tuple))
    for index,layer in enumerate(cache.layers):
        layout.append(['layer',index,type(layer).__name__])
        for name in ('conv_states','recurrent_states','keys','values'):
            if hasattr(layer,name):visit(getattr(layer,name),str(index)+'/'+name)
            else:layout.append([str(index)+'/'+name,'absent'])
    assert tensors>0
    return dict(layout=layout,tensor_count=tensors,layout_sha256=jsha(layout),numerical_cache_equivalence_claimed=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--absolute-deadline-monotonic',type=float,required=True)
    parser.add_argument('--run-authorized',action='store_true');args=parser.parse_args()
    started=time.monotonic();plan=json.loads(args.plan.read_text());out=args.output.resolve()
    assert args.run_authorized and plan['execution_ready'] and plan['scope']=='A_ONLY_NO_AUTOMATIC_B_C'
    assert plan['budget_seconds']==600 and plan['TERM_grace_seconds']==30
    assert args.absolute_deadline_monotonic<=started+600
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid']
    out.mkdir(parents=True,exist_ok=False);fsync_chain(out,Path(plan['project_root']))
    result=dict(status='RUNNING_A_ONLY',plan=ref(args.plan),source=ref(__file__),started_utc=datetime.now(timezone.utc).isoformat(),
        quality_calls=0,gold_read=False,training_steps=0,raw_logit_old_failure_unchanged=True,stages={})
    save(out/'intent.json',result);native=None;q=None
    def deadline():
        if time.monotonic()>=args.absolute_deadline_monotonic-30:raise TimeoutError('A soft570 stop before further work; original absolute600 unchanged')
    def terminate(*_):raise TimeoutError('Owner TERM at absolute A600 deadline')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for name,expected in plan['sources'].items():
            actual=ref(Path(__file__).parent/name);assert all(actual[k]==expected[k] for k in ('bytes','sha256'))
        cpu=json.loads(checked(plan['CPU_result']).read_text());inputs=json.loads(checked(plan['fixed_inputs']).read_text())
        assert cpu['status']=='PASS_CPU_A_ONLY_INPUT_INDEX_AND_SAVE_ORDER'
        assert cpu['fixed_inputs']['sha256']==plan['fixed_inputs']['sha256']
        assert jsha(inputs['fulltext14390']['input_ids'])==plan['long_input_ids_sha256']
        assert inputs['fulltext14390']['positions']==positions(14390)==plan['long_positions']
        assert len(inputs['synthetic128']['input_ids'])==128 and len(inputs['teacher_forced_ids'])==8
        base_manifest=json.loads(checked(plan['model_source_manifest']).read_text());base_files={}
        for item in base_manifest['files']:
            deadline();path=Path(plan['model'])/item['name'];actual=ref(path)
            assert actual['sha256']==item['sha256'] and actual['bytes']==item['size'];base_files[item['name']]=actual
        for name,digest in plan['adapter_files'].items():assert sha(Path(plan['adapter'])/name)==digest
        assert sha(plan['FLA_wheel']['path'])==plan['FLA_wheel']['sha256']
        assert sha(plan['einops_wheel']['path'])==plan['einops_wheel']['sha256']
        import torch,transformers,peft,triton
        from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        from peft import PeftModel
        assert torch.__version__=='2.7.1+cu126' and transformers.__version__=='5.20.2'
        assert peft.__version__=='0.21.2' and triton.__version__=='3.3.1'
        assert metadata.version('fla-core')=='0.6.0' and metadata.version('einops')=='0.8.1'
        assert sha(q.__file__)==plan['runtime_sha256']['modeling']
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.manual_seed(plan['seed']);random.seed(plan['seed'])
        names=('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule')
        fused={n:getattr(q,n) for n in names};native={n:inspect.unwrap(fused[n]) for n in names}
        graph={n:binding_graph(fused[n]) for n in names}
        assert all(any(item['module'].startswith('fla.ops.gated_delta_rule') for item in graph[n]) for n in names)
        assert all(native[n].__module__==q.__name__ for n in names)
        for item in [item for branch in graph.values() for item in branch]:
            if item['file'] and item['file'].endswith('/integrations/hub_kernels.py'):
                assert sha(item['file'])=='a4875051ee4f23a723fe769e94b6ad3030284983a02dba03ecf25ec22d114510'
        conv={n:binding_graph(getattr(q,n)) for n in ('causal_conv1d_fn','causal_conv1d_update')}
        assert not any(item['module'].startswith('causal_conv1d') for g in conv.values() for item in g)
        # Exact .py bytes from the existing verified private wheels, not just versions.
        installed={}
        for package,wheel_record in [('fla-core',plan['FLA_wheel']),('einops',plan['einops_wheel'])]:
            dist=metadata.distribution(package);verified={}
            with zipfile.ZipFile(checked(wheel_record)) as wheel:
                for entry in wheel.namelist():
                    if entry.endswith('.py'):
                        path=Path(dist.locate_file(entry));assert path.read_bytes()==wheel.read(entry)
                        verified[entry]=ref(path)
            assert verified;installed[package]=verified
        save(out/'actual-bindings-and-assets.json',dict(base=base_files,adapter=plan['adapter_files'],
            fused=graph,native={n:binding_graph(native[n]) for n in names},conv=conv,installed_sources=installed,
            binding_source_files={item['file']:ref(item['file']) for g in [*graph.values(),*conv.values()] for item in g if item['file']},
            runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,triton=triton.__version__,
                matmul_TF32=False,cudnn_TF32=True),indexed_head_review_reused=plan['head_review']))
        def select(arm):
            for name,fn in (native if arm=='native' else fused).items():setattr(q,name,fn)
        select('native');deadline();load_start=time.monotonic()
        config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True)
        model,loading=Qwen3_5ForCausalLM.from_pretrained(plan['model'],config=config,dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        save(out/'base-loading.json',loading)
        assert not any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
        assert all(k.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for k in loading.get('unexpected_keys',[]))
        assert sum(p.numel() for p in model.parameters())==8953803264
        model.eval();result['load_seconds']=time.monotonic()-load_start
        budget=dict(used_bytes=0,limit_bytes=1073741824);save(out/'raw-budget.json',budget)
        def persist_tensor(path,tensor,meta):
            copied=tensor.detach().cpu().contiguous()
            assert budget['used_bytes']+copied.numel()*copied.element_size()+65536<=budget['limit_bytes']
            def observe(saved):
                restored=torch.load(saved,map_location='cpu',weights_only=True)
                assert restored.dtype==copied.dtype and restored.shape==copied.shape
                assert torch.equal(restored.view(torch.uint8),copied.view(torch.uint8))
                finite=bool(torch.isfinite(restored).all())
                return dict(finite=finite,shape=list(restored.shape),dtype=str(restored.dtype),
                    nonfinite_count=int((~torch.isfinite(restored)).sum()),statistics=None if not finite else 'computed_after_raw')
            value=persist_then_observe(path,lambda h:torch.save(copied,h),observe,budget,meta)
            save(out/'raw-budget-progress.json',budget,replace=(out/'raw-budget-progress.json').exists())
            return value[0]
        for group in ('synthetic128','fulltext14390'):
            deadline()
            if group=='fulltext14390':
                select('native');model=PeftModel.from_pretrained(model,plan['adapter'],is_trainable=False);model.eval()
                assert sum(p.numel() for p in model.parameters())==8997081600
                assert type(model.get_base_model()).__name__=='Qwen3_5ForCausalLM'
            group_dir=out/group;group_dir.mkdir();fsync_chain(group_dir,Path(plan['project_root']))
            item=inputs[group];ids=torch.tensor([item['input_ids']],device='cuda');mask=torch.ones_like(ids)
            selection=0 if group=='synthetic128' else torch.tensor(item['positions'],device='cuda',dtype=torch.long)
            assert ids.shape==(1,128 if group=='synthetic128' else 14390)
            weights={n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters()}
            rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state().clone()
            runs=[]
            for number,(label,arm) in enumerate([('N1','native'),('F1','fused'),('N2','native'),('F2','fused')]):
                deadline();select(arm);torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
                assert weights=={n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters()}
                run_dir=group_dir/label;run_dir.mkdir();fsync_chain(run_dir,Path(plan['project_root']))
                identity=dict(group=group,label=label,arm=arm,sequence=number,input_ids_sha256=jsha(item['input_ids']),
                    shape=list(ids.shape),head_positions=item['positions'],head_selection='full128' if group=='synthetic128' else 'fixed32_before_head',
                    RNG_CPU_sha256=__import__('hashlib').sha256(rng_cpu.numpy().tobytes()).hexdigest(),
                    RNG_CUDA_sha256=__import__('hashlib').sha256(rng_cuda.cpu().numpy().tobytes()).hexdigest(),
                    weights_inventory_sha256=jsha({n:list(v) for n,v in weights.items()}),cold_or_warm='first_call_this_shape' if number<2 else 'one_repeat_not_steady_benchmark')
                save(run_dir/'identity.json',identity)
                torch.cuda.synchronize();t=time.monotonic()
                with torch.inference_mode():response=model(input_ids=ids,attention_mask=mask,use_cache=True,logits_to_keep=selection)
                torch.cuda.synchronize();seconds=time.monotonic()-t
                logits=response.logits;cache=response.past_key_values
                logits_ref=persist_tensor(run_dir/'prefill.pt',logits,identity)
                assert list(logits.shape)==[1,len(item['positions']),248320]
                layout=cache_layout(cache,torch);save(run_dir/'cache-prefill-layout.json',layout)
                save(run_dir/'prefill-time.json',dict(seconds=seconds,includes_JIT_if_any=True,evidence_IO_included=False))
                del response,logits
                decode_refs=[];decode_seconds=[]
                if group=='fulltext14390':
                    for step,token in enumerate(inputs['teacher_forced_ids']):
                        deadline();torch.cuda.synchronize();t=time.monotonic()
                        with torch.inference_mode():response=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
                        torch.cuda.synchronize();decode_seconds.append(time.monotonic()-t);cache=response.past_key_values
                        dref=persist_tensor(run_dir/('decode-%02d.pt'%step),response.logits,dict(identity,teacher_step=step,teacher_token_id=token))
                        assert list(response.logits.shape)==[1,1,248320];decode_refs.append(dref);del response
                        save(run_dir/('cache-decode-%02d-layout.json'%step),cache_layout(cache,torch))
                del cache
                assert weights=={n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in model.named_parameters()}
                saved=dict(identity=ref(run_dir/'identity.json'),prefill=logits_ref,decode=decode_refs,
                    prefill_seconds=seconds,decode_seconds=decode_seconds,raw_bytes_total=budget['used_bytes'])
                save(run_dir/'complete.json',saved);runs.append(saved);deadline()
            result['stages'][group]=dict(runs=runs,input_shape=list(ids.shape),selected_positions=item['positions'])
            save(out/'progress.json',result,replace=(out/'progress.json').exists())
            for left_index,right_index,name in [(0,2,'native_self_repeat'),(1,3,'fused_self_repeat'),(0,1,'cross_first'),(2,3,'cross_repeat')]:
                deadline();left,right=runs[left_index],runs[right_index]
                a=torch.load(checked(left['prefill']),map_location='cpu',weights_only=True)
                b=torch.load(checked(right['prefill']),map_location='cpu',weights_only=True)
                metric=compare(a,b,torch,item['positions']);del a,b
                if group=='fulltext14390':
                    a=torch.cat([torch.load(checked(x),map_location='cpu',weights_only=True) for x in left['decode']],dim=1)
                    b=torch.cat([torch.load(checked(x),map_location='cpu',weights_only=True) for x in right['decode']],dim=1)
                    metric['fixed_history_decode']=compare(a,b,torch,list(range(8)));del a,b
                save(group_dir/(name+'.json'),metric)
            del ids,mask;gc.collect()
        result.update(status='COMPLETE_A_RAW_DISTRIBUTION_DIAGNOSTIC_NO_ROLLOUT',raw_bytes_total=budget['used_bytes'],
            raw_limit_bytes=budget['limit_bytes'],max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_original_gate_recomputed=True,no_new_acceptance_threshold=True,automatic_B_C=False)
    except BaseException as error:
        result.update(status='FAILED_A_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        if native is not None and q is not None:
            for name,fn in native.items():setattr(q,name,fn)
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',result);fsync_chain(out,Path(plan['project_root']))
        print(json.dumps(dict(status=result['status'],result=ref(out/'result.json'))))


if __name__=='__main__':main()
