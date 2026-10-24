"""Owned CPU-only parent verification, exact delta concat and real PEFT tiny roundtrip."""
from pathlib import Path
import argparse,copy,hashlib,json,os,time
from delta_mean import validate_configs


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda:handle.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def syncdir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def save(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    syncdir(Path(path).parent)


def tiny_contract(output,config_path):
    import torch,peft,transformers
    from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict
    from transformers import PretrainedConfig
    assert peft.__version__=='0.21.2' and transformers.__version__=='5.20.2'
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.config=PretrainedConfig();self.config.model_type='tiny_cpu_contract'
            for name in sorted(json.loads(config_path.read_text())['target_modules']):
                setattr(self,name,torch.nn.Linear(7,5,bias=False))
        def prepare_inputs_for_generation(self,input_ids,**kwargs):return dict(input_ids=input_ids,**kwargs)
        def forward(self,input_ids=None,inputs_embeds=None,**kwargs):return self.q_proj(inputs_embeds)
    torch.manual_seed(20261024)
    base=Tiny();reference=copy.deepcopy(base)
    config=LoraConfig.from_pretrained(str(config_path.parent))
    model=get_peft_model(base,config,adapter_name='epoch1')
    model.add_adapter('epoch2',copy.deepcopy(config))
    modules=[]
    for name,layer in model.named_modules():
        if hasattr(layer,'lora_A') and 'epoch1' in layer.lora_A:
            with torch.no_grad():
                for adapter in ['epoch1','epoch2']:
                    layer.lora_A[adapter].weight.normal_(std=.1)
                    layer.lora_B[adapter].weight.normal_(std=.1)
            modules.append((name,layer))
    assert len(modules)==12
    model.add_weighted_adapter(['epoch1','epoch2'],[.5,.5],'mean',combination_type='cat')
    model.set_adapter('mean');model.eval()
    assert model.peft_config['mean'].r==model.peft_config['mean'].lora_alpha==32
    x=torch.randn(3,7);errors=[]
    for name,layer in modules:
        expected=layer.base_layer(x)+sum(.5*layer.scaling[a]*layer.lora_B[a](layer.lora_A[a](x)) for a in ['epoch1','epoch2'])
        actual=layer(x);errors.append(float((actual-expected).abs().max()))
        assert layer.scaling['mean']==1 and torch.allclose(actual,expected,rtol=2e-5,atol=2e-5)
    before={k:v.detach().clone() for k,v in get_peft_model_state_dict(model,adapter_name='mean').items()}
    model.save_pretrained(output/'tiny-save',selected_adapters=['mean'],safe_serialization=True)
    saved=output/'tiny-save/mean';assert json.loads((saved/'adapter_config.json').read_text())['r']==32
    loaded=PeftModel.from_pretrained(reference,saved,is_trainable=False);loaded.eval()
    after=get_peft_model_state_dict(loaded)
    assert set(before)==set(after) and all(torch.equal(before[k],after[k]) for k in before)
    assert torch.allclose(model(inputs_embeds=x),loaded(inputs_embeds=x),rtol=0,atol=0)
    assert not torch.cuda.is_initialized() and all(p.device.type=='cpu' for p in model.parameters())
    result=dict(status='PASS_REAL_PEFT0212_12FAMILY_CAT_AND_SAVE_LOAD_CPU',modules=12,
        old_r=16,new_r=32,new_alpha=32,new_scaling=1,tensor_keys=len(after),
        maximum_layer_abs_error=max(errors),loaded_state_values_bitexact=True,
        loaded_q_projection_output_bitexact=True,no_big_base_model_loaded=True,no_training_or_GPU=True,
        torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
        adapter_config=ref(saved/'adapter_config.json'),adapter=ref(saved/'adapter_model.safetensors'))
    save(output/'tiny-peft.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True,type=Path);args=parser.parse_args()
    start=time.monotonic();plan=json.loads(args.plan.read_text());out=Path(plan['output'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and plan['phase']=='CPU_SYNTHESIS_ONLY'
    assert plan['equal_weight']==.5 and plan['new_rank']==plan['new_alpha']==32
    out.mkdir(exist_ok=False);save(out/'started.json',dict(pid=os.getpid(),plan=ref(args.plan),at_unix=time.time(),CUDA_VISIBLE_DEVICES=''))
    try:
        import torch,peft,transformers
        from safetensors import safe_open
        from safetensors.torch import load_file,save_file
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__=='2.6.0+cu124' and peft.__version__=='0.21.2' and transformers.__version__=='5.20.2'
        configs=[];metadata=[];headers=[]
        expected=set(plan['expected_keys']);assert len(expected)==496
        for item in plan['parents']:
            for key in ['adapter','config','checkpoint','ACK']:
                q=Path(item[key]['path']);assert q.is_file() and not q.is_symlink() and ref(q)==item[key]
            cfg=json.loads(Path(item['config']['path']).read_text());cp=json.loads(Path(item['checkpoint']['path']).read_text())
            ack=json.loads(Path(item['ACK']['path']).read_text())
            assert cp['completed_epochs']==item['epoch'] and cp['global_step']==11*item['epoch']
            assert cp['files']['adapter/adapter_model.safetensors']==item['adapter']['sha256']
            assert ack['checkpoint_metadata_sha256']==item['checkpoint']['sha256'] and ack['all_members_reopened_and_SHA_verified']
            configs.append(cfg);metadata.append(cp['fingerprint'])
            with safe_open(item['adapter']['path'],framework='pt',device='cpu') as f:
                assert set(f.keys())==expected
                header={k:dict(shape=list(f.get_slice(k).get_shape()),dtype=f.get_slice(k).get_dtype()) for k in f.keys()}
            headers.append(header)
        assert metadata[0]==metadata[1] and headers[0]==headers[1]
        config=validate_configs(*configs);assert all(x['dtype']=='F32' for x in headers[0].values())
        assert sum(__import__('math').prod(x['shape']) for x in headers[0].values())==43278336
        save(out/'actual-parents.json',dict(parents=plan['parents'],headers=headers[0],fingerprint_equal=True,
            config_sha256=plan['parents'][0]['config']['sha256'],no_optimizer_loaded=True))
        tiny=tiny_contract(out,Path(plan['parents'][0]['config']['path']))
        first=load_file(plan['parents'][0]['adapter']['path'],device='cpu');second=load_file(plan['parents'][1]['adapter']['path'],device='cpu')
        merged={};checks=[];gen=torch.Generator(device='cpu').manual_seed(20261024)
        for key in sorted(k for k in expected if k.endswith('.lora_A.weight')):
            other=key.replace('.lora_A.weight','.lora_B.weight');a1,a2=first[key],second[key];b1,b2=first[other],second[other]
            assert a1.shape[0]==b1.shape[1]==16 and a1.shape==a2.shape and b1.shape==b2.shape
            assert all(torch.isfinite(t).all() for t in [a1,b1,a2,b2])
            merged[key]=torch.cat([a1,a2],dim=0);merged[other]=torch.cat([b1,b2],dim=1)
            assert torch.equal(merged[key][:16],a1) and torch.equal(merged[key][16:],a2)
            assert torch.equal(merged[other][:,:16],b1) and torch.equal(merged[other][:,16:],b2)
            x=torch.randn(1,a1.shape[1],generator=gen,dtype=torch.float64)
            expected_value=(x@a1.double().T)@b1.double().T+(x@a2.double().T)@b2.double().T
            value=(x@merged[key].double().T)@merged[other].double().T
            assert torch.allclose(value,expected_value,rtol=1e-10,atol=1e-10)
            checks.append(dict(name=key,maximum_abs_error=float((value-expected_value).abs().max())))
        assert len(checks)==248 and sum(t.numel() for t in merged.values())==86556672
        adapter=out/'adapter';adapter.mkdir();save_file(merged,str(adapter/'adapter_model.safetensors'),metadata={'format':'pt'})
        save(adapter/'adapter_config.json',config)
        with (adapter/'adapter_model.safetensors').open('rb') as handle:os.fsync(handle.fileno())
        syncdir(adapter);syncdir(out)
        with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
            assert set(f.keys())==expected and all(torch.equal(f.get_tensor(k),v) for k,v in merged.items())
        assert not torch.cuda.is_initialized()
        save(out/'complete.json',dict(status='COMPLETE_CPU_EXACT_EFFECTIVE_DELTA_MEAN_NOT_TRAINED_EPOCH',
            parents=plan['parents'],source=ref(__file__),plan=ref(args.plan),new_adapter=ref(adapter/'adapter_model.safetensors'),
            new_config=ref(adapter/'adapter_config.json'),new_r=32,new_alpha=32,new_scaling=1,
            adapter_parameters=86556672,expected_full_text_parameters=9040359936,
            original_tensors_bitexact_concatenated=True,reopened_saved_tensor_values_bitexact=True,
            CPU_FP64_projection_checks=checks,actual_tensor_keys=496,tiny_PEFT=ref(out/'tiny-peft.json'),
            no_optimizer_or_training_checkpoint_created=True,no_GPU=True,no_gold_quality=True,
            seconds=time.monotonic()-start,actual_base_model_loaded=False))
    except BaseException as error:
        save(out/'failure.json',dict(type=type(error).__name__,message=str(error),seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
