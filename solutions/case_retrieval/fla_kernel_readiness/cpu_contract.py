"""CPU-only private FLA readiness. Never runs a forward or initializes CUDA."""
import argparse, hashlib, importlib.metadata as md, inspect, json, os, pathlib, sys, time, traceback

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def graph(fn, seen=None):
    seen=set() if seen is None else seen
    if id(fn) in seen or not inspect.isfunction(fn): return []
    seen.add(id(fn)); out=[{'name':fn.__qualname__,'module':fn.__module__,'file':inspect.getsourcefile(fn)}]
    for cell in fn.__closure__ or ():
        try: v=cell.cell_contents
        except ValueError: continue
        if inspect.isfunction(v): out += graph(v,seen)
    if hasattr(fn,'__wrapped__'): out += graph(fn.__wrapped__,seen)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--native-only',action='store_true');a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    r={'schema':'fla-private-cpu-contract-v1','started_unix':time.time(),'passed':False,'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES')}
    try:
        assert os.getenv('CUDA_VISIBLE_DEVICES')=='', 'CUDA must stay hidden'
        import torch
        assert not torch.cuda.is_initialized()
        r['packages']={k:{'version':md.version(k),'path':str(md.distribution(k).locate_file(''))} for k in ['torch','triton','transformers','peft','einops','fla-core']}
        assert md.version('transformers')=='5.20.2';assert md.version('fla-core')=='0.6.0';assert torch.__version__=='2.7.1+cu126'
        if not a.native_only:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        from transformers import AutoTokenizer,AutoConfig
        r['modeling_sha256']=sha(q.__file__);assert r['modeling_sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
        r['bindings']={name:graph(getattr(q,name)) for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','causal_conv1d_fn','causal_conv1d_update']}
        for name in ['torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:
            assert any(x['module'].startswith('fla.ops.gated_delta_rule') for x in r['bindings'][name]) == (not a.native_only),name
        for name in ['causal_conv1d_fn','causal_conv1d_update']:
            assert not any(x['module'].startswith('causal_conv1d') for x in r['bindings'][name]),name
        r['sources']={x['file']:sha(x['file']) for g in r['bindings'].values() for x in g if x['file']}
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).get_text_config();r['text_config']=cfg.to_dict()
        prompts=[]
        for i,n in enumerate([128,255,256,513]):
            sentence=f'Synthetic sequence {i}: amber cobalt silver green. '
            ids=tokenizer.encode(sentence*600,add_special_tokens=False)[:n]
            text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert tokenizer.encode(text,add_special_tokens=False)==ids
            continuation=tokenizer.encode(' The next colors are amber silver cobalt green.',add_special_tokens=False)[:8]
            assert len(ids)==n and len(continuation)==8
            prompts.append({'id':f'synthetic-{i}','input_ids':ids,'attention_mask':[1]*n,'text':text,'teacher_forced_ids':continuation})
        (out/'synthetic-prompts.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2)+'\n')
        r['synthetic_prompts_sha256']=sha(out/'synthetic-prompts.json')
        r['model_files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p) if p.stat().st_size<20_000_000 else 'inherited-existing-model-manifest-not-rehashed'} for p in pathlib.Path(a.model).iterdir() if p.is_file()}
        assert not torch.cuda.is_initialized();r['cuda_initialized_after']=False;r['passed']=True;r['scope']='native_cpu_preparation_only_fused_import_blocked' if a.native_only else 'fused_cpu_binding_contract'
    except Exception as exc:
        r['error']=f'{type(exc).__name__}: {exc}';r['traceback']=traceback.format_exc()
    r['elapsed_seconds']=time.time()-r['started_unix'];(out/'cpu-contract.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2,default=str));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
