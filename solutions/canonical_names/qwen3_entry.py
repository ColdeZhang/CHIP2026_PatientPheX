"""Canonical-name adapter around the unchanged shared Qwen3 probe/train loops."""
import argparse
import gc
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

def checked(item):
    p=Path(item['path']);b=p.read_bytes()
    assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],str(p)
    return p

def generate(plan,out,rows,torch,model,c):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from schema_contract import CanonicalList,strict_parse
    tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True)
    native=Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json'))
    names=json.loads(gzip.decompress(checked(plan['names_gzip']).read_bytes()))
    codec=CanonicalList(native,names,151645);model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
    out.mkdir(exist_ok=False);records=[]
    for index,row in enumerate(rows):
        prompt=row['input_ids'];assert len(prompt)<=plan['inference']['max_input_reject']
        ids=torch.tensor([prompt],device='cuda',dtype=torch.long);start=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,num_beams=1,
                max_new_tokens=512,use_cache=True,logits_to_keep=1,eos_token_id=151645,pad_token_id=151643,
                prefix_allowed_tokens_fn=codec.callback(ids.shape[1]))
        torch.cuda.synchronize();tokens=generated[0,len(prompt):].tolist()
        natural=bool(tokens and tokens[-1]==151645);cap=len(tokens)>=512 and not natural
        identity=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],target_patient_id=row['target_patient_id'],
            index=index,input_tokens=len(prompt),output_tokens=len(tokens),natural_eos=natural,hit_token_limit=cap,
            seconds=time.monotonic()-start,fresh=True)
        # IDs and raw response are durable before any parser/assertion can reject them.
        c.save(out/f'{index:03d}-ids.json',{**identity,'generated_ids':tokens})
        raw=tk.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        body=tk.decode(tokens[:-1] if natural else tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        c.save(out/f'{index:03d}-raw.json',{**identity,'raw_with_specials':raw,'body':body})
        for k,t in enumerate(tokens):assert t in codec.allowed(tokens[:k]),(row['row_id'],k)
        parsed=None;error=None
        try:parsed=strict_parse(body,names,eos=natural,capped=cap)
        except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
        record={**identity,'hpo_ids':parsed if parsed is not None else [],'strict_complete':parsed is not None,
            'error':error,'all_generated_prefix_tokens_legal':True}
        c.save(out/f'{index:03d}-parsed.json',record);records.append(record)
        c.update(out/'progress.json',dict(completed=len(records),rows=len(rows),last_row=row['row_id']))
        del generated,ids
    with (out/'canonical.jsonl').open('x') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    return dict(rows=len(rows),strict_complete=sum(x['strict_complete'] for x in records),
        caps=sum(x['hit_token_limit'] for x in records),natural_eos=sum(x['natural_eos'] for x in records),
        all_generated_tokens_legal=True,raw_ids_and_text_before_parse=True,canonical=c.ref(out/'canonical.jsonl'),memory=c.memory(torch))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['probe','formal','infer']);p.add_argument('--plan',required=True)
    a=p.parse_args();start=time.monotonic();plan=json.loads(Path(a.plan).read_text())
    assert plan['authorized_phases']==['probe','formal','infer'] and plan['fixed_model']=='Qwen3-8B-canonical-name-v1'
    for item in plan['source_refs'].values():checked(item)
    # Import the exact shared helper/module pair, not a forked training loop.
    assert 'common' not in sys.modules and 'native' not in sys.modules
    sys.path.insert(0,plan['backend_source_dir']);c=importlib.import_module('common');shared=importlib.import_module('native')
    assert Path(c.__file__).resolve()==Path(plan['backend_source_dir'],'common.py').resolve()
    output=Path(plan['run_root'])/a.mode;output.mkdir(exist_ok=False)
    c.save(output/'started.json',dict(mode=a.mode,plan=c.ref(a.plan),source=c.ref(__file__),pid=os.getpid(),at_utc=c.stamp()))
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original phase deadline reached')))
    torch=None;stage='source_data_fresh_gate'
    try:
        cpu=json.loads(checked(plan['cpu_result']).read_text());assert cpu['counts']['train']['rows']==87
        native_receipt=json.loads(checked(plan['native_cpu_receipt']).read_text())
        assert native_receipt['status']=='PASS_ACTUAL_NATIVE_HF_PREFIX_PROCESSOR_NO_MODEL' and native_receipt['all126_native_HF_IDs_decode_exact']
        assert cpu['native_cpu_receipt']==plan['native_cpu_receipt']
        for split in ['train','held']:checked(cpu['counts'][split]['encoded'])
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid'] and plan['gpu_index']==0
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        selected=[line.split(', ') for line in gpu.splitlines() if plan['gpu_uuid'] in line]
        disk_free=shutil.disk_usage(output).free
        c.save(output/'fresh-gate.json',dict(gpu=gpu,compute_apps=apps,disk_free_bytes=disk_free,at_utc=c.stamp()))
        assert len(selected)==1 and selected[0][0]=='0' and int(selected[0][-1])>=23000
        assert not any(plan['gpu_uuid'] in line for line in apps.splitlines())
        assert disk_free>=3*1024**3
        if a.mode=='probe':assets=c.verify_assets(plan,full=True)
        else:
            previous=json.loads((Path(plan['run_root'])/'probe/verified-assets.json').read_text())
            assets=c.verify_assets(plan,full=False,previous=previous)
        c.save(output/'verified-assets.json',assets)
        adapter=None
        if a.mode=='formal':
            previous=json.loads((Path(plan['run_root'])/'probe/complete.json').read_text())
            assert previous['formal_budget_gate'] and previous['probe_state_must_be_discarded']
            assert previous['generation_interface']['all_generated_tokens_legal']
        if a.mode=='infer':
            formal=json.loads((Path(plan['run_root'])/'formal/complete.json').read_text());assert formal['global_step']==22 and len(formal['acknowledgements'])==2
            for epoch in [1,2]:
                ack=json.loads((Path(plan['run_root'])/f'formal/epoch-{epoch:03d}-persistent-ack.json').read_text())
                checked(ack['checkpoint'])
                for item in ack['files'].values():checked(item)
            adapter=str(Path(plan['run_root'])/'formal/epoch-002/adapter')
        stage='model_load';torch,np,model,named,model_manifest=c.load_model(plan,output,adapter)
        stage=a.mode
        if a.mode=='probe':
            result=shared.probe(plan,output,cpu,torch,model,named,model_manifest)
            assert result['indexed_forward_backward']==13 and result['optimizer_updates']==1
            # Shared function's optimizer was local and is now released; remove
            # its remaining gradients before the independent inference shape.
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            assert all(p.grad is None for p in model.parameters())
            train=shared.read_encoded(cpu,'train');row=train[plan['probe_generation_train_index']]
            probe_row={k:row[k] for k in ['row_id','pmc_id','target_patient_id']}
            probe_row['input_ids']=row['input_ids'][:row['prefix_tokens']]
            c.save(output/'before-generation.json',dict(all_gradients_released=True,optimizer_not_retained=True,training_row=probe_row['row_id'],memory=c.memory(torch)))
            result['generation_interface']=generate(plan,output/'generation-interface',[probe_row],torch,model,c)
            result['status']='PASS_CANONICAL_NEW_SHAPES_REAL_ADAM_AND_GENERATE_INTERFACE'
        elif a.mode=='formal':result=shared.train(plan,output,cpu,torch,np,model,named)
        else:
            rows=shared.read_encoded(cpu,'held');assert len(rows)==39
            result=generate(plan,output/'generation',rows,torch,model,c)
            result.update(status='COMPLETE_ALL39_FRESH_CANONICAL_NAMES_NO_QUALITY',fresh_rows=39)
        result.update(at_utc=c.stamp(),elapsed_seconds=time.monotonic()-start,phase=a.mode,no_held_labels_or_quality=True)
        c.save(output/'complete.json',result)
    except BaseException as exc:
        failure=dict(status='FAILED_ORIGINAL_NO_RETRY',stage=stage,error=type(exc).__name__,message=str(exc)[:3000],elapsed_seconds=time.monotonic()-start)
        if torch is not None:failure['memory']=c.memory(torch)
        c.save(output/'failure.json',failure);raise

if __name__=='__main__':main()
