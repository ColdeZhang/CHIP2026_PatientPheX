"""Two fixed native nonthinking streams, all160 fresh, one model/load/budget."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import probe_runner as probe
from sampling import GENERATION,SEEDS,row_seed
from cpu_gate import stat_ref
legacy=probe.legacy;save=probe.save;jsha=probe.jsha
progress={'completed':0,'seed':None,'original_index':None,'row_id':None}

def jsonl(path,records):
    import os
    with Path(path).open('x') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    assert plan['cap_seconds']==1800 and plan['term_grace_seconds']==30
    assert plan['seeds']==list(SEEDS) and plan['generation_kwargs']==GENERATION
    for name,r in plan['input_refs'].items():assert Path(r['remote_path']).stat().st_size==r['bytes'] and legacy.digest(r['remote_path'])==r['sha256'],name
    cpu=json.loads(Path(plan['cpu_gate']['path']).read_text());assert legacy.digest(plan['cpu_gate']['path'])==plan['cpu_gate']['sha256'] and cpu['status']=='PASS_NATIVE_ASSETS_ORIGINAL80_G_EOS_CPU_SAMPLING_INTERFACE'
    for path,expected in cpu['asset_stats'].items():assert stat_ref(path)==expected,path
    # CPU gate fully SHA-read the original immutable model/adapter immediately
    # before launch. Check identity/size/mtime unchanged rather than rehash 19GB.
    rows=legacy.read_prompts(plan['input_refs']['prompts']['remote_path']);audit=json.loads(Path(plan['input_refs']['audit']['remote_path']).read_text())
    assert len(rows)==len(audit)==80 and len({r['pmc_id'] for r in rows})==16
    import torch,transformers,peft,tokenizers
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,Qwen3_5TextConfig
    from peft import PeftModel
    assert {'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'tokenizers':tokenizers.__version__}==plan['versions']
    assert importlib.util.find_spec('fla') is None and importlib.util.find_spec('causal_conv1d') is None
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    tok=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True);native=tokenizers.Tokenizer.from_file(str(Path(plan['model'])/'tokenizer.json'))
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    assert tok.pad_token_id==plan['pad_token_id'];assert type(tok).__name__=='Qwen3_5Tokenizer'
    prepared=[]
    for i,(r,t) in enumerate(zip(rows,audit)):
        assert t['index']==i and t['row_id']==r['row_id'] and jsha(r)==t['expanded_row_sha256']
        rendered=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(rendered,add_special_tokens=False)
        assert len(ids)==t['expanded']['tokens']<=14400 and jsha(ids)==t['expanded']['input_ids_sha256']
        assert hashlib.sha256(rendered.encode()).hexdigest()==t['expanded']['rendered_sha256']
        prepared.append((len(ids),r['row_id'],i,rendered,ids))
    assert sum(p[0] for p in prepared)==670122 and max(p[0] for p in prepared)==14390
    prepared.sort(key=lambda p:(p[0],p[1]));config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True);assert config.max_position_embeddings>=15990
    save(out/'actual-input-and-assets.json',{'rows':80,'fresh_passes':2,'documents':16,'candidate_codes':sum(len(r['code_map']) for r in rows),'input_tokens_each_pass':670122,'all80_input_sha_exact':True,'max_input_tokens':14390,'no_truncation':True,'CPU_verified_assets':plan['cpu_gate'],'all_asset_stats_unchanged':True,'backend':'native PyTorch GDN/conv and SDPA; FLA and causal_conv1d absent','runtime':plan['versions'],'matmul_TF32':False,'cudnn_TF32':True,'processing_original_indices':[p[2] for p in prepared],'think_token_ids':cpu['think_token_ids'],'no_new_training_or_gold':True})
    begin=time.monotonic();model,loading=Qwen3_5ForCausalLM.from_pretrained(plan['model'],config=config,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    save(out/'base-loading.json',loading);assert not any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs']);assert all(k.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for k in loading.get('unexpected_keys',[]))
    base_count=sum(p.numel() for p in model.parameters());assert base_count==legacy.TEXT_PARAMETERS
    model=PeftModel.from_pretrained(model,plan['adapter'],is_trainable=False);model.eval();total_count=sum(p.numel() for p in model.parameters());assert total_count==8_997_081_600<10_000_000_000
    assert model.get_base_model().config._attn_implementation=='sdpa'
    loading_seconds=time.monotonic()-begin
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None);assert sorted(eos)==plan['eos_ids']
    save(out/'actual-generation.json',{'kwargs':{**GENERATION,'eos_token_id':sorted(eos),'pad_token_id':tok.pad_token_id},'presence_penalty':0,'presence_processor_added':False,'model_generation_config':model.generation_config.to_dict(),'seed_derivation':'int.from_bytes(SHA256(f"{seed}|{row_id}")[:8],"big") & ((1<<63)-1)','all160_fresh':True,'G_not_loaded_as_generation_cache':True})
    results=[];generation_total=0.
    for seed in SEEDS:
        seed_dir=out/str(seed);seed_dir.mkdir();(seed_dir/'generated-ids').mkdir();(seed_dir/'raw-responses').mkdir();(seed_dir/'canonical-records').mkdir();raws=[];canonical=[]
        for length,_,i,rendered,ids in prepared:
            r=rows[i];progress.update(seed=seed,original_index=i,row_id=r['row_id']);rs=row_seed(seed,r['row_id'])
            encoded={'input_ids':torch.tensor([ids],dtype=torch.long,device='cuda:0'),'attention_mask':torch.ones((1,length),dtype=torch.long,device='cuda:0')}
            random.seed(rs);torch.manual_seed(rs);torch.cuda.manual_seed_all(rs);torch.cuda.synchronize();begin=time.monotonic()
            with torch.inference_mode():generated=model.generate(**encoded,**GENERATION,eos_token_id=sorted(eos),pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize();seconds=time.monotonic()-begin;generation_total+=seconds;values=generated[0,length:].tolist();natural=bool(values and values[-1] in eos);hit=len(values)>=1600 and not natural
            ids_record={'row_id':r['row_id'],'seed':seed,'row_seed':rs,'input_ids_sha256':jsha(ids),'generated_ids':values,'configured_eos_ids':sorted(eos),'natural_eos':natural,'hit_token_limit':hit,'generation_seconds':seconds}
            save(seed_dir/'generated-ids'/f'{i:03d}.json',ids_record)
            response=tok.decode(values,skip_special_tokens=True);special=tok.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False);native_response=native.decode(values,skip_special_tokens=True);native_special=native.decode(values,skip_special_tokens=False)
            think=bool(set(values)&set(cpu['think_token_ids'].values())) or '<think>' in special or '</think>' in special
            raw={'row_id':r['row_id'],'pmc_id':r['pmc_id'],'response':response,'raw_with_special_tokens':special,'input_tokens':length,'output_tokens':len(values),'natural_eos':natural,'hit_token_limit':hit,'batch_seconds':seconds,'removed_demonstrations':0,'seed':seed,'row_seed':rs,'fresh':True,'thinking':False,'has_think_marker':think,'generation_limit':1600,'input_ids_sha256':jsha(ids),'rendered_sha256':audit[i]['expanded']['rendered_sha256'],'generated_ids':values,'configured_eos_ids':sorted(eos),'native_response':native_response,'native_raw_with_special_tokens':native_special}
            save(seed_dir/'raw-responses'/f'{i:03d}.json',raw)
            assert response==native_response and special==native_special,'Decoder identity differs; preserve raw, no silent repair'
            parsed=None;error=None
            try:
                if not natural or hit or think:raise ValueError('No natural EOS, token cap, or forbidden thinking marker')
                parsed=probe.parse_complete(response,r)
            except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
            c={'row_id':r['row_id'],'pmc_id':r['pmc_id'],'candidate_patients':parsed if parsed is not None else {},'strict_complete':parsed is not None,'abstention':error,'provenance':'fresh','seed':seed,'row_seed':rs}
            save(seed_dir/'canonical-records'/f'{i:03d}.json',c);raws.append(raw);canonical.append(c);progress['completed']+=1
            legacy.atomic_json(out/'progress.json',{**progress,'output_tokens':len(values),'strict_complete':parsed is not None,'generation_seconds':seconds})
            print(json.dumps({**progress,'output_tokens':len(values),'strict_complete':parsed is not None}),flush=True)
        assert len(raws)==80 and {r['row_id'] for r in raws}=={r['row_id'] for r in rows}
        jsonl(seed_dir/'responses.jsonl',raws);jsonl(seed_dir/'canonical.jsonl',canonical)
        results.append({'seed':seed,'rows':80,'fresh_rows':80,'strict_complete_rows':sum(c['strict_complete'] for c in canonical),'whole_row_abstentions':sum(not c['strict_complete'] for c in canonical),'input_tokens':sum(r['input_tokens'] for r in raws),'output_tokens':sum(r['output_tokens'] for r in raws),'natural_eos_rows':sum(r['natural_eos'] for r in raws),'cap_rows':sum(r['hit_token_limit'] for r in raws),'think_rows':sum(r['has_think_marker'] for r in raws),'responses_sha256':legacy.digest(seed_dir/'responses.jsonl'),'canonical_sha256':legacy.digest(seed_dir/'canonical.jsonl')})
    assert progress['completed']==160
    result={'status':'COMPLETE_FIXED160_FRESH_NATIVE_NONTHINKING_NO_QUALITY','seeds':results,'fresh_rows':160,'cached_rows':0,'loading_seconds':loading_seconds,'generation_seconds':generation_total,'elapsed_seconds':time.monotonic()-started,'base_parameters':base_count,'total_parameters':total_count,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'raw_ids_saved_before_decode':True,'raw_saved_before_parse':True,'raw_native_decoder_exact':True,'plan_sha256':legacy.digest(a.plan),'quality_calls':0,'gold_read':False}
    save(out/'complete.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        # A terminated/incomplete pass is never silently resumed or quality-scored.
        path=Path(sys.argv[sys.argv.index('--output')+1]);path.mkdir(parents=True,exist_ok=True)
        save(path/'failure.json',{'exception':type(exc).__name__,'message':str(exc),'progress':progress,'incomplete_no_quality':True});raise
