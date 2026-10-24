"""Reuse native Qwen training; add only mention constraint inference and timing."""
import argparse,gc,gzip,json,math,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'reuse/doc'))
import native
from common import checked,ref,save,sha,update,stamp,verify_assets,gpu_gate,load_model,memory
from mention_schema import MentionCodec,strict_parse


def decode_one(torch,model,tk,codec,row,out,index,max_new):
    from tokenizers import Tokenizer
    ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda');tick=time.monotonic()
    with torch.inference_mode():
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,num_beams=1,
            max_new_tokens=max_new,use_cache=True,prefix_allowed_tokens_fn=codec.callback(ids.shape[1],[row['quote']]),
            eos_token_id=codec.eos_id,pad_token_id=codec.eos_id)
    torch.cuda.synchronize();seconds=time.monotonic()-tick;tokens=generated[0,ids.shape[1]:].tolist()
    raw=tk.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    record={'row_id':row['row_id'],'index':index,'input_tokens':ids.shape[1],'generated_ids':tokens,'raw_text_with_special':raw,'seconds':seconds,'max_new_tokens':max_new,'eos_token_id':codec.eos_id}
    save(out/f'{index:04d}.raw.json',record) # Before any parse or rejection.
    eos=bool(tokens and tokens[-1]==codec.eos_id);capped=len(tokens)>=max_new;body=tokens[:-1] if eos else tokens
    for k,token in enumerate(tokens):assert token in codec.allowed(tokens[:k],row['quote']),('constraint violated',index,k)
    text=tk.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert codec.tokenizer.decode(body,skip_special_tokens=False)==text
    try:parsed=strict_parse(text,row['quote'],codec.names,eos=eos,capped=capped);reason=None
    except (ValueError,TypeError) as exc:parsed=None;reason=str(exc)
    result={**record,'text':text,'eos':eos,'capped':capped,'parsed':parsed,'rejection':reason};save(out/f'{index:04d}.parsed.json',result)
    del ids,generated;return result


def probe(plan,out,cpu,torch,model,named,manifest):
    # Instrument the original Adam step transparently; never replace its math.
    original=torch.optim.AdamW.step;adam=[]
    def timed(optimizer,*args,**kwargs):
        torch.cuda.synchronize();tick=time.monotonic();value=original(optimizer,*args,**kwargs);torch.cuda.synchronize();adam.append(time.monotonic()-tick);return value
    torch.optim.AdamW.step=timed;tick=time.monotonic()
    try:base=native.probe(plan,out,cpu,torch,model,named,manifest)
    finally:torch.optim.AdamW.step=original
    elapsed=time.monotonic()-tick;assert len(adam)==1 and base['indexed_forward_backward']==14
    save(out/'native-probe-unaltered-result.json',base)
    # Prewritten bound: one cold FB, maximum of all13 subsequent actual FBs (including all Pareto),
    # one observed Adam allocation/update cost for every planned update, +25% and120s ACK/I/O.
    times=[x['seconds'] for x in base['durations']];steady=max(times[1:]);estimate=manifest['load_seconds']+times[0]+1.25*(4706*steady+590*adam[0])+120
    cost={'rule':'load+cold_first_FB+1.25*(4706*max13_steady_FB+590*actual_first_Adam)+120s_two_ACK_IO_allowance','steady_FB_max_seconds':steady,'actual_first_Adam_seconds':adam[0],'native_probe_wall_seconds':elapsed,'cold_first_FB_seconds':times[0],'estimated_formal_seconds':estimate,'budget_seconds':2400,'fits':estimate<=2400,'legacy_native_token_scaled_estimate':base['formal_estimated_seconds'],'legacy_estimate_not_authoritative_for_this_new_step_count':True}
    save(out/'cost.json',cost)
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True);codec=MentionCodec(Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json')),json.loads(checked(plan['public_names']).read_text()),tk.eos_token_id)
    rows=native.read_encoded(cpu,'train');source=[json.loads(line) for line in checked(plan['train_source']).read_text().splitlines()];assert len(rows)==len(source)==2353
    raw=out/'generation';raw.mkdir();generated=[]
    for state in ['HPO','NIL','NO']:
        index=max((i for i,r in enumerate(source) if r['target_status']==state),key=lambda i:len(rows[i]['input_ids']))
        row={**rows[index],'input_ids':rows[index]['input_ids'][:rows[index]['prefix_tokens']],'quote':source[index]['quote']}
        item=decode_one(torch,model,tk,codec,row,raw,index,plan['max_new_tokens']);generated.append({'training_state':state,'index':index,'seconds':item['seconds'],'generated_tokens':len(item['generated_ids']),'eos':item['eos'],'capped':item['capped'],'rejection':item['rejection']})
    return {'status':'COMPLETE_MENTION_REAL_FB_ADAM_AND_THREE_STATE_GENERATE_PROBE','indexed_forward_backward':14,'optimizer_updates':1,'cost':cost,'formal_budget_gate':cost['fits'],'probe_state_must_be_discarded':True,'generation':generated,'all_generated_tokens_within_actual_constraint':True,'no_semantic_quality_claim':True,'memory':memory(torch)}


def infer(plan,out,cpu,torch,model):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True);codec=MentionCodec(Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json')),json.loads(checked(plan['public_names']).read_text()),tk.eos_token_id)
    rows=native.read_encoded(cpu,'dev');assert len(rows)==1416;raw=out/'generation';raw.mkdir();results=[]
    for index,row in enumerate(rows):
        update(out/'progress.json',{'at_utc':stamp(),'completed':index,'row_id':row['row_id'],'stage':'constrained_generate'})
        value=decode_one(torch,model,tk,codec,row,raw,index,plan['max_new_tokens']);results.append(value)
    base=[json.loads(x) for x in checked(plan['baseline']).read_text().splitlines()];entities={str(x['pmc_id']):[] for x in base};seen=set();counts={'HPO':0,'NIL':0,'NO':0,'rejected':0,'cap':0}
    for row,result in zip(rows,results):
        counts['cap']+=result['capped'];parsed=result['parsed']
        if parsed is None:counts['rejected']+=1;continue
        counts[parsed['status']]+=1
        for hid in parsed['ids']:
            key=(row['pmc_id'],row['offset'],row['length'],hid,None)
            if key in seen:continue
            seen.add(key);entities[row['pmc_id']].append({'identifier':hid,'offset':row['offset'],'length':row['length'],'text':row['quote'],'type':'Phenotype','note':None})
    save(out/'entities.json',entities);save(out/'results.json',results)
    return {'status':'COMPLETE_FIXED_MENTION_CANONICAL_INFERENCE','documents':16,'patients':39,'processed_rows':1416,'counts':counts,'accepted_entity_atoms':len(seen),'entities_sha256':sha(out/'entities.json'),'results_sha256':sha(out/'results.json'),'training_epoch':2,'memory':memory(torch),'no_quality_gold_read':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['probe','formal','infer'],required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(plan['run_root'])/a.mode;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch=None;stage='source'
    try:
        for name,h in plan['source_files'].items():assert sha(Path(__file__).parent/name)==h,name
        cpu=json.loads(checked(plan['cpu_result']).read_text());assert cpu['status']=='PASS_COMPLETE_MENTION_NATIVE_HF_INPUTS_AND_THREE_STATE_CONSTRAINT_CPU_ONLY';assert cpu['train_rows']==2353 and cpu['held_rows']==1416;assert cpu['schedule']=={'epochs':2,'updates_per_epoch':295,'updates':590,'last_group_size':1,'warmup_steps':30}
        previous_assets=checked(plan['prior_probe_assets']) if 'prior_probe_assets' in plan else Path(plan['run_root'])/'probe/verified-assets.json'
        gpu_gate(plan,out);assets=verify_assets(plan,full=a.mode=='probe',previous=None if a.mode=='probe' else json.loads(previous_assets.read_text()));save(out/'verified-assets.json',assets)
        adapter=None
        if a.mode=='formal':
            if 'prior_probe_complete' in plan:
                p=json.loads(checked(plan['prior_probe_complete']).read_text())
                assert p['status']=='COMPLETE_MENTION_REAL_FB_ADAM_AND_THREE_STATE_GENERATE_PROBE' and p['indexed_forward_backward']==14 and p['optimizer_updates']==1
                assert p['probe_state_must_be_discarded'] and p['cost']['estimated_formal_seconds']<=plan['budgets']['formal']
                assert plan['phases']==['formal','infer'] and plan['budget_revision']['original_gate_closed'] is True
            else:
                p=json.loads((Path(plan['run_root'])/'probe/complete.json').read_text());assert p['formal_budget_gate'] and p['probe_state_must_be_discarded']
        if a.mode=='infer':
            f=json.loads((Path(plan['run_root'])/'formal/complete.json').read_text());assert f['global_step']==590 and len(f['acknowledgements'])==2
            for epoch in [1,2]:
                ack=json.loads((Path(plan['run_root'])/f'formal/epoch-{epoch:03d}-persistent-ack.json').read_text());checked(ack['checkpoint'])
                for item in ack['files'].values():checked(item)
            adapter=str(Path(plan['run_root'])/'formal/epoch-002/adapter')
        stage='model_load';torch,np,model,named,manifest=load_model(plan,out,adapter);stage=a.mode
        if a.mode=='probe':result=probe(plan,out,cpu,torch,model,named,manifest)
        elif a.mode=='formal':result=native.train(plan,out,cpu,torch,np,model,named)
        else:result=infer(plan,out,cpu,torch,model)
        result.update(at_utc=stamp(),elapsed_seconds=time.monotonic()-start,phase=a.mode);save(out/'complete.json',result)
    except BaseException:
        save(out/'failure.json',{'stage':stage,'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'memory':memory(torch) if torch is not None else None});raise

if __name__=='__main__':main()
