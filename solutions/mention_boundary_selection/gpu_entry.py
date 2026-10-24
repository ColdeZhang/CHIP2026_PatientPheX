"""Reuse native Qwen training; add only mention constraint inference and timing."""
import argparse,gc,gzip,json,math,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'reuse/doc'))
import native
from common import checked,ref,save,sha,update,stamp,verify_assets,gpu_gate,load_model,memory
from schema import CandidateCodec,strict_parse


def decode_one(torch,model,tk,codec,row,out,index,max_new):
    from tokenizers import Tokenizer
    ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda');tick=time.monotonic()
    with torch.inference_mode():
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,num_beams=1,
            max_new_tokens=max_new,use_cache=True,prefix_allowed_tokens_fn=codec.callback(ids.shape[1]),
            eos_token_id=codec.eos_id,pad_token_id=codec.eos_id)
    torch.cuda.synchronize();seconds=time.monotonic()-tick;tokens=generated[0,ids.shape[1]:].tolist()
    raw=tk.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    record={'row_id':row['row_id'],'index':index,'input_tokens':ids.shape[1],'generated_ids':tokens,'raw_text_with_special':raw,'seconds':seconds,'max_new_tokens':max_new,'eos_token_id':codec.eos_id}
    save(out/f'{index:04d}.raw.json',record) # Before any parse or rejection.
    eos=bool(tokens and tokens[-1]==codec.eos_id);capped=len(tokens)>=max_new;body=tokens[:-1] if eos else tokens
    for k,token in enumerate(tokens):assert token in codec.allowed(tokens[:k]),('constraint violated',index,k)
    text=tk.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False);assert codec.native_backend.decode(body,skip_special_tokens=False)==text
    try:parsed=strict_parse(text,row['candidates'],eos=eos,capped=capped);reason=None
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
    elapsed=time.monotonic()-tick;assert len(adam)==1 and base['indexed_forward_backward']==11+len(cpu['pareto_indices'])
    save(out/'native-probe-unaltered-result.json',base)
    # Fixed before this probe: warm max seconds/token with one cold FB, real Adam and ACK allowance.
    times=[x['seconds'] for x in base['durations']];steady=max(x['seconds']/x['tokens'] for x in base['durations'][1:]);total=plan['recipe']['epochs']*cpu['counts']['train']['tokens']
    estimate=manifest['load_seconds']+times[0]+1.25*(total*steady+plan['recipe']['updates']*adam[0])+120
    row_bound=manifest['load_seconds']+times[0]+1.25*(6940*max(times[1:])+868*adam[0])+120
    cost={'rule':'load+cold_first_FB+1.25*(actual_full_training_tokens*max_remaining_steady_FB_seconds_per_input_token+868*actual_first_Adam)+120s_two_ACK_IO','actual_full_training_tokens':total,'optimizer_updates':plan['recipe']['updates'],'steady_max_seconds_per_token':steady,'actual_first_Adam_seconds':adam[0],'native_probe_wall_seconds':elapsed,'cold_first_FB_seconds':times[0],'estimated_formal_seconds':estimate,'budget_seconds':7200,'fits':estimate<=7200,'legacy_native_including_cold_scaled_estimate':base['formal_estimated_seconds'],'diagnostic_worst_row_everywhere_estimate':row_bound,'estimation_limit':'token scaling is an empirical cost estimate at observed shapes, not a worst-case runtime proof; recipe never shortened to fit'}
    save(out/'cost.json',cost)
    model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True);native_tokenizer=Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json'));codec=CandidateCodec(native_tokenizer,tk.eos_token_id,8);codec.native_backend=native_tokenizer
    rows=native.read_encoded(cpu,'train');source=[json.loads(line) for line in checked(plan['train_source']).read_text().splitlines()];assert len(rows)==len(source)==3470
    raw=out/'generation';raw.mkdir();generated=[]
    for state in ['HPO','NIL','NO','OTHER','BOUNDARY']:
        candidates=[i for i,r in enumerate(source) if ((r['target_status']=='HPO' and bool(json.loads(r['answer'])['selected'])) if state=='HPO' else (r['target_status']==state if state!='OTHER' else r['target_status']=='HPO' and json.loads(r['answer'])=={'selected':[],'missing':True}))]
        index=max(candidates,key=lambda i:len(rows[i]['input_ids']))
        row={**rows[index],'input_ids':rows[index]['input_ids'][:rows[index]['prefix_tokens']],'quote':source[index]['quote'],'candidates':source[index]['candidates']}
        item=decode_one(torch,model,tk,codec,row,raw,index,plan['max_new_tokens']);generated.append({'training_target_state':state,'index':index,'seconds':item['seconds'],'generated_tokens':len(item['generated_ids']),'eos':item['eos'],'capped':item['capped'],'rejection':item['rejection'],'generated_mode':item['parsed']['status'] if item['parsed'] else None})
        assert item['eos'] and not item['capped'] and item['rejection'] is None
    return {'status':'COMPLETE_CANDIDATE_REAL_FB_ADAM_AND_CLOSED_MODE_GENERATE_PROBE','indexed_forward_backward':base['indexed_forward_backward'],'optimizer_updates':1,'cost':cost,'formal_budget_gate':cost['fits'],'probe_state_must_be_discarded':True,'generation':generated,'all_generated_tokens_within_actual_constraint':True,'no_semantic_quality_claim':True,'memory':memory(torch)}


def infer(plan,out,cpu,torch,model):
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tk=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True);native_tokenizer=Tokenizer.from_file(str(Path(plan['model_dir'])/'tokenizer.json'));codec=CandidateCodec(native_tokenizer,tk.eos_token_id,8);codec.native_backend=native_tokenizer
    rows=native.read_encoded(cpu,'dev');assert len(rows)==1416;raw=out/'generation';raw.mkdir();results=[]
    for index,row in enumerate(rows):
        update(out/'progress.json',{'at_utc':stamp(),'completed':index,'row_id':row['row_id'],'stage':'constrained_generate'})
        value=decode_one(torch,model,tk,codec,row,raw,index,plan['max_new_tokens']);results.append(value)
    base=[json.loads(x) for x in checked(plan['baseline']).read_text().splitlines()];entities={str(x['pmc_id']):[] for x in base};seen=set();counts={'HPO':0,'NIL':0,'NO':0,'BOUNDARY':0,'HPO_missing':0,'HPO_empty_OTHER':0,'rejected':0,'cap':0}
    for row,result in zip(rows,results):
        counts['cap']+=result['capped'];parsed=result['parsed']
        if parsed is None:counts['rejected']+=1;continue
        counts[parsed['status']]+=1
        if parsed['status']=='HPO':counts['HPO_missing']+=parsed['missing'];counts['HPO_empty_OTHER']+=not parsed['selected']
        for hid in parsed['ids']:
            key=(row['pmc_id'],row['offset'],row['length'],hid,None)
            if key in seen:continue
            seen.add(key);entities[row['pmc_id']].append({'identifier':hid,'offset':row['offset'],'length':row['length'],'text':row['quote'],'type':'Phenotype','note':None})
    save(out/'entities.json',entities);save(out/'results.json',results)
    return {'status':'COMPLETE_FIXED_MENTION_CANDIDATE_INFERENCE','documents':16,'patients':39,'processed_rows':1416,'counts':counts,'accepted_entity_atoms':len(seen),'entities_sha256':sha(out/'entities.json'),'results_sha256':sha(out/'results.json'),'training_epoch':2,'memory':memory(torch),'no_quality_gold_read':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--mode',choices=['probe','formal','infer'],required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(plan['run_root'])/a.mode;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch=None;stage='source'
    try:
        for name,h in plan['source_files'].items():assert sha(Path(__file__).parent/name)==h,name
        cpu=json.loads(checked(plan['cpu_result']).read_text());assert cpu['status']=='PASS_COMPLETE_BOUNDARY_SELECTION_NATIVE_HF_AND_FINITE_GRAMMAR_CPU_ONLY';assert cpu['train_rows']==3470 and cpu['held_rows']==1416;assert cpu['schedule']=={'epochs':2,'updates_per_epoch':434,'updates':868,'last_group_size':6,'warmup_steps':44}
        previous_assets=Path(plan['run_root'])/'probe/verified-assets.json'
        gpu_gate(plan,out)
        inherited=json.loads(checked(plan['prior_verified_assets']).read_text()) if a.mode=='probe' else json.loads(previous_assets.read_text())
        assets=verify_assets(plan,full=False,previous=inherited);save(out/'verified-assets.json',assets)
        save(out/'asset-verification-scope.json',{'mode':'all13 current size/mtime_ns/inode exact plus prior full-SHA certificate','prior':plan['prior_verified_assets'] if a.mode=='probe' else str(previous_assets),'new_full_weight_hash':False})
        adapter=None
        if a.mode=='formal':
            p=json.loads((Path(plan['run_root'])/'probe/complete.json').read_text());assert p['formal_budget_gate'] and p['probe_state_must_be_discarded'];assert p['indexed_forward_backward']==11+len(cpu['pareto_indices']) and p['optimizer_updates']==1
        if a.mode=='infer':
            f=json.loads((Path(plan['run_root'])/'formal/complete.json').read_text());assert f['global_step']==868 and len(f['acknowledgements'])==2
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
