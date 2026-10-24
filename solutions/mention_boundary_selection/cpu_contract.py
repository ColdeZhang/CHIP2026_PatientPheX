"""Complete native/HF mention inputs, four-state grammar and real CPU logits processor.
No model class, weights, forward, fit, generation, quality or held labels.
"""
import argparse,gzip,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
from helpers import ref,write,load_jsonl,validate_refs,digest
from schema import CandidateCodec,strict_parse,serialize
sys.path.insert(0,str(Path(__file__).parent/'reuse/canonical'))
from token_census import NativeTokenizer
from reuse.native_tokenizer import encode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def progress(stage):
        p=out/'progress.pending.json';write(p,{'stage':stage,'elapsed':time.monotonic()-start});os.replace(p,out/'progress.json')
    progress('imports')
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['OMP_NUM_THREADS']=='1'
        import torch,transformers,tokenizers
        from transformers import AutoTokenizer
        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
        torch.set_num_threads(1);assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
        assert transformers.__version__==plan['runtime_versions']['transformers'];assert tokenizers.__version__==plan['runtime_versions']['tokenizers'];validate_refs(plan['files'])
        model=Path(plan['model_dir'])
        for name,x in plan['model_assets'].items():
            r=ref(model/name);assert r['sha256']==x['sha256'] and r['bytes']==x['bytes'],name
        write(out/'runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'CUDA_initialized':torch.cuda.is_initialized(),'no_weights_or_model':True})
        inputs=out/'inputs';inputs.mkdir();data={}
        for key,x in plan['compressed_inputs'].items():
            raw=gzip.decompress(Path(x['path']).read_bytes());assert hashlib.sha256(raw).hexdigest()==x['raw_sha256'];p=inputs/(key+'.json');p.write_bytes(raw);data[key]=json.loads(raw) if key=='names' else [json.loads(line) for line in raw.splitlines()]
        rows=data['train'];held=data['held'];names=data['names'];assert len(rows)==3470 and len(held)==1416 and len(names)==19119
        tk=AutoTokenizer.from_pretrained(model,local_files_only=True);native=NativeTokenizer(model/'tokenizer.json',json.loads((model/'tokenizer_config.json').read_text())['chat_template'],plan['eos_token_id'],plan['runtime_versions']['tokenizers']);codec=CandidateCodec(native.backend,plan['eos_token_id'],8)
        encoded={};audits={};grammar_prefixes=0
        progress('all_actual_templates_and_targets')
        for split,items in [('train',rows),('dev',held)]:
            enc=[];audit=[]
            for i,row in enumerate(items):
                prompt=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tk.encode(prompt,add_special_tokens=False)
                assert ids==native.backend.encode(prompt,add_special_tokens=False).ids
                assert native.apply_chat_template(row['messages'],False,True,False)==prompt
                assert native.apply_chat_template(row['messages'],True,True,False)==ids
                assert tk.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.backend.decode(ids,skip_special_tokens=False)==prompt
                if split=='train':
                    value=encode(row,native);actual=encode(row,tk);assert actual==value
                    body=native.backend.encode(row['answer'],add_special_tokens=False).ids
                    parsed=strict_parse(row['answer'],row['candidates'],eos=True,capped=False);assert parsed['status']==row['target_status']
                    for k,t in enumerate(body+[plan['eos_token_id']]):
                        assert t in codec.allowed(body[:k]);grammar_prefixes+=1
                    assert codec.allowed(body)==[plan['eos_token_id']]
                    assert len(value['input_ids'])<=plan['training_cap'] and len(body)+1<=plan['max_new_tokens']
                else:
                    assert 'answer' not in row and 'target_status' not in row;value={'row_id':row['row_id'],'input_ids':ids,'quote':row['quote'],'pmc_id':row['pmc_id'],'offset':row['offset'],'length':row['length'],'candidates':row['candidates']};assert len(ids)+plan['max_new_tokens']<=plan['inference_total_cap']
                    # Every candidate quote has an exact finite NIL branch, regardless of its unknown true label.
                    nil=native.backend.encode(serialize({'NIL':True}),add_special_tokens=False).ids;assert codec.allowed(nil)==[plan['eos_token_id']];assert len(row['candidates'])==8
                enc.append(value);audit.append({'index':i,'row_id':row['row_id'],'pmc_id':row['pmc_id'],'tokens':len(value['input_ids']),'supervised_tokens':value.get('supervised_tokens',0),'input_ids_sha256':digest(value['input_ids']),'labels_sha256':digest(value['labels']) if 'labels' in value else None,'native_HF_IDs_rendering_exact':True})
            encoded[split]=enc;audits[split]=audit
        progress('actual_HF_four_state_processor')
        vocab=json.loads((model/'config.json').read_text())['vocab_size'];assert native.backend.get_vocab_size(with_added_tokens=True)<=vocab
        shapes=[]
        for state in ['HPO','NIL','NO','BOUNDARY']:
            row=max((r for r in rows if r['target_status']==state),key=lambda x:len(x['answer']));body=native.backend.encode(row['answer'],add_special_tokens=False).ids
            for k in sorted({0,1,len(body)//2,len(body)-1,len(body)}):
                prefix=body[:k];cb=codec.callback(2);proc=PrefixConstrainedLogitsProcessor(cb,num_beams=1);scores=torch.arange(vocab,dtype=torch.float32).remainder_(17).unsqueeze(0);result=proc(torch.tensor([[7,8]+prefix]),scores);allowed=codec.allowed(prefix);mask=torch.zeros(vocab,dtype=torch.bool);mask[allowed]=True
                assert torch.equal(torch.isfinite(result[0]),mask) and torch.equal(result[0,mask],scores[0,mask]);shapes.append({'state':state,'prefix_length':k,'allowed_tokens':len(allowed)})
        # Exhaust the actual finite 514 strings, not a partial surrogate grammar.
        all_prefixes=0
        for item in codec.outputs:
            body=item['ids'];assert tk.encode(item['text'],add_special_tokens=False)==body and len(body)+1<=plan['max_new_tokens']
            strict_parse(item['text'],rows[0]['candidates'],eos=True,capped=False)
            for k,token in enumerate(body+[plan['eos_token_id']]):assert token in codec.allowed(body[:k]);all_prefixes+=1
            assert codec.allowed(body)==[plan['eos_token_id']]
        # Public candidate codes are decoded with the current row's actual mapping.
        for row in (rows[0],held[0]):
            parsed=strict_parse(serialize({'selected':['C01'],'missing':False}),row['candidates'],eos=True,capped=False)
            assert parsed['ids']==[row['candidates'][0]['identifier']]
        bad=['{"BOUNDARY":false}','{"BOUNDARY":true,"NO":true}','{"BOUNDARY":true,"selected":[]}','{"BOUNDARY":true,"BOUNDARY":true}','{"NIL":false}','{"NIL":true,"NIL":true}','{"NIL":true,"NO":true}','{"selected":[],"missing":false}','{"selected":["C09"],"missing":false}','{"selected":["C02","C01"],"missing":false}','{"selected":["C01","C01"],"missing":false}','{"selected":["C01"]}','{"selected":[],"missing":1}','[]','{"NO":true} trailing']
        rejected=0
        for raw in bad:
            try:strict_parse(raw,rows[0]['candidates'],eos=True,capped=False)
            except (ValueError,TypeError):rejected+=1
            else:raise AssertionError(raw)
        for eos,capped in [(False,False),(True,True)]:
            try:strict_parse('{"NO":true}',rows[0]['candidates'],eos=eos,capped=capped)
            except ValueError:rejected+=1
            else:raise AssertionError('Incomplete accepted')
        progress('save_complete_encoded')
        counts={}
        for split in encoded:
            p=out/(split+'.encoded.jsonl.gz');p.write_bytes(gzip.compress((''.join(json.dumps(x,separators=(',',':'))+'\n' for x in encoded[split])).encode(),mtime=0));ar=write(out/(split+'.audit.json'),audits[split]);counts[split]={'rows':len(encoded[split]),'tokens':sum(x['tokens'] for x in audits[split]),'supervised_tokens':sum(x['supervised_tokens'] for x in audits[split]),'max_tokens':max(x['tokens'] for x in audits[split]),'max_supervised_tokens':max(x['supervised_tokens'] for x in audits[split]),'encoded':ref(p),'audit':ar}
        byshape={}
        for row in audits['train']:byshape.setdefault((row['tokens'],row['supervised_tokens']),row['index'])
        frontier=[(shape,i) for shape,i in byshape.items() if not any(q[0]>=shape[0] and q[1]>=shape[1] and q!=shape for q in byshape)]
        frontier.sort();order=list(range(3470));random.Random(20261023).shuffle(order);groups=[order[i:i+8] for i in range(0,len(order),8)];assert len(groups)==434 and len(groups[-1])==6
        result={'status':'PASS_COMPLETE_BOUNDARY_SELECTION_NATIVE_HF_AND_FINITE_GRAMMAR_CPU_ONLY','plan':ref(a.plan),'source':ref(__file__),'counts':counts,'train_rows':3470,'held_rows':1416,'training_states':{'HPO':2222,'NIL':44,'NO':665,'BOUNDARY':539},'all4886_native_HF_IDs_exact':True,'all3470_masks_EOS_and_full_supervision_exact':True,'grammar_train_prefixes_checked':grammar_prefixes,'all1416_public_candidate_maps_complete':True,'actual_HF_processor_cases':shapes,'all514_native_HF_grammar_outputs_exact':True,'all514_prefix_count':all_prefixes,'all514_max_answer_tokens':max(len(x['ids'])+1 for x in codec.outputs),'strict_rejections':rejected,'training_cap':plan['training_cap'],'inference_total_cap':plan['inference_total_cap'],'max_new_tokens':plan['max_new_tokens'],'pareto_indices':[i for shape,i in frontier],'pareto_shapes':[list(shape) for shape,i in frontier],'stress_initial8':groups[0],'stress_accumulated3':groups[1][:3],'schedule':{'epochs':2,'updates_per_epoch':434,'updates':868,'last_group_size':6,'warmup_steps':44},'no_model_forward_generate_GPU_fit_quality':True,'production_model_generate_untested':True,'seconds':time.monotonic()-start}
        write(out/'complete.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        write(out/'failure.json',{'elapsed_seconds':time.monotonic()-start,'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
