"""One original901 stream, fixed published inputs/decoder, independent FLA backend."""
import argparse
from datetime import datetime,timezone
import hashlib,inspect,json,os,signal,time,traceback
from pathlib import Path
import legacy_runner as legacy
from common import checked,check_sources,ref,save,fsync_chain
from backend import verify_fla_backend

def ids_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--flow',required=True);ap.add_argument('--soft-deadline-monotonic',type=float,required=True);args=ap.parse_args()
    started=time.monotonic();plan_path=args.plan.resolve();plan=json.loads(plan_path.read_text());flow=plan['flows'][args.flow]
    out=plan_path.parent/'actual'/args.flow;out.mkdir(parents=True,exist_ok=False);fsync_chain(out,Path(plan['project_root']))
    report=dict(status='RUNNING_FRESH_ORIGINAL901_FLOW',flow=args.flow,source=ref(__file__),plan=ref(plan_path),started_utc=datetime.now(timezone.utc).isoformat(),gold_read=False,quality_calls=0,cached_rows=0,completed_rows=0)
    stage='verify_sources_and_inputs'
    def terminate(*_):raise TimeoutError('Original shared1800 deadline; no reset')
    signal.signal(signal.SIGTERM,terminate)
    try:
        assert args.flow in ('normal','reverse','public-new','patient') and plan['cap_seconds']==1800 and plan['soft_stop_seconds']==1770 and plan['term_grace_seconds']==30
        assert plan['generation_limit']==1600 and plan['input_limit']==14000 and flow['cached_rows']==0
        assert args.soft_deadline_monotonic>time.monotonic()
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        assert os.environ.get('PYTORCH_ALLOC_CONF') is None and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None
        check_sources(plan,plan_path.parent)
        rows=legacy.read_prompts(checked(flow['prompts']));audit=json.loads(checked(flow['native_audit']).read_text())
        assert audit['status']=='PASS_NATIVE_INPUT_BUDGET' and audit['source']['sha256']==plan['native_audit_source_sha256']
        assert audit['prompts']['sha256']==flow['prompts']['sha256'] and len(rows)==audit['count']==flow['rows']
        assert audit['total_input_tokens']==flow['input_tokens'] and audit['maximum_input_tokens']==flow['max_input_tokens']
        by_id={r['row_id']:r for r in audit['rows']};assert len(by_id)==len(rows)
        base_files=legacy.verify_model(plan['model'],checked(plan['model_source_manifest']))
        adapter_files={}
        for name,expected in flow['adapter_files'].items():
            p=Path(flow['adapter'])/name;actual=ref(p);assert all(actual[k]==expected[k] for k in ('bytes','sha256'));adapter_files[name]=actual
        import torch,transformers,peft,tokenizers
        from transformers import AutoTokenizer,Qwen3_5ForCausalLM,Qwen3_5TextConfig,StoppingCriteria,StoppingCriteriaList,LogitsProcessor,LogitsProcessorList
        from transformers.generation.utils import GenerationMixin
        from peft import PeftModel
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        assert (torch.__version__,transformers.__version__,peft.__version__,tokenizers.__version__)==('2.7.1+cu126','5.20.2','0.21.2','0.23.1')
        assert legacy.digest(q.__file__)==plan['runtime_sha256']['qwen_modeling'] and legacy.digest(inspect.getsourcefile(GenerationMixin))==plan['runtime_sha256']['generation_utils']
        fused,backend=verify_fla_backend(plan,q);save(out/'backend.json',backend)
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
        tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
        native=tokenizers.Tokenizer.from_file(str(Path(plan['model'])/'tokenizer.json'))
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        prepared=[]
        for index,row in enumerate(rows):
            rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(rendered,add_special_tokens=False);expected=by_id[row['row_id']]
            assert ids==native.encode(rendered,add_special_tokens=False).ids and ids_sha(ids)==expected['native_input_ids_sha256']
            assert hashlib.sha256(rendered.encode()).hexdigest()==expected['render_sha256']
            assert hashlib.sha256(tokenizer.decode(ids,skip_special_tokens=False).encode()).hexdigest()==expected['native_decoded_sha256']
            assert len(ids)==expected['input_tokens']<=14000 and str(row['pmc_id'])==expected['pmc_id']
            prepared.append((len(ids),row['row_id'],index,rendered,ids))
        prepared.sort(key=lambda x:(x[0],x[1]))
        assert [x[1] for x in prepared]==flow['ordered_row_ids']
        save(out/'input-and-assets.json',dict(flow=args.flow,rows=len(rows),prompts=flow['prompts'],native_audit=flow['native_audit'],input_tokens=sum(x[0] for x in prepared),max_input_tokens=max(x[0] for x in prepared),processing_original_indices=[x[2] for x in prepared],original_order_exact=True,base_files=base_files,adapter_files=adapter_files,actual_native_IDs_verified=True,runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__,matmul_TF32=False,cudnn_TF32=True,conv_native=True,FLA_GDN=True),decoder=flow['decoder'],hardware_disclosure=plan['hardware_disclosure']))
        stage='base_and_adapter_load';load_started=time.monotonic()
        config=Qwen3_5TextConfig.from_pretrained(plan['model'],local_files_only=True);assert config.max_position_embeddings>=15600
        model,loading=Qwen3_5ForCausalLM.from_pretrained(plan['model'],config=config,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        save(out/'loading.json',loading)
        assert not any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs'])
        assert all(k.startswith(('mtp.','model.mtp.','model.visual.','visual.')) for k in loading.get('unexpected_keys',[]))
        base_count=sum(x.numel() for x in model.parameters());assert base_count==legacy.TEXT_PARAMETERS
        model=PeftModel.from_pretrained(model,flow['adapter'],is_trainable=False);model.eval()
        total_count=sum(x.numel() for x in model.parameters());assert total_count==8997081600<10_000_000_000
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tokenizer.eos_token_id);eos.discard(None)
        assert sorted(eos)==[248044,248046] and tokenizer.pad_token_id==248044
        save(out/'generation.json',dict(actual=model.generation_config.to_dict(),do_sample=False,use_cache=True,max_new_tokens=1600,eos_ids=sorted(eos),pad_token_id=tokenizer.pad_token_id,thinking=False,finite_logits_only=(args.flow=='patient'),decoder=flow['decoder'],no_manual_seed=True))
        class FiniteLogitsOnly(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if not torch.isfinite(scores).all().item():raise FloatingPointError('Nonfinite generation logits; original patient policy stops without retry')
                return scores
        finite_processors=LogitsProcessorList([FiniteLogitsOnly()])
        for name in ['generated-ids','raw-responses','partial-generated-ids']:(out/name).mkdir()
        fsync_chain(out,Path(plan['project_root']));records=[];responses=[];load_seconds=time.monotonic()-load_started;generation_seconds=0.
        for length,row_id,index,rendered,ids in prepared:
            stage='row_'+str(index);row=rows[index]
            if time.monotonic()>=args.soft_deadline_monotonic:raise TimeoutError('Shared1770 soft deadline before next row')
            assert all(getattr(q,k) is v for k,v in fused.items())
            encoded=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to('cuda:0');assert encoded['input_ids'][0].tolist()==ids
            class EvidenceStop(StoppingCriteria):
                def __init__(self):self.last=0;self.wall_stop=False
                def __call__(self,input_ids,scores,**kwargs):
                    count=input_ids.shape[1]-len(ids);self.wall_stop=time.monotonic()>=args.soft_deadline_monotonic
                    if count-self.last>=32 or self.wall_stop:
                        save(out/'partial-generated-ids'/('%03d.json'%index),dict(row_id=row_id,generated_token_ids=input_ids[0,len(ids):].tolist(),wall_stop=self.wall_stop),replace=True);self.last=count
                    return self.wall_stop
            stop=EvidenceStop();extra=dict(logits_processor=finite_processors) if args.flow=='patient' else {}
            torch.cuda.synchronize();began=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=False,use_cache=True,max_new_tokens=1600,eos_token_id=sorted(eos),pad_token_id=tokenizer.pad_token_id,stopping_criteria=StoppingCriteriaList([stop]),**extra)
            torch.cuda.synchronize();seconds=time.monotonic()-began;generation_seconds+=seconds
            suffix=generated[0,length:];values=suffix.tolist();natural=bool(values and values[-1] in eos);cap=bool(len(values)>=1600 and not natural)
            ids_path=out/'generated-ids'/('%03d.json'%index)
            save(ids_path,dict(row_id=row_id,pmc_id=row['pmc_id'],original_index=index,input_ids_sha256=ids_sha(ids),generated_token_ids=values,input_tokens=length,output_tokens=len(values),eos_ids=sorted(eos),ended_with_eos=natural,hit_token_limit=cap,wall_stop=stop.wall_stop,input_prefix_exact=generated[0,:length].tolist()==ids,generation_seconds=seconds,raw_ids_before_decode=True))
            if args.flow=='patient':
                text_ids=values[:-1] if natural else values
                text=tokenizer.decode(text_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);native_text=native.decode(text_ids,skip_special_tokens=False)
            else:
                text=tokenizer.decode(suffix,skip_special_tokens=True);native_text=native.decode(values,skip_special_tokens=True)
            raw=dict(row_id=row_id,pmc_id=row['pmc_id'],response=text,input_tokens=length,output_tokens=len(values),removed_demonstrations=0,hit_token_limit=cap,batch_seconds=round(seconds,3),ended_with_eos=natural,wall_stop=stop.wall_stop)
            if args.flow=='patient':raw['generated_token_ids']=values
            raw_path=out/'raw-responses'/('%03d.json'%index);save(raw_path,raw)
            assert text==native_text and generated[0,:length].tolist()==ids
            responses.append(raw);records.append(dict(row_id=row_id,original_index=index,ids=ref(ids_path),raw=ref(raw_path),fresh=True));report['completed_rows']+=1
            save(out/'progress.json',dict(completed=report['completed_rows'],rows=len(rows),current_original_index=index,elapsed_seconds=time.monotonic()-started),replace=True)
            del generated,encoded
            if stop.wall_stop:raise TimeoutError('Shared deadline; current IDs and raw preserved before stop')
        assert report['completed_rows']==len(rows)==len(responses) and len({x['row_id'] for x in responses})==len(rows)
        with (out/'responses.jsonl').open('xb') as f:
            f.write(b''.join((json.dumps(x,ensure_ascii=False)+'\n').encode() for x in responses));f.flush();os.fsync(f.fileno())
        save(out/'record-manifest.json',records)
        report.update(status='COMPLETE_FRESH_ORIGINAL901_FLOW_NO_QUALITY',rows=len(rows),fresh_rows=len(rows),cached_rows=0,responses=ref(out/'responses.jsonl'),record_manifest=ref(out/'record-manifest.json'),backend=ref(out/'backend.json'),input_and_assets=ref(out/'input-and-assets.json'),load_seconds=load_seconds,generation_seconds=generation_seconds,natural_EOS_rows=sum(x['ended_with_eos'] for x in responses),token_cap_rows=sum(x['hit_token_limit'] for x in responses),base_parameters=base_count,total_parameters=total_count,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),raw_before_parser=True,parsed_rows=0,original_decoder_and_order_preserved=True)
        assert all(getattr(q,k) is v for k,v in fused.items());check_sources(plan,plan_path.parent);save(out/'complete.json',report)
    except BaseException as error:
        report.update(status='FAILED_ORIGINAL901_FLOW_PRESERVED',stage=stage,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save(out/'summary.json',report);fsync_chain(out,Path(plan['project_root']))
        print(json.dumps({k:report.get(k) for k in ['status','flow','completed_rows','elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
