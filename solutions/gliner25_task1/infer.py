"""Complete F1 native entity API inference with all candidate scores and exact offsets."""
import argparse
import dataclasses
import functools
import json
import math
from pathlib import Path
import time
import traceback
from coordinates import original_span
from probe import write, sha


def main():
    ap=argparse.ArgumentParser()
    for n in ('model','published-tokenizer','input','training-complete','output'):ap.add_argument('--'+n,required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();progress={'windows':0}
    try:
        import torch
        from tokenizers import Tokenizer
        from gliner2.joint_ie import JointIE
        from gliner2.inference.chunking import split_text_into_chunks
        torch.set_num_threads(4)
        trained=json.loads(Path(args.training_complete).read_text());assert trained['status']=='FRESH_NATIVE_TRAIN32_10EPOCH_1600UPDATES_COMPLETE';assert trained['summary']['total_steps']==1600 and trained['summary']['total_epochs']==10
        ref=trained['final_state']['files']['model.safetensors'];assert (Path(args.model)/'model.safetensors').stat().st_size==ref['bytes'] and sha(Path(args.model)/'model.safetensors')==ref['sha256']
        assert sha(args.input)=='32599878940a98d32327541a79ba73fa43869954dfa41f7d040fda15f680d48b'
        assert sha(args.published_tokenizer)=='cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3'
        docs=[json.loads(x) for x in Path(args.input).read_text().splitlines()];assert len(docs)==16 and sum(len(d['patients']) for d in docs)==39
        engine=JointIE.from_pretrained(args.model,device='cpu',dtype='float32',local_files_only=True);model=engine.model
        assert sum(p.numel() for p in model.parameters())==193581591 and len(model.state_dict())==334
        model.to(device='cuda',dtype=torch.float32);model.eval();native=Tokenizer.from_file(args.published_tokenizer);cache={};windows=[]
        for doc in docs:
            chunks=split_text_into_chunks(doc['text'],384,64,model.processor.word_splitter);tokens=list(model.processor.word_splitter(doc['text'],lower=False))
            assert chunks[0].start_word==0 and chunks[-1].end_word==len(tokens) and all(a.end_word>=b.start_word for a,b in zip(chunks,chunks[1:]))
            for i,chunk in enumerate(chunks):assert chunk.text==doc['text'][chunk.start_char:chunk.end_char];windows.append((doc,i,chunk))
        assert len(windows)==164
        write(out/'inventory.json',{'parameters':193581591,'tensors':334,'dtype':'float32','model_sha256':ref['sha256'],'source_sha256':sha(__file__),'training_complete_sha256':sha(args.training_complete),'input_sha256':sha(args.input),'published_tokenizer_sha256':sha(args.published_tokenizer),'windows':164,'documents':16,'patients':39,'schema':['phenotype'],'threshold':.5,'overlap_policy':'allow','boundary_settings':dataclasses.asdict(model.boundary_settings)})
        ctx={};native_extract=model._extract_from_batch
        @functools.wraps(native_extract)
        def observed_extract(batch,*a,**kw):
            assert len(batch)==1;combined=[]
            for item in batch.schema_tokens_list[0]:combined.extend(item);combined.append(model.processor.SEP_STRUCT)
            if combined:combined.pop()
            combined.append(model.processor.SEP_TEXT);combined.extend(batch.text_tokens[0]);expected=[]
            for token in combined:
                if token not in cache:cache[token]=native.encode(token,add_special_tokens=False).ids
                expected.extend(cache[token])
            n=int(batch.attention_mask[0].sum());actual=batch.input_ids[0,:n].tolist();ctx['batch']=batch
            write(out/(ctx['prefix']+'.input.json'),{'pmc_id':ctx['doc']['pmc_id'],'window':ctx['window'],'schema_tokens':batch.schema_tokens_list[0],'query_layout':dataclasses.asdict(batch.query_layouts[0]),'actual_ids':actual,'published_native_ids':expected,'start_mappings':batch.start_mappings[0],'end_mappings':batch.end_mappings[0],'model_appended_period':batch.original_texts[0]!=ctx['chunk'].text})
            assert actual==expected and n<=4096 and batch.original_texts[0] in (ctx['chunk'].text,ctx['chunk'].text+'.')
            assert batch.query_marker_mask.tolist()==[[True]]
            return native_extract(batch,*a,**kw)
        model._extract_from_batch=observed_extract
        def capture_head(module,inputs,output):
            batch=ctx['batch'];c=output.candidates;assert c.indices.shape[0:2]==(1,1)
            items=[];chunk=ctx['chunk'];doc=ctx['doc']
            for k,keep in enumerate(c.valid_mask[0,0].tolist()):
                if not keep:continue
                start,end=c.indices[0,0,k].tolist();logit=float(c.pair_logits[0,0,k].item());assert math.isfinite(logit)
                assert 0<=start<end<=len(batch.start_mappings[0]);a,b=batch.start_mappings[0][start],batch.end_mappings[0][end-1]
                source,reason=original_span(doc,a+chunk.start_char,b+chunk.start_char) if 0<=a<b<=len(chunk.text) else (None,'outside_original_chunk')
                text=batch.original_texts[0][a:b]
                if source and source['text']!=text:source,reason=None,'native_text_not_original_slice'
                items.append({'candidate':k,'token_start':start,'token_end':end,'chunk_start':a,'chunk_end':b,'text':text,'logit':logit,'probability':float(torch.sigmoid(c.pair_logits[0,0,k]/model.boundary_settings.pair_temperature).item()),'source':source,'rejection':reason})
            write(out/(ctx['prefix']+'.candidates.json'),{'pmc_id':doc['pmc_id'],'window':ctx['window'],'declared_padded_cap':int(c.indices.shape[2]),'valid_candidates':len(items),'native_cap_limited':True,'null_logits':None if output.null_logits is None else output.null_logits.detach().cpu().tolist(),'count_log_rates':None if output.count_log_rates is None else output.count_log_rates.detach().cpu().tolist(),'candidates':items})
        hook=model.boundary_head.register_forward_hook(capture_head);selected=[];census=[]
        with torch.inference_mode():
            for doc,i,chunk in windows:
                prefix=f'{doc["pmc_id"]}-{i:04d}';ctx.update(doc=doc,window=i,chunk=chunk,prefix=prefix);t=time.monotonic()
                result=model.extract_entities(chunk.text,['phenotype'],threshold=.5,include_confidence=True,include_spans=True,overlap_policy='allow')
                write(out/(prefix+'.raw.json'),{'pmc_id':doc['pmc_id'],'window':i,'chunk':dataclasses.asdict(chunk),'native_result':result})
                assert set(result)=={'entities'} and set(result['entities'])=={'phenotype'}
                for e in result['entities']['phenotype']:
                    a,b=e['start'],e['end'];source,reason=original_span(doc,a+chunk.start_char,b+chunk.start_char) if 0<=a<b<=len(chunk.text) else (None,'outside_original_chunk')
                    if source and source['text']!=e['text']:source,reason=None,'native_text_not_original_slice'
                    selected.append({'pmc_id':doc['pmc_id'],'window':i,'confidence':e['confidence'],'native':e,'source':source,'rejection':reason})
                census.append({'pmc_id':doc['pmc_id'],'window':i,'start_char':chunk.start_char,'end_char':chunk.end_char,'start_word':chunk.start_word,'end_word':chunk.end_word,'selected':len(result['entities']['phenotype']),'seconds':time.monotonic()-t})
                progress={'windows':len(census),'target_windows':164,'selected':len(selected),'elapsed_seconds':time.monotonic()-started};write(out/'progress.json',progress)
        hook.remove();write(out/'selected.json',selected);write(out/'windows.json',census)
        report={'status':'COMPLETE_F1_NATIVE_TASK1_EXTRACTION','documents':16,'patients':39,'windows':164,'selected_windows_entities':len(selected),'legal_windows_entities':sum(x['rejection'] is None for x in selected),'selected_sha256':sha(out/'selected.json'),'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'source_sha256':sha(__file__),'quality_evaluations':0}
        write(out/'complete.json',report);print(json.dumps(report),flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'progress':progress,'elapsed_seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
