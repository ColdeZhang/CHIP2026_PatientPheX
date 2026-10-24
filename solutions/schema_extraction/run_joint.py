"""Official JointIE runner, exhaustive declared candidate evidence and strict geometry."""
import argparse
from collections.abc import Mapping
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from prepare import original_span, patient_for_span, sha
from dataset_contract import check_dataset

def json_value(x):
    if x is None or isinstance(x,(str,bool,int)):return x
    if isinstance(x,float):
        if not math.isfinite(x):raise ValueError('Nonfinite candidate score')
        return x
    if dataclasses.is_dataclass(x):return {f.name:json_value(getattr(x,f.name)) for f in dataclasses.fields(x)}
    if isinstance(x,Mapping):
        if not all(isinstance(k,str) for k in x):raise TypeError('Non-string metadata key')
        return {k:json_value(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [json_value(v) for v in x]
    raise TypeError('Unsupported evidence type: '+type(x).__name__)

def write(path,value):
    temp=path.with_name(path.name+'.pending')
    with temp.open('w') as f:json.dump(json_value(value),f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def all_windows(docs,engine,protocol):
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from
    splitter=word_splitter_from(engine)
    windows=[]
    for doc in docs:
        chunks=split_text_into_chunks(doc['text'],protocol['chunking']['size'],protocol['chunking']['overlap'],splitter)
        word_count=sum(1 for _ in splitter(doc['text'],lower=False))
        if word_count:
            assert chunks[0].start_word==0 and chunks[-1].end_word==word_count
            assert all(a.end_word>=b.start_word for a,b in zip(chunks,chunks[1:]))
        for i,c in enumerate(chunks):
            assert c.text==doc['text'][c.start_char:c.end_char]
            windows.append((doc,i,c))
    return windows

def inspect_inputs(engine,windows,schema,protocol,model_path):
    from tokenizers import Tokenizer
    native=Tokenizer.from_file(str(Path(model_path)/"tokenizer.json"))
    token_cache={}
    compiled=engine.compile_schema(schema)
    census=[]
    for doc,i,c in windows:
        record=engine.model.processor.transform_record(c.text,compiled.model_schema,max_len=None)
        assert record.text in (c.text,c.text+'.')
        expected=list(engine.model.processor.word_splitter(record.text,lower=True))
        assert record.start_token_idx==[x[1] for x in expected]
        assert record.end_token_idx==[x[2] for x in expected]
        assert record.text_tokens==[x[0] for x in expected]
        combined=[]
        for struct in record.schema_tokens_list:
            combined.extend(struct);combined.append(engine.model.processor.SEP_STRUCT)
        if combined:combined.pop()
        combined.append(engine.model.processor.SEP_TEXT);combined.extend(record.text_tokens)
        expected_ids=[]
        for token in combined:
            if token not in token_cache:token_cache[token]=native.encode(token,add_special_tokens=False).ids
            expected_ids.extend(token_cache[token])
        assert expected_ids==record.input_ids,'Runtime tokenizer differs from published tokenizer.json processing chain'
        size=len(record.input_ids)
        assert size<=protocol['chunking']['maximum_wordpieces_with_schema'],size
        census.append(dict(pmc_id=doc['pmc_id'],window=i,start=c.start_char,end=c.end_char,
                           start_word=c.start_word,end_word=c.end_word,words=len(expected),native_appended_period=record.text!=c.text,input_ids=size,actual_input_ids=record.input_ids,
                           input_ids_sha256=hashlib.sha256(json.dumps(record.input_ids,separators=(',',':')).encode()).hexdigest()))
    return census,compiled

def remap_result(doc,chunk,result):
    entities={}
    for e in result.entities:
        assert chunk.text[e.start:e.end]==e.text
        a,b=e.start+chunk.start_char,e.end+chunk.start_char
        source,reason=original_span(doc,a,b)
        patient,p_reason=patient_for_span(doc,a,b) if e.type=='patient' else (None,None)
        entities[e.id]=dict(**e.to_dict(),joined_start=a,joined_end=b,source=source,
                            source_rejection=reason,patient_id=patient,patient_rejection=p_reason)
    return dict(entities=list(entities.values()),relations=[dict(**r.to_dict(),head_entity=entities[r.head],tail_entity=entities[r.tail]) for r in result.relations],feasible=result.feasible)

def native_catalog(doc,chunk,lattice):
    mentions=[]
    for mention in lattice.mentions:
        item=json_value(mention)
        if not 0 <= mention.start < mention.end <= len(lattice.start_mappings):
            raise ValueError('Native candidate outside caller token coordinates')
        a=lattice.start_mappings[mention.start]+chunk.start_char
        b=lattice.end_mappings[mention.end-1]+chunk.start_char
        source,reason=original_span(doc,a,b)
        patient,p_reason=patient_for_span(doc,a,b) if mention.entity_type=='patient' else (None,None)
        mentions.append(dict(**item,joined_start=a,joined_end=b,source=source,source_rejection=reason,
                             patient_id=patient,patient_rejection=p_reason))
    keys={(m['entity_type'],m['start'],m['end']) for m in mentions}
    for edge in lattice.edges:
        assert tuple(edge.head) in keys and tuple(edge.tail) in keys
    return dict(mentions=mentions,edges=json_value(lattice.edges),native_cap_limited=True)

def execute(engine,schema,config,doc,i,chunk,out):
    import torch
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    result,lattice=engine.extract(chunk.text,schema,config=config,return_lattice=True)
    if not hasattr(lattice,'mentions') or not hasattr(lattice,'edges'):raise TypeError('Expected boundary CandidateScoreSet')
    # Save the native raw object before application mapping; all proposed scores
    # remain available even if the selected relation fails patient/HPO mapping.
    raw=dict(pmc_id=doc['pmc_id'],window=i,chunk=json_value(chunk),result=result.to_dict(include_text=True),lattice=json_value(lattice))
    write(out/f'{doc["pmc_id"]}-{i:04d}.raw.json',raw)
    write(out/f'{doc["pmc_id"]}-{i:04d}.candidates.json',native_catalog(doc,chunk,lattice))
    mapped=remap_result(doc,chunk,result)
    record=dict(pmc_id=doc['pmc_id'],window=i,elapsed_seconds=time.monotonic()-start,
                native_mentions=len(lattice.mentions),native_edges=len(lattice.edges),
                cuda_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                mapped=mapped)
    write(out/f'{doc["pmc_id"]}-{i:04d}.mapped.json',record)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--input',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True);ap.add_argument('--mode',choices=['audit','probe','full'],required=True);ap.add_argument('--dtype',default='float32',choices=['float32','float16','bfloat16']);ap.add_argument('--dataset-contract');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        import torch,transformers,gliner2
        from gliner2.joint_ie import JointIE,JointIEConfig,JointSchema
        torch.set_num_threads(4)
        protocol=json.loads(Path(args.protocol).read_text())
        docs=[json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
        data=check_dataset(args.dataset_contract,args.input,docs,'prepared')
        schema=JointSchema.from_dict(protocol['schema']);config=JointIEConfig(**protocol['joint_config'])
        device='cpu' if args.mode=='audit' else 'cuda'
        if args.mode=='audit':assert not torch.cuda.is_available(),'Audit must hide CUDA devices'
        if device=='cuda':assert torch.cuda.is_available()
        engine=JointIE.from_pretrained(args.model,device='cpu',dtype='float32',local_files_only=True)
        # Stock loader calls strict load_state_dict; check actual resident inventory.
        parameters=sum(p.numel() for p in engine.model.parameters());assert parameters==193581591
        original=engine.model.processor.collate_fn_inference
        engine.model.processor.collate_fn_inference=functools.partial(original,error_policy='raise')
        inventory=dict(parameters=parameters,trainable=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
                       tensor_count=len(engine.model.state_dict()),torch=torch.__version__,transformers=transformers.__version__,
                       transformers_file=transformers.__file__,gliner2_file=gliner2.__file__,dtype=args.dtype,
                       checkpoint_strict_load=True,source_sha256=sha(__file__),protocol_sha256=sha(args.protocol),input_sha256=sha(args.input),
                       boundary_settings=engine.model.boundary_settings)
        write(out/'inventory.json',inventory)
        windows=all_windows(docs,engine,protocol);census,compiled=inspect_inputs(engine,windows,schema,protocol,args.model)
        write(out/'input-census.json',dict(documents=len(docs),patients=sum(len(d['patients']) for d in docs),windows=len(windows),
              max_input_ids=max(c['input_ids'] for c in census),word_and_character_mapping_exact=True,
              no_truncation=True,compiled_schema=compiled,rows=census))
        if args.mode=='audit':
            write(out/'complete.json',dict(status='CPU_STRICT_LOAD_AND_ALL_INPUTS_PASS',windows=len(windows),elapsed_seconds=time.monotonic()-start));return
        engine.to(device=device,dtype=args.dtype)
        if args.mode=='probe':
            # One largest actual input; fixed before any model output, no labels.
            largest=max(range(len(windows)),key=lambda i:census[i]['input_ids']);windows=[windows[largest]]
        reports=[]
        for doc,i,chunk in windows:
            reports.append(execute(engine,schema,config,doc,i,chunk,out))
            write(out/'progress.json',dict(mode=args.mode,completed_windows=len(reports),total_windows=len(windows),elapsed_seconds=time.monotonic()-start,
                                         mentions=sum(r['native_mentions'] for r in reports),edges=sum(r['native_edges'] for r in reports)))
            print(json.dumps({k:reports[-1][k] for k in ['pmc_id','window','elapsed_seconds','native_mentions','native_edges','cuda_peak_bytes']}),flush=True)
        write(out/'complete.json',dict(status='COMPLETE_NATIVE_CANDIDATE_EXTRACTION',mode=args.mode,windows=len(windows),
             documents=len({d['pmc_id'] for d,i,c in windows}),elapsed_seconds=time.monotonic()-start,
             max_window_seconds=max(r['elapsed_seconds'] for r in reports),peak_cuda_bytes=max(r['cuda_peak_bytes'] for r in reports),
             quality_evaluations=0))
    except BaseException as e:
        write(out/'failure.json',dict(error_type=type(e).__name__,message=str(e),elapsed_seconds=time.monotonic()-start,quality_evaluations=0));raise
if __name__=='__main__':main()
