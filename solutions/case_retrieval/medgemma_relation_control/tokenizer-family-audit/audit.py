"""One hidden-CUDA80x2 tokenizer contract after actual R1 family reconstruction failure."""
from pathlib import Path
import os,json,hashlib,gzip,time
started=time.monotonic();ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import transformers,tokenizers
from transformers import AutoTokenizer
from tokenizers import Tokenizer
sha=lambda b:hashlib.sha256(b).hexdigest()
jsha=lambda v:sha(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
raw=gzip.decompress((ROOT/'prompts.jsonl.gz').read_bytes());assert sha(raw)=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==80 and not any('answer' in r for r in rows)
models={'medgemma15':'research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30','qwen35':'case-retrieval/qwen35-preparation/models/Qwen3.5-9B'}
results={}
for name,directory in models.items():
 directory=Path(directory);tokenizer_file=directory/'tokenizer.json';published=json.loads(tokenizer_file.read_text());native=Tokenizer.from_file(str(tokenizer_file));auto=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False,padding_side='right');backend=json.loads(auto.backend_tokenizer.to_str());kwargs={'enable_thinking':False} if name=='qwen35' else {}
 records=[]
 for row in rows:
  rendered=auto.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,**kwargs)
  aids=auto(rendered,add_special_tokens=False)['input_ids'];nids=native.encode(rendered,add_special_tokens=False).ids
  chat=auto.apply_chat_template(row['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,**kwargs)
  decoded={}
  for source,ids in [('auto_ids',aids),('native_ids',nids)]:
   for skip in (False,True):
    left=auto.decode(ids,skip_special_tokens=skip,clean_up_tokenization_spaces=False);right=native.decode(ids,skip_special_tokens=skip)
    decoded[source+'_skip_'+str(skip)]=dict(equal=left==right,auto_sha256=sha(left.encode()),native_sha256=sha(right.encode()),auto_chars=len(left),native_chars=len(right))
  different=[i for i,(a,b) in enumerate(zip(aids,nids)) if a!=b]
  records.append(dict(row_id=row['row_id'],rendered_sha256=sha(rendered.encode()),auto_tokens=len(aids),native_tokens=len(nids),ids_equal=aids==nids,auto_ids_sha256=jsha(aids),native_ids_sha256=jsha(nids),chat_ids_equal_encode=chat==aids,first_difference_index=(different[0] if different else min(len(aids),len(nids)) if len(aids)!=len(nids) else None),decoded=decoded))
 components={key:dict(equal=backend.get(key)==published.get(key),auto_sha256=jsha(backend.get(key)),native_sha256=jsha(published.get(key)),auto_type=backend.get(key,{}).get('type') if isinstance(backend.get(key),dict) else None,native_type=published.get(key,{}).get('type') if isinstance(published.get(key),dict) else None) for key in ['normalizer','pre_tokenizer','post_processor','decoder','model','added_tokens']}
 modelkeys=set(backend['model'])|set(published['model']);modeldiff={k:dict(auto_sha256=jsha(backend['model'].get(k)),native_sha256=jsha(published['model'].get(k)),auto_length=len(backend['model'][k]) if isinstance(backend['model'].get(k),(list,dict,str)) else None,native_length=len(published['model'][k]) if isinstance(published['model'].get(k),(list,dict,str)) else None) for k in sorted(modelkeys) if backend['model'].get(k)!=published['model'].get(k)}
 specials={t:dict(auto_id=auto.convert_tokens_to_ids(t),native_id=native.token_to_id(t),equal=auto.convert_tokens_to_ids(t)==native.token_to_id(t)) for t in auto.all_special_tokens}
 files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in directory.iterdir() if p.name in ['tokenizer.json','tokenizer.model','tokenizer_config.json','chat_template.jinja','special_tokens_map.json','added_tokens.json','config.json']}
 results[name]=dict(directory=str(directory),auto_class=type(auto).__name__,auto_module=type(auto).__module__,template_kwargs=kwargs,tokenizer_assets=files,components=components,serialized_model_differing_fields=modeldiff,special_ids=specials,bos_token_id=auto.bos_token_id,eos_token_id=auto.eos_token_id,pad_token_id=auto.pad_token_id,all80_ids_equal=all(r['ids_equal'] for r in records),all80_chat_equals_auto_encode=all(r['chat_ids_equal_encode'] for r in records),all80_decodes_equal=all(v['equal'] for r in records for v in r['decoded'].values()),auto_total_tokens=sum(r['auto_tokens'] for r in records),native_total_tokens=sum(r['native_tokens'] for r in records),rows=records)
import torch
assert not torch.cuda.is_initialized()
report=dict(status='complete_one_CPU_audit_not_quality',source_sha256=sha(Path(__file__).read_bytes()),seconds=time.monotonic()-started,transformers=transformers.__version__,tokenizers=tokenizers.__version__,input_sha256=sha(raw),rows_per_model=80,model_weights_loaded=False,gold_read=False,prediction_or_quality=False,cuda_initialized=torch.cuda.is_initialized(),results=results)
with (ROOT/'complete.json').open('x') as h:json.dump(report,h,ensure_ascii=False,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
print(json.dumps({k:{q:v[q] for q in ['all80_ids_equal','all80_chat_equals_auto_encode','all80_decodes_equal','auto_total_tokens','native_total_tokens']} for k,v in results.items()}),flush=True)
print(json.dumps(dict(seconds=report['seconds'],complete_sha256=sha((ROOT/'complete.json').read_bytes()))),flush=True)
