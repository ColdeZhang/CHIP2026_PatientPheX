"""Read-only published/actual tokenizer decoding contracts; no model or new prediction."""
from pathlib import Path
import argparse,gzip,hashlib,inspect,json,os,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def state(obj):
 if obj is None:return None
 x=obj.__getstate__();return json.loads(x.decode() if isinstance(x,bytes) else x)
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 start=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();pp=Path(a.plan);plan=json.loads(pp.read_text());out=pp.parent;result={'status':'running_CPU_decoder_contract_only','source_sha256':sha(__file__),'plan_sha256':sha(pp)}
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
  for e in plan['refs'].values():assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
  model=Path(plan['model_dir']);tconfig=json.loads((model/'tokenizer_config.json').read_text());published=json.loads((model/'tokenizer.json').read_text());rows=[json.loads(x) for x in gzip.decompress(Path(plan['refs']['prompts']['path']).read_bytes()).decode().splitlines()];audit=json.loads(Path(plan['refs']['all80_audit']['path']).read_text());generated=json.loads(Path(plan['refs']['generated_row']['path']).read_text());assert len(rows)==len(audit)==80 and generated['generated_tokens']==2644
  import torch,transformers,tokenizers
  from transformers import AutoTokenizer,PreTrainedTokenizerFast
  assert not torch.cuda.is_initialized()
  auto=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);raw=tokenizers.Tokenizer.from_file(str(model/'tokenizer.json'))
  def content(n):
   x=tconfig.get(n);return x.get('content') if isinstance(x,dict) else x
  generic=PreTrainedTokenizerFast(tokenizer_file=str(model/'tokenizer.json'),bos_token=content('bos_token'),eos_token=content('eos_token'),pad_token=content('pad_token'),unk_token=content('unk_token'),chat_template=tconfig['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tconfig['model_max_length'])
  auto_backend=json.loads(auto.backend_tokenizer.to_str());generic_backend=json.loads(generic.backend_tokenizer.to_str());components={k:{'auto_equal_published':auto_backend.get(k)==published.get(k),'generic_equal_published':generic_backend.get(k)==published.get(k),'published_SHA':canon(published.get(k)),'auto_SHA':canon(auto_backend.get(k)),'generic_SHA':canon(generic_backend.get(k))} for k in ['normalizer','pre_tokenizer','post_processor','decoder','model','added_tokens']}
  decoder={'published':published['decoder'],'direct_from_file':state(raw.decoder),'AutoTokenizer':{'class':type(auto).__name__,'decoder_class':type(auto.backend_tokenizer.decoder).__name__,'state':state(auto.backend_tokenizer.decoder)},'generic':{'class':type(generic).__name__,'decoder_class':type(generic.backend_tokenizer.decoder).__name__,'state':state(generic.backend_tokenizer.decoder)},'components':components,'full_vocab_equal':raw.get_vocab()==auto.get_vocab()==generic.get_vocab()}
  class_file=Path(inspect.getsourcefile(type(auto)));class_source=inspect.getsource(type(auto));write(out/'actual-tokenizer-source.json',{'class':type(auto).__name__,'module':type(auto).__module__,'source_path':str(class_file),'file_sha256':sha(class_file),'class_source':class_source})
  specials=[content('bos_token'),content('eos_token'),content('pad_token'),'<｜User｜>','<｜Assistant｜>','<think>','</think>'];specials={s:{'auto':auto.convert_tokens_to_ids(s),'raw':raw.token_to_id(s),'generic':generic.convert_tokens_to_ids(s)} for s in set(specials) if s is not None};assert all(len(set(x.values()))==1 for x in specials.values())
  all80=[]
  for row,old in zip(rows,audit):
   rendered=auto.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True);gr=generic.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True);aid=auto.encode(rendered,add_special_tokens=False);rid=raw.encode(rendered,add_special_tokens=False).ids;gid=generic.encode(gr,add_special_tokens=False);native=generic.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
   assert row['row_id']==old['row_id'] and hashlib.sha256(rendered.encode()).hexdigest()==old['native_prompt_sha256'] and hashlib.sha256(json.dumps(aid,separators=(',',':')).encode()).hexdigest()==old['native_ids_sha256']
   all80.append({'row_id':row['row_id'],'input_tokens':len(aid),'native_old_ids_SHA':old['native_ids_sha256'],'generic_rendered_equal':rendered==gr,'raw_ids_equal':aid==rid,'generic_ids_equal':aid==gid==native,'auto_roundtrip_exact':auto.decode(aid,skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered,'raw_roundtrip_exact':raw.decode(rid,skip_special_tokens=False)==rendered,'generic_roundtrip_exact':generic.decode(gid,skip_special_tokens=False,clean_up_tokenization_spaces=False)==rendered,'add_special_tokens_prefix_all_exact':auto.encode(rendered,add_special_tokens=True)==raw.encode(rendered,add_special_tokens=True).ids==generic.encode(rendered,add_special_tokens=True)})
  fixtures=[]
  for text in ['ASCII plain 123,!?','中文病例','line1\nline2\n',' leading  and trailing ','中文 English\n  double\tspace','A\r\nB']:
   ids=auto.encode(text,add_special_tokens=False);ids_raw=raw.encode(text,add_special_tokens=False).ids;ids_generic=generic.encode(text,add_special_tokens=False);dec={'auto':auto.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),'raw':raw.decode(ids_raw,skip_special_tokens=False),'generic':generic.decode(ids_generic,skip_special_tokens=False,clean_up_tokenization_spaces=False)}
   fixtures.append({'input':text,'ids_equal':ids==ids_raw==ids_generic,'decoded':dec,'roundtrip':{k:v==text for k,v in dec.items()},'add_special_tokens_all_equal':auto.encode(text,add_special_tokens=True)==raw.encode(text,add_special_tokens=True).ids==generic.encode(text,add_special_tokens=True)})
  ids=generated['generated_ids'];decoded={'AutoTokenizer':auto.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),'raw_published_Tokenizer':raw.decode(ids,skip_special_tokens=False),'generic':generic.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)};assert decoded['AutoTokenizer']==generated['raw_generated_text'];assert ids[-1]==auto.eos_token_id==generic.eos_token_id
  # This is a separate CPU decoding/parser observation; the original GPU row is immutable.
  plain=raw.decode(ids[:-1],skip_special_tokens=False);first=next(r for r in rows if r['row_id']==generated['row_id']);import sys;sys.path.insert(0,str(Path(plan['refs']['strict_parser']['path']).parent));from strict_final import parse_final
  try:parsed=parse_final(plain,eos_observed=True,stopped_at_cap=False,code_map=first['code_map'],patient_ids=first['patient_ids']);parser={'valid':True,'parsed':parsed}
  except (ValueError,TypeError) as e:parser={'valid':False,'reason':str(e)}
  final=plain.split('</think>',1)[1].strip() if plain.count('</think>')==1 else None
  try:j=json.loads(final);json_observation={'valid_JSON_syntax':True,'keys':sorted(j) if isinstance(j,dict) else None,'expected_keys':sorted(first['code_map']),'exact_all_CID_keys':isinstance(j,dict) and set(j)==set(first['code_map'])}
  except (ValueError,TypeError):json_observation={'valid_JSON_syntax':False}
  write(out/'all80-roundtrip.json',all80);write(out/'fixture-roundtrip.json',fixtures);write(out/'generated-decode-comparison.json',{'original_generated_row_ref':plan['refs']['generated_row'],'same_generated_IDs':True,'decodings':decoded,'original_GPU_raw_unchanged':True,'separate_published_decoder_strict_parser':parser,'separate_published_decoder_JSON_observation':json_observation});write(out/'decoder-state.json',decoder)
  assert not torch.cuda.is_initialized() and all(sha(e['path'])==e['sha256'] for e in plan['refs'].values())
  checks=['generic_rendered_equal','raw_ids_equal','generic_ids_equal','auto_roundtrip_exact','raw_roundtrip_exact','generic_roundtrip_exact','add_special_tokens_prefix_all_exact'];result.update(status='COMPLETE_CPU_DECODER_CONTRACT_OBSERVATIONS_NO_MODEL',decoder_states=decoder,special_IDs=specials,all80={k:sum(x[k] for x in all80) for k in checks},all80_rows=80,fixtures={k:sum(x['roundtrip'][k] for x in fixtures) for k in ['auto','raw','generic']},fixture_count=len(fixtures),generated_decodings_raw_equal_generic=decoded['raw_published_Tokenizer']==decoded['generic'],generated_decodings_auto_equal_published=decoded['AutoTokenizer']==decoded['raw_published_Tokenizer'],separate_strict_parser=parser,separate_JSON=json_observation,CUDA_initialized=False,weights_model_install_new_prediction_quality=False,outputs={n:{'path':str(out/n),'bytes':(out/n).stat().st_size,'sha256':sha(out/n)} for n in ['actual-tokenizer-source.json','all80-roundtrip.json','fixture-roundtrip.json','generated-decode-comparison.json','decoder-state.json']},runtime={'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'torch':torch.__version__})
 except BaseException as e:result.update(status='FAILED_CPU_DIAGNOSTIC_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
 finally:result['program_seconds']=time.monotonic()-start;write(out/'result.json',result);print(json.dumps({k:result.get(k) for k in ['status','program_seconds','all80','fixtures','separate_strict_parser','separate_JSON','error']}))
if __name__=='__main__':main()
