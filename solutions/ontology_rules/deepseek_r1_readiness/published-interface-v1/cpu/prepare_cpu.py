"""Freeze published raw/generic token IDs and explicit public-runtime config semantics."""
from pathlib import Path
import argparse,copy,datetime,gzip,hashlib,inspect,json,os,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ref(p):return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def main():
 st=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();pp=Path(a.plan).resolve();plan=json.loads(pp.read_text());out=pp.parent/'actual';out.mkdir(exist_ok=False);result={'status':'running_published_interface_CPU','source':ref(__file__),'plan':ref(pp)}
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.getuid()==1006 and sha(__file__)==plan['source_sha256']
  for e in plan['refs'].values():assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
  model=Path(plan['model_dir']);tc=json.loads((model/'tokenizer_config.json').read_text());cfg_original=json.loads((model/'config.json').read_text());original_json=json.loads((model/'tokenizer.json').read_text());rawbytes=gzip.decompress(Path(plan['refs']['prompts']['path']).read_bytes());assert hashlib.sha256(rawbytes).hexdigest()=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe';rows=[json.loads(x) for x in rawbytes.decode().splitlines()];old=json.loads(Path(plan['refs']['old_all80_audit']['path']).read_text());assert len(rows)==len(old)==80
  import torch,transformers,tokenizers
  from transformers import PreTrainedTokenizerFast,AutoConfig
  from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
  assert not torch.cuda.is_initialized()
  def content(n):
   x=tc.get(n);return x.get('content') if isinstance(x,dict) else x
  raw=tokenizers.Tokenizer.from_file(str(model/'tokenizer.json'));generic=PreTrainedTokenizerFast(tokenizer_file=str(model/'tokenizer.json'),bos_token=content('bos_token'),eos_token=content('eos_token'),pad_token=content('pad_token'),unk_token=content('unk_token'),chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tc['model_max_length'])
  raw_state=json.loads(raw.to_str());generic_state=json.loads(generic.backend_tokenizer.to_str());keys=['model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'];component_equal={k:raw_state.get(k)==generic_state.get(k) for k in keys};assert all(component_equal.values()) and raw.get_vocab()==generic.get_vocab()
  assert original_json['model']['vocab']==raw_state['model']['vocab']
  def merge_pairs(values):
   pairs=[tuple(x.split(' ')) if isinstance(x,str) else tuple(x) for x in values];assert all(len(x)==2 and all(isinstance(v,str) for v in x) for x in pairs);return pairs
  assert merge_pairs(original_json['model']['merges'])==merge_pairs(raw_state['model']['merges'])
  model_default_differences={k:{'published':original_json['model'].get(k),'loaded_native':raw_state['model'].get(k)} for k in set(original_json['model'])|set(raw_state['model']) if k not in ['vocab','merges'] and original_json['model'].get(k)!=raw_state['model'].get(k)}
  model_default_differences['vocab_and_ordered_merge_pairs_exact']=True
  model_default_differences['merge_JSON_representation_identical']=original_json['model']['merges']==raw_state['model']['merges']
  specials={s:{'raw':raw.token_to_id(s),'generic':generic.convert_tokens_to_ids(s),'generic_encoded':generic.encode(s,add_special_tokens=False),'raw_encoded':raw.encode(s,add_special_tokens=False).ids} for s in sorted(set([content('bos_token'),content('eos_token'),content('pad_token'),'<｜User｜>','<｜Assistant｜>','<think>','</think>']))};assert all(x['raw']==x['generic'] and x['raw_encoded']==x['generic_encoded']==[x['raw']] for x in specials.values())
  audits=[];encs=[];rendered_all=[]
  for i,(row,prior) in enumerate(zip(rows,old)):
   assert set(row)=={'row_id','pmc_id','code_map','patient_ids','messages'} and row['row_id']==prior['row_id']
   text=generic.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True);expected=content('bos_token')+row['messages'][0]['content']+'<｜User｜>'+row['messages'][1]['content']+'<｜Assistant｜>';assert text==expected and hashlib.sha256(text.encode()).hexdigest()==prior['native_prompt_sha256']
   rid=raw.encode(text,add_special_tokens=False).ids;gid=generic.encode(text,add_special_tokens=False);native=generic.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
   assert rid==gid==native and raw.decode(rid,skip_special_tokens=False)==generic.decode(gid,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text
   assert raw.encode(text,add_special_tokens=True).ids==generic.encode(text,add_special_tokens=True)
   assert rid[0]==generic.bos_token_id==cfg_original['bos_token_id'] and rid[-1]==specials['<｜Assistant｜>']['raw'] and generic.eos_token_id==cfg_original['eos_token_id']
   audit={'index':i,'row_id':row['row_id'],'pmc_id':row['pmc_id'],'native_input_tokens':len(rid),'native_ids_sha256':encoded_sha(rid),'native_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'original_row_sha256':hashlib.sha256(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'raw_generic_native_IDs_exact':True,'both_roundtrip_exact':True,'special_prefix_exact':True,'prior_TF5_reconstructed_input_tokens':prior['native_input_tokens'],'prior_ids_equal':encoded_sha(rid)==prior['native_ids_sha256']};audits.append(audit);encs.append(rid);rendered_all.append(text)
  longest=max(range(80),key=lambda i:len(encs[i]));selected=list(dict.fromkeys([0,longest]));assert len(selected) in [1,2];roles=['original_first']+(['longest_published_length_first_original_order_tie'] if len(selected)==2 else [])
  with (out/'selected-prompts.jsonl').open('x') as f:
   for i in selected:f.write(json.dumps(rows[i],ensure_ascii=False)+'\n')
  write(out/'selected-encoded.json',{'policy':'original first then longest published input; earliest original row wins ties; deduplicate if same','rows':[dict(audits[i],selection_role=role,input_ids=encs[i],rendered_prompt=rendered_all[i]) for i,role in zip(selected,roles)]})
  write(out/'all80-input-audit.json',audits)
  encoded_raw=''.join(json.dumps({'row_id':r['row_id'],'input_ids':ids},separators=(',',':'))+'\n' for r,ids in zip(rows,encs)).encode();(out/'all80-published-encoded.jsonl.gz').write_bytes(gzip.compress(encoded_raw,mtime=0))
  # Explicit recognized HF mapping of the fixed vLLM formula, preserving original config file bytes.
  derived=copy.deepcopy(cfg_original);derived['rope_scaling']['attention_factor']=1.0;write(out/'derived-config.json',derived);cfg=AutoConfig.from_pretrained(str(out/'derived-config.json'),local_files_only=True,trust_remote_code=False);fn=ROPE_INIT_FUNCTIONS['yarn'];inv,scale=fn(cfg,device=torch.device('cpu'));assert inv.dtype==torch.float32 and tuple(inv.shape)==(64,) and scale==1.0;freq_sha=hashlib.sha256(inv.numpy().tobytes()).hexdigest();assert freq_sha=='427ed49dc1d6e18683800eca336467c70fc6817bfdae8a61166cbc7e01b05cb1';assert sha(model/'config.json')==plan['refs']['config.json']['sha256']
  write(out/'effective-config.json',cfg.to_dict());assert not torch.cuda.is_initialized()
  outputs={n:ref(out/n) for n in ['selected-prompts.jsonl','selected-encoded.json','all80-input-audit.json','all80-published-encoded.jsonl.gz','derived-config.json','effective-config.json']}
  result.update(status='PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT',documents=len({r['pmc_id'] for r in rows}),rows=80,all80_raw_generic_native_IDs_exact=True,all80_roundtrip_exact=True,specials=specials,all_raw_generic_components_exact=component_equal,original_JSON_model_defaults_normalized_on_native_load=model_default_differences,selected=[dict(audits[i],selection_role=role) for i,role in zip(selected,roles)],selected_indices=selected,longest_tied_indices=[i for i in range(80) if len(encs[i])==len(encs[longest])],native_lengths={'min':min(map(len,encs)),'max':max(map(len,encs)),'median':sorted(map(len,encs))[40],'total':sum(map(len,encs))},old_TF5_ids_equal_count=sum(x['prior_ids_equal'] for x in audits),raw_model_backend_SHA=hashlib.sha256(json.dumps(raw_state['model'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),generic_class=type(generic).__name__,generic_class_source_path=inspect.getsourcefile(type(generic)),generic_class_source_SHA=sha(inspect.getsourcefile(type(generic))),published_decoder=original_json['decoder'],derived_config_policy='Original config bytes retained; independent file adds only rope_scaling.attention_factor=1.0 from fixed vLLM native formula. Not a single-variable causal experiment.',native_CPU_inv_freq_bytes_SHA=freq_sha,native_CPU_attention_scaling=scale,outputs=outputs,CUDA_initialized=False,no_model_weight_read_GPU_quality=True,runtime={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__})
 except BaseException as e:result.update(status='FAILED_PUBLISHED_CPU_CONTRACT_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
 finally:result['program_seconds']=time.monotonic()-st;write(out/'result.json',result);print(json.dumps({k:result.get(k) for k in ['status','program_seconds','native_lengths','selected_indices','error']}))
if __name__=='__main__':main()
