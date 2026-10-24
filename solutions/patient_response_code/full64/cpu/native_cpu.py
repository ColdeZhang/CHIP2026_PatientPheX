"""One CPU native-tokenizer census of frozen short265/full64 response-code data."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,statistics,time
os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from response_codec import transform,decode_response
from qwen35_smoke import encode
sha=lambda b:hashlib.sha256(b).hexdigest()
jsha=lambda x:sha(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);a=ap.parse_args();B=a.plan.parent;p=json.loads(a.plan.read_text());start=time.monotonic()
 assert p['CPU_only'] and not p['training_authorized'];assert not (B/'native-intent.json').exists()
 for name,r in p['files'].items():b=(B/name).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
 for name,r in p['tokenizer_files'].items():b=(a.base/name).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
 save(B/'native-intent.json',dict(pid=os.getpid(),started_unix=time.time(),source_sha256=sha(Path(__file__).read_bytes()),plan_sha256=sha(a.plan.read_bytes()),model_loaded=False))
 import torch,transformers,tokenizers
 assert {k:v for k,v in zip(['torch','transformers','tokenizers'],[torch.__version__,transformers.__version__,tokenizers.__version__])}==p['runtime']
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 auto=transformers.AutoTokenizer.from_pretrained(a.base,local_files_only=True);native=tokenizers.Tokenizer.from_file(str(a.base/'tokenizer.json'));native.no_truncation();native.no_padding()
 outputs={};audit=[];encoded_groups={}
 for group in ['train','held']:
  b=gzip.decompress((B/f'{group}.original.jsonl.gz').read_bytes());assert sha(b)==p['groups'][group]['source']['sha256'];old=[json.loads(l) for l in b.splitlines()];new=[transform(r) for r in old]
  newbytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new).encode();assert sha(newbytes)==p['groups'][group]['output']['sha256']
  codes=sorted({c for r in new for c in r['response_patient_codes']});code_ids={c:native.encode(c,add_special_tokens=False).ids for c in codes};assert all(native.decode(v,skip_special_tokens=False)==k for k,v in code_ids.items())
  stats={};encoded_groups[group]={}
  for arm,rows in [('original',old),('response_code',new)]:
   lengths=[];sup=[];encoded=[]
   for i,row in enumerate(rows):
    messages=row['messages'];prefix=auto.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix_ids=auto.encode(prefix,add_special_tokens=False);assert prefix_ids==native.encode(prefix,add_special_tokens=False).ids==auto.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    assert auto.decode(prefix_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(prefix_ids,skip_special_tokens=False)==prefix
    record=dict(group=group,arm=arm,index=i,row_id=row['row_id'],pmc_id=str(row['pmc_id']),prefix_tokens=len(prefix_ids),prefix_ids_sha256=jsha(prefix_ids),published_native_Auto_template_decode_exact=True)
    if group=='train':
     e=encode(row,auto);full=auto.apply_chat_template(messages+[dict(role='assistant',content=row['answer'])],tokenize=False,add_generation_prompt=False,enable_thinking=False)
     assert native.encode(full,add_special_tokens=False).ids==e['input_ids'];assert native.decode(e['input_ids'],skip_special_tokens=False)==auto.decode(e['input_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)==full
     assert e['input_ids'][:e['prefix_tokens']]==prefix_ids and all(v==-100 for v in e['labels'][:e['prefix_tokens']]) and e['labels'][e['prefix_tokens']:]==e['input_ids'][e['prefix_tokens']:]
     lengths.append(len(e['input_ids']));sup.append(e['supervised_tokens']);encoded.append(e)
     record.update(full_tokens=len(e['input_ids']),supervised_tokens=e['supervised_tokens'],input_ids_sha256=jsha(e['input_ids']),labels_sha256=jsha(e['labels']),eos_supervised_position=e['eos_supervised_position'],prefix_mask_exact=True)
    else:lengths.append(len(prefix_ids));encoded.append(dict(row_id=row['row_id'],input_ids=prefix_ids))
    audit.append(record)
   stats[arm]=dict(rows=len(rows),total_tokens=sum(lengths),max_tokens=max(lengths),longest_indices=[i for i,n in enumerate(lengths) if n==max(lengths)],max_response_or_training_tokens=sum(sup) if group=='train' else None,supervised_tokens=sum(sup) if group=='train' else 0,max_supervised_tokens=max(sup) if sup else None,sum_squared_lengths=sum(n*n for n in lengths),over_caps={str(c):sum(n>c for n in lengths) for c in [6144,13824,14400,14848,16384,32768]},proposed_512_rejection_cap=((max(lengths)+511)//512)*512,encoded_sha256=jsha(encoded))
   encoded_groups[group][arm]=encoded
  outputs[group]=dict(arms=stats,response_code_ids=code_ids,source_sha256=p['groups'][group]['source']['sha256'],response_data_sha256=p['groups'][group]['output']['sha256'])
 assert outputs['train']['arms']['original']['rows']==265 and outputs['held']['arms']['original']['rows']==64
 assert outputs['held']['arms']['original']['total_tokens']==512091 and outputs['held']['arms']['original']['max_tokens']==13230
 # All nondominated real training (full input length, supervised length) shapes.
 shapes=[x for x in audit if x['group']=='train' and x['arm']=='response_code']
 pareto=[x for x in shapes if not any(y['full_tokens']>=x['full_tokens'] and y['supervised_tokens']>=x['supervised_tokens'] and (y['full_tokens']>x['full_tokens'] or y['supervised_tokens']>x['supervised_tokens']) for y in shapes)]
 outputs['train']['shape_pareto']=[{k:x[k] for k in ['index','row_id','full_tokens','supervised_tokens','input_ids_sha256','labels_sha256']} for x in pareto]
 outputs['train']['max_supervised_indices']=[x['index'] for x in shapes if x['supervised_tokens']==max(y['supervised_tokens'] for y in shapes)]
 outputs['train']['future_schedule_metadata_only']=dict(epochs=2,batch_size=1,accumulate=8,steps_per_epoch=34,tail_microbatches=1,total_updates=68,warmup_updates=4,seed=20261023,fresh_base_required=True,training_not_authorized=True)
 context=json.loads((a.base/'config.json').read_text())['text_config']['max_position_embeddings'];assert all(outputs[g]['arms']['response_code']['max_tokens']+(1600 if g=='held' else 0)<=context for g in outputs)
 save(B/'native-row-audit.json',audit)
 result=dict(status='PASS_NATIVE_CPU_RESPONSE_CODE_FULL64_CENSUS_NO_MODEL',groups=outputs,context_tokens=context,assets=p['tokenizer_files'],runtime=p['runtime'],cuda_initialized=torch.cuda.is_initialized(),truncation=False,model_loaded=False,training_performed=False,held_gold_read=False,source_sha256=sha(Path(__file__).read_bytes()),plan_sha256=sha(a.plan.read_bytes()),audit_sha256=sha((B/'native-row-audit.json').read_bytes()),seconds=time.monotonic()-start)
 assert not result['cuda_initialized'];save(B/'native-result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
