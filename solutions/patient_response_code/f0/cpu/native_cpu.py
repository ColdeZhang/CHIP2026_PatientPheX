"""Complete F0 pure-input native census; same unchanged PT codec, no model/gold."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,time
os.environ.update(CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from response_codec import transform
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
 b=gzip.decompress((B/'held.original.jsonl.gz').read_bytes());assert sha(b)==p['groups']['held']['source']['sha256'];old=[json.loads(l) for l in b.splitlines()];assert len(old)==58 and all('answer' not in r for r in old);new=[transform(r) for r in old]
 newbytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new).encode();assert sha(newbytes)==p['groups']['held']['output']['sha256']
 codes=sorted({c for r in new for c in r['response_patient_codes']});code_ids={c:native.encode(c,add_special_tokens=False).ids for c in codes};assert all(native.decode(v,skip_special_tokens=False)==k for k,v in code_ids.items())
 stats={};audit=[]
 for arm,rows in [('original',old),('response_code',new)]:
  lengths=[];encoded=[]
  for i,row in enumerate(rows):
   prefix=auto.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=auto.encode(prefix,add_special_tokens=False)
   assert ids==native.encode(prefix,add_special_tokens=False).ids==auto.apply_chat_template(row['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
   assert auto.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(ids,skip_special_tokens=False)==prefix
   lengths.append(len(ids));encoded.append(dict(row_id=row['row_id'],input_ids=ids));audit.append(dict(group='held',arm=arm,index=i,row_id=row['row_id'],pmc_id=str(row['pmc_id']),prefix_tokens=len(ids),prefix_ids_sha256=jsha(ids),published_native_Auto_template_decode_exact=True))
  stats[arm]=dict(rows=len(rows),total_tokens=sum(lengths),max_tokens=max(lengths),longest_indices=[i for i,n in enumerate(lengths) if n==max(lengths)],supervised_tokens=0,sum_squared_lengths=sum(n*n for n in lengths),over_caps={str(c):sum(n>c for n in lengths) for c in [14848,16384,16896,32768]},proposed_512_rejection_cap=((max(lengths)+511)//512)*512,encoded_sha256=jsha(encoded))
 assert stats['original']['total_tokens']==505090 and stats['original']['max_tokens']==16245
 context=json.loads((a.base/'config.json').read_text())['text_config']['max_position_embeddings'];assert stats['response_code']['max_tokens']+1600<=context
 save(B/'native-row-audit.json',audit)
 result=dict(status='PASS_NATIVE_CPU_RESPONSE_CODE_F0_CENSUS_NO_MODEL',groups={'held':dict(arms=stats,response_code_ids=code_ids,source_sha256=p['groups']['held']['source']['sha256'],response_data_sha256=p['groups']['held']['output']['sha256'])},context_tokens=context,assets=p['tokenizer_files'],runtime=p['runtime'],cuda_initialized=torch.cuda.is_initialized(),truncation=False,model_loaded=False,training_performed=False,held_gold_read=False,source_sha256=sha(Path(__file__).read_bytes()),plan_sha256=sha(a.plan.read_bytes()),audit_sha256=sha((B/'native-row-audit.json').read_bytes()),seconds=time.monotonic()-start)
 assert not result['cuda_initialized'];save(B/'native-result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
