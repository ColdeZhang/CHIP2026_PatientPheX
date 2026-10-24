#!/usr/bin/env python3
"""Same published R1 raw/generic/template contract, now on all15 train-only inputs."""
import argparse,datetime,gzip,hashlib,json,os,time,traceback
from pathlib import Path

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);z=a.parse_args();p=Path(z.plan);plan=json.loads(p.read_text());out=p.parent/'actual';out.mkdir(exist_ok=False);start=time.monotonic();r=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(p),source=ref(__file__))
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']==''
  assert ref(__file__)['sha256']==plan['source_sha256']
  for item in plan['references'].values():assert ref(item['path'])==item
  rawbytes=gzip.decompress(Path(plan['references']['inputs_gz']['path']).read_bytes());assert hashlib.sha256(rawbytes).hexdigest()==plan['inputs_uncompressed_sha256']
  rows=[json.loads(l) for l in rawbytes.splitlines()];assert len(rows)==15 and len({x['pmc_id'] for x in rows})==15
  from tokenizers import Tokenizer
  import tokenizers,transformers,torch
  from transformers import PreTrainedTokenizerFast
  assert not torch.cuda.is_initialized()
  tc=json.loads(Path(plan['references']['tokenizer_config']['path']).read_text())
  def content(k):
   x=tc.get(k);return x.get('content') if isinstance(x,dict) else x
  raw=Tokenizer.from_file(plan['references']['tokenizer']['path']);g=PreTrainedTokenizerFast(tokenizer_file=plan['references']['tokenizer']['path'],bos_token=content('bos_token'),eos_token=content('eos_token'),pad_token=content('pad_token'),unk_token=content('unk_token'),chat_template=tc['chat_template'],clean_up_tokenization_spaces=False,model_max_length=tc['model_max_length'])
  rs=json.loads(raw.to_str());gs=json.loads(g.backend_tokenizer.to_str());assert all(rs.get(k)==gs.get(k) for k in ('model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens'))
  audits=[];encoded=[]
  for row in rows:
   assert row['train_label_conditioned'] is True and [m['role'] for m in row['messages']]==['system','user']
   text=g.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
   expected=content('bos_token')+row['messages'][0]['content']+'<｜User｜>'+row['messages'][1]['content']+'<｜Assistant｜>';assert text==expected
   a=raw.encode(text,add_special_tokens=False).ids;b=g.encode(text,add_special_tokens=False);c=g.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False);assert a==b==c
   assert raw.decode(a,skip_special_tokens=False)==g.decode(a,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text
   assert a[0]==151643 and a[-1]==151670 and g.eos_token_id==151645
   assert 151667 not in a and 151668 not in a
   audits.append(dict(row_id=row['row_id'],input_tokens=len(a),input_ids_sha256=digest(a),rendered_sha256=hashlib.sha256(text.encode()).hexdigest()))
   encoded.append(dict(row_id=row['row_id'],input_ids=a))
  payload=''.join(json.dumps(x,separators=(',',':'))+'\n' for x in encoded).encode();ep=out/'encoded15.jsonl.gz'
  with ep.open('xb') as f:f.write(gzip.compress(payload,mtime=0));f.flush();os.fsync(f.fileno())
  write(out/'audit.json',audits)
  assert not torch.cuda.is_initialized()
  r.update(status='PASS_ALL15_RAW_GENERIC_TEMPLATE_IDS_ROUNDTRIP',rows=15,input_tokens_total=sum(x['input_tokens'] for x in audits),input_tokens_min=min(x['input_tokens'] for x in audits),input_tokens_max=max(x['input_tokens'] for x in audits),max_total_with3072=max(x['input_tokens'] for x in audits)+3072,old_full80_max_input=5689,old_context_exceeded=sum(x['input_tokens']>5689 for x in audits),rows_audit=audits,artifacts={'encoded':ref(ep),'audit':ref(out/'audit.json')},runtime={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},model_weight_reads=0,model_calls=0,CUDA_initialized=False,held_gold_reads=0)
 except BaseException as e:r.update(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc());raise
 finally:r['seconds']=time.monotonic()-start;write(out/'result.json',r);print(json.dumps(r))
if __name__=='__main__':main()
