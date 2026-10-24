"""Conditional originaldev fulltext native tokenizer census, hidden CUDA, no NN."""
from pathlib import Path
import argparse,hashlib,json,os,time

def ref(p):
 b=Path(p).read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def jsha(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--model',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();beg=time.monotonic();out=Path(a.output);out.mkdir(exist_ok=False);assert os.environ['CUDA_VISIBLE_DEVICES']==''
 import torch,transformers,tokenizers
 from transformers import AutoTokenizer
 assert not torch.cuda.is_initialized();torch.set_num_threads(1);model=Path(a.model);adapter=Path(a.adapter)
 expected={'tokenizer.json':'5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42','chat_template.jinja':'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715','tokenizer_config.json':'316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'};assets={n:ref(model/n) for n in expected};assert all(assets[n]['sha256']==v for n,v in expected.items())
 adapter_assets={n:ref(adapter/n) for n in ['adapter_model.safetensors','adapter_config.json']};assert adapter_assets['adapter_model.safetensors']['sha256']=='1a703b244b17c1378b383e4d6feefe2486b1a888e78cbd6f9f25c3fa6e46f45b';assert adapter_assets['adapter_config.json']['sha256']=='c4a478a7dd1aa65613279cdd4250da344ca16df8605de4249f445a1de2e8f2e2'
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);native=tokenizers.Tokenizer.from_file(str(model/'tokenizer.json'));assert type(tok).__name__=='Qwen3_5Tokenizer'
 rows=[json.loads(x) for x in Path(a.input).read_text().splitlines()];assert len(rows)==64 and all('answer' not in r for r in rows);audit=[]
 for i,r in enumerate(rows):
  rendered=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False);native_ids=native.encode(rendered,add_special_tokens=False).ids;direct=tok.apply_chat_template(r['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False);assert ids==native_ids==direct
  decoded=native.decode(ids,skip_special_tokens=False);normalized=native.normalizer.normalize_str(rendered) if native.normalizer else rendered;assert decoded==normalized and tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==decoded
  audit.append({'index':i,'row_id':r['row_id'],'expanded_row_sha256':jsha(r),'expanded':{'tokens':len(ids),'input_ids_sha256':jsha(ids),'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest()},'native_ids_exact':True,'published_normalizer_roundtrip_exact':True})
 lengths=[r['expanded']['tokens'] for r in audit];assert not torch.cuda.is_initialized();(out/'row-audit.json').write_text(json.dumps(audit,indent=2)+'\n');result={'status':'PASS_CONDITIONAL_FULL64_COMPLETE_INPUT_NATIVE_CPU_NO_GPU_GOLD','rows':64,'total_input_tokens':sum(lengths),'max_input_tokens':max(lengths),'min_input_tokens':min(lengths),'longest_original_index':lengths.index(max(lengths)),'two_seed_fresh_input_tokens':2*sum(lengths),'fresh_generation_rows_if_later_authorized':128,'input':ref(a.input),'tokenizer_assets':assets,'adapter_assets':adapter_assets,'audit':ref(out/'row-audit.json'),'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},'no_CUDA_initialized':True,'new_NN_calls':0,'gold_read':False,'quality_calls':0,'elapsed_seconds':time.monotonic()-beg};(out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
