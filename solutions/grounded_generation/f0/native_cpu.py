"""Full47 actual A800 native input replay; no model weights, gold or CUDA."""
import argparse,gzip,hashlib,importlib.util,json,os,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 p=p.resolve().parent
 while True:
  fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if p==p.parent:break
  p=p.parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 began=time.monotonic();p=json.loads(Path(a.plan).read_text());assert os.environ['CUDA_VISIBLE_DEVICES']==''
 for name,item in p['files'].items():assert ref(name)==dict(path=name,**item)
 out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
 import torch,transformers,peft,tokenizers
 from transformers import AutoTokenizer
 from tokenizers import Tokenizer
 import transformers.models.qwen3_5.modeling_qwen3_5 as native
 runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__,modeling=ref(native.__file__),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=torch.cuda.is_initialized(),cuda_available=torch.cuda.is_available(),no_model_weights=True)
 save(out/'runtime.json',runtime)
 assert [runtime[k] for k in ['torch','transformers','peft','tokenizers']]==['2.7.1+cu126','5.20.2','0.21.2','0.23.1']
 assert not runtime['cuda_initialized'] and not runtime['cuda_available'] and importlib.util.find_spec('fla') is None and importlib.util.find_spec('causal_conv1d') is None
 assert runtime['modeling']['sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
 model=Path(p['model']);source=json.loads(Path(p['model_source_manifest']).read_text());assert sha(p['model_source_manifest'])=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
 assets={v['name']:v for v in source['files']};checked={}
 for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json']:
  actual=ref(model/name);assert actual['bytes']==assets[name]['size'] and actual['sha256']==assets[name]['sha256'];checked[name]=actual
 save(out/'tokenizer-assets.json',checked)
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True);native_tok=Tokenizer.from_file(str(model/'tokenizer.json'))
 rows=[json.loads(x) for x in Path(p['held_input']).read_text().splitlines()];audit={r['row_id']:r for r in json.loads(Path(p['held_audit']).read_text())};expected={r['row_id']:r['input_ids'] for r in json.loads(gzip.decompress(Path(p['held_local_ids']).read_bytes()))}
 assert len(rows)==len(audit)==len(expected)==47 and len({r['pmc_id'] for r in rows})==16
 assert {r['row_id'] for r in rows}==set(audit)==set(expected) and len({(r['pmc_id'],r['target_patient_id']) for r in rows})==47
 observations=[]
 for row in rows:
  assert 'answer' not in row
  text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=tok.encode(text,add_special_tokens=False);native_ids=native_tok.encode(text,add_special_tokens=False).ids;audit_row=audit[row['row_id']]
  checks=dict(native_actual_local_IDs_equal=ids==native_ids==expected[row['row_id']],rendered_prefix_SHA_equal=hashlib.sha256(text.encode()).hexdigest()==audit_row['rendered_prefix_sha256'],all_IDs_SHA_equal=hashlib.sha256(json.dumps(ids).encode()).hexdigest()==audit_row['input_ids_sha256'],count_equal=len(ids)==audit_row['input_tokens'],native_actual_decode_equal=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native_tok.decode(native_ids,skip_special_tokens=False))
  observations.append(dict(row_id=row['row_id'],tokens=len(ids),checks=checks));assert all(checks.values()),row['row_id']
 save(out/'observations.json',observations)
 result=dict(status='PASS_NATIVE_COMPLETE47_F0_INPUT',plan=ref(a.plan),source=ref(__file__),runtime=ref(out/'runtime.json'),tokenizer_assets=ref(out/'tokenizer-assets.json'),observations=ref(out/'observations.json'),encoded=ref(p['held_local_ids']),rows=47,documents=16,all47_local_native_actual_IDs_rendering_decode_exact=True,input_tokens=sum(len(x) for x in expected.values()),max_input_tokens=max(map(len,expected.values())),no_weights_no_GPU_no_fit_no_quality=True,seconds=time.monotonic()-began,training_runtime_torch='2.6.0+cu124',inference_runtime_torch='2.7.1+cu126',same_native_backend_no_FLA=True)
 save(out/'result.json',result);print(json.dumps(ref(out/'result.json')),flush=True)
if __name__=='__main__':main()
